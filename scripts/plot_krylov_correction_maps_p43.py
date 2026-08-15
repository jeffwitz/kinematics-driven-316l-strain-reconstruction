#!/usr/bin/env python3
"""Maps of the J2 equivalent strain, DIC against simulation, mode by mode.

The trade-off curve says how much of the discrepancy a plastic correction
removes. It does not say whether the corrected field *looks* like the measured
one, and that is what decides whether the reconstruction means anything.

This draws, at state 40 referenced to state 20 and for a growing number of LSQR
iterations:

* the von Mises equivalent of the measured strain increment;
* the same for the simulation, elastic extension plus plastic correction;
* their difference on a shared colour scale;
* the equivalent plastic field the correction is made of.

Everything is in the native two-sub-cell layout, averaged over the two
sub-cells only for display, and no interpolation is performed anywhere.

The forward operator here is the **physical** one, without the whitener: the
whitening belongs to the inverse problem, not to the strain being looked at.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr

from fem_inhouse.core.constitutive import PLANE_STRESS_VON_MISES_METRIC, von_mises
from fem_inhouse.identification.pointwise_whitening import (
    PointwiseFieldWhitener,
)
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.measurement import image_flow_to_canonical
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior

ROOT = Path(__file__).resolve().parents[1]
NOISE = (
    ROOT
    / "validation/reference_data/dic_uncertainty_propagation_p0043_v1"
    / "centred_repeat_flow_pixels.npy"
)
HISTORY = (
    ROOT
    / "validation/reference_data/dic_multistep_history_p0043_repaired_v1"
    / "repaired_history_mm.npy"
)
PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30


class _Identity:
    def apply(self, values):
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values):
        return np.asarray(values, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", nargs=2, type=int, default=(1580, 1030))
    parser.add_argument("--pixels", type=int, default=100)
    parser.add_argument("--reference-state", type=int, default=20)
    parser.add_argument("--state", type=int, default=40)
    parser.add_argument("--iterations", nargs="+", type=int, default=[8, 32, 64, 128, 512])
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    pixels = arguments.pixels
    x0, y0 = arguments.origin
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    operator = TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=_Identity(),
        whitener=_Identity(),
    )
    strain_shape = (pixels, pixels, 6)

    def extension(field):
        strain = np.asarray(operator.kinematics.strain(field)).reshape(-1, 3)
        stress = np.einsum("pi,pij->pj", strain, operator.elasticity)
        forcing = (
            -pack_interior(
                operator.kinematics.divergence_from_sample_stress(
                    stress.reshape((pixels, pixels, 2, 3))
                )
            )
            / operator.quadrature_weight
        )
        result = field.copy()
        result[1:-1, 1:-1, :] -= operator.solve_stiffness(forcing).reshape(
            pixels - 1, pixels - 1, 2
        )
        return result

    def physical_response(plastic):
        """`A` without the whitener: the strain a plastic field actually produces."""

        stress = np.einsum(
            "pi,pij->pj", np.asarray(plastic).reshape(-1, 3), operator.elasticity
        )
        displacement = unpack_interior(
            operator.solve_stiffness(operator._strain_transpose(stress.reshape(-1))), grid
        )
        return np.asarray(operator.kinematics.strain(displacement)).reshape(
            pixels, pixels, 2, 3
        )

    noise_source = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    canonical = np.asarray(
        image_flow_to_canonical(
            np.asarray(noise_source[:2100, :2100]), pixel_size_mm=PIXEL_SIZE_MM
        )
    )
    step = pixels + 1
    samples = np.asarray(
        [
            np.asarray(
                operator.kinematics.strain(
                    (patch := np.ascontiguousarray(canonical[r : r + step, c : c + step, :]))
                    - extension(patch)
                )
            ).reshape(strain_shape)
            for r in range(0, canonical.shape[0] - step, step)
            for c in range(0, canonical.shape[1] - step, step)
        ]
    )
    whitener = PointwiseFieldWhitener.fit(samples)

    report = json.loads((HISTORY.with_name("report.json")).read_text(encoding="utf-8"))
    bounds = list(map(int, report["solve_bounds"]))
    source = np.load(HISTORY, mmap_mode="r", allow_pickle=False)
    history = np.asarray(
        source[
            :,
            x0 - bounds[0] : x0 + pixels - bounds[0] + 1,
            y0 - bounds[2] : y0 + pixels - bounds[2] + 1,
            :,
        ],
        dtype=np.float64,
    )
    increment = history[arguments.state] - history[arguments.reference_state]
    elastic = extension(increment)
    measured_strain = np.asarray(operator.kinematics.strain(increment)).reshape(
        pixels, pixels, 2, 3
    )
    elastic_strain = np.asarray(operator.kinematics.strain(elastic)).reshape(
        pixels, pixels, 2, 3
    )

    def forward(plastic):
        return whitener.apply(
            physical_response(plastic).reshape(strain_shape)
        ).reshape(-1)

    def backward(observation):
        dual = np.asarray(
            whitener.adjoint(np.asarray(observation).reshape(strain_shape)), dtype=np.float64
        ).reshape(-1, 3)
        displacement = unpack_interior(
            operator.solve_stiffness(operator._strain_transpose(dual.reshape(-1))), grid
        )
        strain = np.asarray(operator.kinematics.strain(displacement)).reshape(-1, 3)
        return np.einsum("pi,pij->pj", strain, operator.elasticity).reshape(-1)

    action = LinearOperator(
        (int(np.prod(strain_shape)), operator.plastic_size),
        matvec=forward,
        rmatvec=backward,
        dtype=np.float64,
    )
    target = whitener.apply(
        (measured_strain - elastic_strain).reshape(strain_shape)
    ).reshape(-1)

    gauge = np.linalg.inv(PLANE_STRESS_VON_MISES_METRIC)
    measured_equivalent = von_mises(measured_strain).mean(axis=2)
    elastic_equivalent = von_mises(elastic_strain).mean(axis=2)

    arguments.output.mkdir(parents=True, exist_ok=True)
    rows = len(arguments.iterations) + 1
    figure, axes = plt.subplots(rows, 4, figsize=(15.0, 3.4 * rows), constrained_layout=True)
    strain_scale = float(np.quantile(measured_equivalent, 0.995))
    difference_scale = strain_scale * 0.5

    def show(axis, field, title, scale, cmap="viridis", centred=False):
        if centred:
            image = axis.imshow(field.T, origin="lower", cmap="coolwarm",
                                vmin=-scale, vmax=scale)
        else:
            image = axis.imshow(field.T, origin="lower", cmap=cmap, vmin=0.0, vmax=scale)
        axis.set_title(title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046)

    show(axes[0, 0], measured_equivalent, "DIC, equivalent strain 20 to 40", strain_scale)
    show(axes[0, 1], elastic_equivalent, "elastic extension only", strain_scale)
    show(
        axes[0, 2],
        measured_equivalent - elastic_equivalent,
        "DIC minus elastic",
        difference_scale,
        centred=True,
    )
    axes[0, 3].axis("off")
    axes[0, 3].text(
        0.0,
        0.5,
        "row 0: the gap to close.\n\n"
        "Below, one row per LSQR\n"
        "iteration count: the\n"
        "simulation, its difference\n"
        "to the DIC, and the\n"
        "plastic field responsible.",
        fontsize=9,
        va="center",
    )

    records = []
    for index, limit in enumerate(arguments.iterations, start=1):
        solution = lsqr(action, target, iter_lim=limit, atol=0.0, btol=0.0, conlim=0.0)[0]
        correction = physical_response(solution)
        simulated = elastic_strain + correction
        simulated_equivalent = von_mises(simulated).mean(axis=2)
        plastic = solution.reshape(pixels, pixels, 2, 3)
        plastic_equivalent = np.sqrt(
            np.maximum(np.einsum("xyci,ij,xycj->xyc", plastic, gauge, plastic), 0.0)
        ).mean(axis=2)
        difference = measured_equivalent - simulated_equivalent
        relative = float(
            np.linalg.norm(simulated - measured_strain) / np.linalg.norm(
                elastic_strain - measured_strain
            )
        )
        records.append(
            {
                "iterations": limit,
                "relative_strain_error": relative,
                "plastic_peak": float(plastic_equivalent.max()),
                "plastic_rms": float(np.sqrt((plastic_equivalent**2).mean())),
                "equivalent_map_correlation": float(
                    np.corrcoef(
                        simulated_equivalent.reshape(-1), measured_equivalent.reshape(-1)
                    )[0, 1]
                ),
            }
        )
        show(axes[index, 0], simulated_equivalent, f"simulation, {limit} iterations", strain_scale)
        show(
            axes[index, 1],
            difference,
            f"DIC minus simulation  (rel {relative:.3f})",
            difference_scale,
            centred=True,
        )
        show(
            axes[index, 2],
            plastic_equivalent,
            f"plastic correction, peak {plastic_equivalent.max():.2e}",
            float(np.quantile(plastic_equivalent, 0.995)),
            cmap="magma",
        )
        show(
            axes[index, 3],
            np.abs(difference),
            "absolute difference",
            difference_scale,
            cmap="inferno",
        )

    figure.suptitle(
        f"P43 M100 crop ({x0}, {y0}), state {arguments.state} referenced to "
        f"{arguments.reference_state}: DIC against simulation as Krylov directions are added",
        fontsize=11,
    )
    figure.savefig(arguments.output / "krylov_correction_maps.png", dpi=130)
    plt.close(figure)

    # A second, compact figure: the trade-off the maps illustrate.
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    iterations = [entry["iterations"] for entry in records]
    left.plot(iterations, [entry["relative_strain_error"] for entry in records], "o-")
    left.set_xscale("log")
    left.set_xlabel("LSQR iterations")
    left.set_ylabel("remaining strain error, relative to elastic")
    left.grid(alpha=0.3)
    right.plot(
        [entry["plastic_rms"] for entry in records],
        [entry["relative_strain_error"] for entry in records],
        "o-",
    )
    for entry in records:
        right.annotate(
            str(entry["iterations"]),
            (entry["plastic_rms"], entry["relative_strain_error"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    right.set_xlabel("plastic RMS required")
    right.set_ylabel("remaining strain error")
    right.grid(alpha=0.3)
    figure.suptitle("What the correction costs, and what it buys", fontsize=11)
    figure.savefig(arguments.output / "krylov_tradeoff.png", dpi=130)
    plt.close(figure)

    (arguments.output / "report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "origin_nodes": [x0, y0],
                "state": arguments.state,
                "reference_state": arguments.reference_state,
                "dic_peak_equivalent_strain": float(measured_equivalent.max()),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"DIC peak equivalent strain (20 to {arguments.state}): {measured_equivalent.max():.4e}")
    print("\n iters | rel strain error | plastic RMS | plastic peak | map correlation")
    for entry in records:
        print(
            f" {entry['iterations']:5d} | {entry['relative_strain_error']:16.4f} | "
            f"{entry['plastic_rms']:11.3e} | {entry['plastic_peak']:12.3e} | "
            f"{entry['equivalent_map_correlation']:15.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
