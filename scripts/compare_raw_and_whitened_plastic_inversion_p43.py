#!/usr/bin/env python3
"""Whitened or raw objective? And which strain component fails?

The whitened inversion removes 96.5 % of the residual it minimises while leaving
78 % of the raw tensor strain difference. That is not a contradiction: the
whitener weights components by how improbable the noise model finds them, which
is right for **detecting** a signal and is not obviously right for
**reproducing** a strain field.

So the objective is separated from the noise information. Two inversions are run
on exactly the same operator, the same Krylov regularisation and the same
stopping points, differing only in whether `W` appears in the objective:

```text
raw       min || A z - r ||^2
whitened  min || W (A z - r) ||^2
```

Both are then scored in **both** metrics at every iteration, so the cost of the
statistical choice is measured rather than argued. The noise keeps its job --
saying when to stop -- and loses the other one, deciding which part of the field
is worth reproducing.

The maps are also broken down by component. A von Mises correlation of 0.889
alongside a tensor error of 0.643 is exactly what two fields with similar
equivalent magnitudes but different principal directions look like, so `e_xx`,
`e_yy` and `g_xy` are compared separately. Which component fails says which
constitutive freedom is missing, and that is the question SRIX and
Meric-Cailletaud will be asked next.
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

from fem_inhouse.core.constitutive import PLANE_STRESS_VON_MISES_METRIC
from fem_inhouse.identification.pointwise_whitening import PointwiseFieldWhitener
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
COMPONENTS = ("e_xx", "e_yy", "g_xy")


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
    parser.add_argument(
        "--iterations", nargs="+", type=int, default=[8, 32, 64, 128, 256, 512, 1024]
    )
    parser.add_argument("--map-iterations", nargs="+", type=int, default=[128, 512])
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
    shape = (pixels, pixels, 6)

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

    def response(plastic):
        stress = np.einsum(
            "pi,pij->pj", np.asarray(plastic).reshape(-1, 3), operator.elasticity
        )
        displacement = unpack_interior(
            operator.solve_stiffness(operator._strain_transpose(stress.reshape(-1))), grid
        )
        return np.asarray(operator.kinematics.strain(displacement)).reshape(-1)

    def transpose(observation):
        dual = np.asarray(observation, dtype=np.float64).reshape(-1, 3)
        displacement = unpack_interior(
            operator.solve_stiffness(operator._strain_transpose(dual.reshape(-1))), grid
        )
        strain = np.asarray(operator.kinematics.strain(displacement)).reshape(-1, 3)
        return np.einsum("pi,pij->pj", strain, operator.elasticity).reshape(-1)

    noise_source = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    canonical = np.asarray(
        image_flow_to_canonical(
            np.asarray(noise_source[:2100, :2100]), pixel_size_mm=PIXEL_SIZE_MM
        )
    )
    step = pixels + 1
    samples = []
    for row in range(0, canonical.shape[0] - step, step):
        for column in range(0, canonical.shape[1] - step, step):
            patch = np.ascontiguousarray(canonical[row : row + step, column : column + step, :])
            samples.append(
                np.asarray(
                    operator.kinematics.strain(patch - extension(patch))
                ).reshape(shape)
            )
    whitener = PointwiseFieldWhitener.fit(np.asarray(samples))

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
    measured = np.asarray(operator.kinematics.strain(increment)).reshape(-1)
    elastic = np.asarray(operator.kinematics.strain(extension(increment))).reshape(-1)
    target = measured - elastic

    def whiten(flat):
        return np.asarray(whitener.apply(flat.reshape(shape)), dtype=np.float64).reshape(-1)

    raw_action = LinearOperator(
        (target.size, operator.plastic_size), matvec=response, rmatvec=transpose, dtype=np.float64
    )
    whitened_action = LinearOperator(
        (target.size, operator.plastic_size),
        matvec=lambda z: whiten(response(z)),
        rmatvec=lambda y: transpose(
            np.asarray(whitener.adjoint(np.asarray(y).reshape(shape))).reshape(-1)
        ),
        dtype=np.float64,
    )
    raw_reference = float(np.linalg.norm(target))
    whitened_reference = float(np.linalg.norm(whiten(target)))
    gauge = np.linalg.inv(PLANE_STRESS_VON_MISES_METRIC)

    solutions: dict[str, dict[int, np.ndarray]] = {"raw": {}, "whitened": {}}
    curves: dict[str, list[dict[str, float]]] = {"raw": [], "whitened": []}
    for name, action, rhs in (
        ("raw", raw_action, target),
        ("whitened", whitened_action, whiten(target)),
    ):
        for limit in arguments.iterations:
            solution = lsqr(action, rhs, iter_lim=limit, atol=0.0, btol=0.0, conlim=0.0)[0]
            solutions[name][limit] = solution
            difference = response(solution) - target
            plastic = solution.reshape(pixels, pixels, 2, 3)
            equivalent = np.sqrt(
                np.maximum(np.einsum("xyci,ij,xycj->xyc", plastic, gauge, plastic), 0.0)
            ).mean(axis=2)
            per_component = {}
            reshaped = difference.reshape(pixels, pixels, 2, 3)
            base = target.reshape(pixels, pixels, 2, 3)
            for index, label in enumerate(COMPONENTS):
                per_component[label] = float(
                    np.linalg.norm(reshaped[..., index]) / np.linalg.norm(base[..., index])
                )
            curves[name].append(
                {
                    "iterations": limit,
                    "raw_error": float(np.linalg.norm(difference) / raw_reference),
                    "whitened_error": float(
                        np.linalg.norm(whiten(difference)) / whitened_reference
                    ),
                    "plastic_rms": float(np.sqrt((equivalent**2).mean())),
                    "plastic_peak": float(equivalent.max()),
                    "component_error": per_component,
                }
            )
        print(f"\n{name.upper()} objective")
        print("  iters | raw err | whitened err | p RMS      | e_xx  | e_yy  | g_xy")
        for entry in curves[name]:
            component = entry["component_error"]
            print(
                f"  {entry['iterations']:5d} | {entry['raw_error']:7.4f} | "
                f"{entry['whitened_error']:12.4f} | {entry['plastic_rms']:.3e} | "
                f"{component['e_xx']:5.3f} | {component['e_yy']:5.3f} | {component['g_xy']:5.3f}"
            )

    arguments.output.mkdir(parents=True, exist_ok=True)

    # Component maps, DIC against simulation, for each objective.
    measured_field = measured.reshape(pixels, pixels, 2, 3).mean(axis=2)
    elastic_field = elastic.reshape(pixels, pixels, 2, 3).mean(axis=2)
    for name in ("raw", "whitened"):
        for limit in arguments.map_iterations:
            if limit not in solutions[name]:
                continue
            simulated = (elastic + response(solutions[name][limit])).reshape(
                pixels, pixels, 2, 3
            ).mean(axis=2)
            figure, axes = plt.subplots(3, 4, figsize=(15.0, 10.0), constrained_layout=True)
            for index, label in enumerate(COMPONENTS):
                data = measured_field[..., index]
                model = simulated[..., index]
                base = elastic_field[..., index]
                scale = float(np.quantile(np.abs(data), 0.995))
                for column, (field, title) in enumerate(
                    (
                        (data, f"DIC  {label}"),
                        (model, f"simulation  {label}"),
                        (data - model, f"DIC minus simulation  {label}"),
                        (data - base, f"DIC minus elastic  {label}"),
                    )
                ):
                    axis = axes[index, column]
                    image = axis.imshow(
                        field.T, origin="lower", cmap="coolwarm", vmin=-scale, vmax=scale
                    )
                    axis.set_title(title, fontsize=9)
                    axis.set_xticks([])
                    axis.set_yticks([])
                    figure.colorbar(image, ax=axis, fraction=0.046)
            figure.suptitle(
                f"{name} objective, {limit} LSQR iterations — component by component",
                fontsize=12,
            )
            figure.savefig(arguments.output / f"components_{name}_{limit}.png", dpi=120)
            plt.close(figure)

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for name, style in (("raw", "o-"), ("whitened", "s--")):
        left.plot(
            [entry["iterations"] for entry in curves[name]],
            [entry["raw_error"] for entry in curves[name]],
            style,
            label=f"{name} objective",
        )
        right.plot(
            [entry["plastic_rms"] for entry in curves[name]],
            [entry["raw_error"] for entry in curves[name]],
            style,
            label=f"{name} objective",
        )
    left.set_xscale("log")
    left.set_xlabel("LSQR iterations")
    left.set_ylabel("raw tensor strain error")
    left.legend()
    left.grid(alpha=0.3)
    right.set_xlabel("plastic RMS required")
    right.set_ylabel("raw tensor strain error")
    right.legend()
    right.grid(alpha=0.3)
    figure.suptitle("What the whitened objective costs in raw strain agreement", fontsize=11)
    figure.savefig(arguments.output / "raw_against_whitened.png", dpi=130)
    plt.close(figure)

    (arguments.output / "report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "origin_nodes": [x0, y0],
                "state": arguments.state,
                "reference_state": arguments.reference_state,
                "note": (
                    "the noise keeps its job -- saying when to stop -- and loses the other, "
                    "deciding which part of the field is worth reproducing"
                ),
                "curves": curves,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
