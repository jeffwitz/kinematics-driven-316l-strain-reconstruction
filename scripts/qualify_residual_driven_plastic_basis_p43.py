#!/usr/bin/env python3
"""A plastic basis built from the residual, not from the observability spectrum.

The observability modes answer "which plastic direction is most visible?". They
are ranked by singular gain, they came out concentrated at the crop border, and
the field reconstructed on them was uncorrelated with the localisation the DIC
shows. The question worth asking is the other one:

    which plastic direction would best correct the error actually observed?

That is answered by back-propagating the measured residual through the
equilibrium operator, `Phi_1 = A^T r`, and enriching with a Krylov sequence
`(A^T A)^k A^T r`, compressed by POD -- Krylov enriches, POD compresses, in the
spirit of Ryckelynck's a priori adaptive reduction. If the discrepancy is a band
in the middle of the image, `A^T r` is fed by that band, not by whichever edge
mode happens to have the largest gain.

The regularisation is then the **rank**, not a Tikhonov weight: no length scale
to choose, no assumption that plasticity is spatially smooth, and one number
that means "how many mechanically relevant plastic directions does it take to
explain the DIC".

Two notes on the physics.

**The deviatoric projector is the identity here.** Plastic incompressibility
fixes the out-of-plane component, `e33 = -(e11 + e22)`, but `e33` does not enter
the in-plane equilibrium under plane stress, so the three tracked components are
unconstrained. Imposing a projector would be a no-op, and imposing one on the
in-plane trace instead would be wrong.

**The metric is the strain of the residual, whitened pointwise.** A Dirichlet
residual is pinned to zero on the boundary, so displacement understates a narrow
band; and the whitener has to be the one that passed the held-out null test, not
the raw-DIC spectral one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fem_inhouse.core.constitutive import PLANE_STRESS_VON_MISES_METRIC, von_mises
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
    parser.add_argument("--states", nargs="+", type=int, default=[25, 30, 35, 40])
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--krylov-depth", type=int, default=10)
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

    def extension(field: np.ndarray) -> np.ndarray:
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

    def residual_strain(field: np.ndarray) -> np.ndarray:
        return np.asarray(
            operator.kinematics.strain(field - extension(field))
        ).reshape(strain_shape)

    # The whitener that passed the held-out null test, on independent patches.
    noise_source = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    canonical = np.asarray(
        image_flow_to_canonical(
            np.asarray(noise_source[:2100, :2100]), pixel_size_mm=PIXEL_SIZE_MM
        )
    )
    step = pixels + 1
    patches = [
        (row, column)
        for row in range(0, canonical.shape[0] - step, step)
        for column in range(0, canonical.shape[1] - step, step)
    ]
    samples = np.asarray(
        [
            residual_strain(
                np.ascontiguousarray(canonical[row : row + step, column : column + step, :])
            )
            for row, column in patches
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

    reference = history[arguments.reference_state]
    residuals = np.asarray(
        [
            whitener.apply(residual_strain(history[state] - reference)).reshape(-1)
            for state in arguments.states
        ]
    )

    def forward(plastic: np.ndarray) -> np.ndarray:
        stress = np.einsum("pi,pij->pj", plastic.reshape(-1, 3), operator.elasticity)
        displacement = unpack_interior(
            operator.solve_stiffness(operator._strain_transpose(stress.reshape(-1))), grid
        )
        strain = np.asarray(operator.kinematics.strain(displacement))
        return np.asarray(whitener.apply(strain.reshape(strain_shape))).reshape(-1)

    def backward(observation: np.ndarray) -> np.ndarray:
        dual = np.asarray(
            whitener.adjoint(observation.reshape(strain_shape)), dtype=np.float64
        ).reshape(-1, 3)
        displacement = unpack_interior(
            operator.solve_stiffness(operator._strain_transpose(dual.reshape(-1))), grid
        )
        strain = np.asarray(operator.kinematics.strain(displacement)).reshape(-1, 3)
        return np.einsum("pi,pij->pj", strain, operator.elasticity).reshape(-1)

    probe = np.random.default_rng(4).normal(size=operator.plastic_size)
    dual_probe = np.random.default_rng(5).normal(size=residuals.shape[1])
    adjoint_error = abs(
        float(forward(probe) @ dual_probe) - float(probe @ backward(dual_probe))
    ) / abs(float(forward(probe) @ dual_probe))

    # Krylov enrichment driven by the measured residual, all states at once.
    block = np.asarray([backward(row) for row in residuals]).T
    basis, _ = np.linalg.qr(block)
    columns = [basis]
    current = basis
    for _ in range(arguments.krylov_depth):
        current = np.asarray([backward(forward(current[:, k])) for k in range(current.shape[1])]).T
        stacked = np.concatenate([*columns, current], axis=1)
        orthonormal, _ = np.linalg.qr(stacked)
        current = orthonormal[:, -current.shape[1] :]
        columns.append(current)
    full = np.concatenate(columns, axis=1)
    full, _ = np.linalg.qr(full)

    # POD compression on the responses those directions produce.
    responses = np.asarray([forward(full[:, k]) for k in range(full.shape[1])]).T
    _, spectrum, right_transposed = np.linalg.svd(responses, full_matrices=False)
    modes = full @ right_transposed.T

    dic_strain = np.asarray(
        operator.kinematics.strain(history[arguments.states[-1]] - reference)
    ).reshape(pixels, pixels, 2, 3)
    dic_equivalent = von_mises(dic_strain).mean(axis=2)
    gauge = np.linalg.inv(PLANE_STRESS_VON_MISES_METRIC)

    baseline = float(np.linalg.norm(residuals))
    records = []
    for rank in arguments.ranks:
        if rank > modes.shape[1]:
            continue
        design = np.asarray([forward(modes[:, k]) for k in range(rank)]).T
        coefficients, *_ = np.linalg.lstsq(design, residuals.T, rcond=None)
        misfit = float(np.linalg.norm(design @ coefficients - residuals.T))
        plastic = (modes[:, :rank] @ coefficients[:, -1]).reshape(pixels, pixels, 2, 3)
        equivalent = np.sqrt(
            np.maximum(np.einsum("xyci,ij,xycj->xyc", plastic, gauge, plastic), 0.0)
        ).mean(axis=2)
        flat_model, flat_data = equivalent.reshape(-1), dic_equivalent.reshape(-1)
        top = flat_data >= np.quantile(flat_data, 0.9)
        records.append(
            {
                "rank": rank,
                "misfit_over_elastic_baseline": misfit / baseline,
                "peak_equivalent_plastic_strain": float(equivalent.max()),
                "correlation_with_dic_equivalent_strain": float(
                    np.corrcoef(flat_model, flat_data)[0, 1]
                ),
                "share_inside_the_dic_top_decile": float(
                    flat_model[top].sum() / flat_model.sum()
                ),
            }
        )
        entry = records[-1]
        print(
            f"rank {rank:3d} | misfit / elastic {entry['misfit_over_elastic_baseline']:.4f} | "
            f"peak p_eq {entry['peak_equivalent_plastic_strain']:.3e} | "
            f"corr {entry['correlation_with_dic_equivalent_strain']:+.3f} | "
            f"top decile {entry['share_inside_the_dic_top_decile']:.3f}"
        )

    output = {
        "schema_version": 1,
        "pixels": pixels,
        "origin_nodes": [x0, y0],
        "reference_state": arguments.reference_state,
        "states": arguments.states,
        "adjoint_relative_error": adjoint_error,
        "deviatoric_projector": "identity: e33 absorbs the trace and does not enter plane stress",
        "krylov_depth": arguments.krylov_depth,
        "basis_size": int(modes.shape[1]),
        "pod_spectrum": spectrum.tolist(),
        "dic_peak_equivalent_strain": float(dic_equivalent.max()),
        "unstructured_top_decile_share": 0.1,
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", "utf-8")
    print(
        f"\nadjoint {adjoint_error:.2e}   basis {modes.shape[1]}   "
        f"DIC peak equivalent strain {dic_equivalent.max():.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
