#!/usr/bin/env python3
"""How far can a plastic field close the DIC gap, and at what amplitude?

The previous attempt built a Krylov space from the measured residual and then
compressed it with an SVD of the responses `A Phi`. That SVD reorders the
directions by singular gain -- reintroducing exactly the observability ranking
the residual-driven construction was meant to avoid, so the leading direction
was no longer `A^T r` and the rank-1 misfit was meaningless. The ordering is
removed here: LSQR is run on the same operator, and the iteration count is the
regularisation, as in any iterative regularisation of an ill-posed inverse.

Cutting at rank 32 was also arbitrary. The question is not how much a fixed
number of directions explains, it is the **trade-off**: how far the residual
falls, against how much plastic strain that costs. If the gap closes while the
field stays at the amplitude the experiment actually reached, plasticity is a
credible explanation; if closing it demands ten times that, it is not.

The reconstructed field is compared with the residual it is meant to explain,
not with the total DIC strain. The DIC measures `e = e_el + e_p` and does not
measure `p`, so a weak correlation with the total strain refutes nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    parser.add_argument(
        "--iterations", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    )
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
        return np.asarray(operator.kinematics.strain(field - extension(field))).reshape(
            strain_shape
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
            residual_strain(
                np.ascontiguousarray(canonical[row : row + step, column : column + step, :])
            )
            for row in range(0, canonical.shape[0] - step, step)
            for column in range(0, canonical.shape[1] - step, step)
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

    def forward(plastic):
        stress = np.einsum(
            "pi,pij->pj", np.asarray(plastic).reshape(-1, 3), operator.elasticity
        )
        displacement = unpack_interior(
            operator.solve_stiffness(operator._strain_transpose(stress.reshape(-1))), grid
        )
        strain = np.asarray(operator.kinematics.strain(displacement))
        return np.asarray(whitener.apply(strain.reshape(strain_shape))).reshape(-1)

    def backward(observation):
        dual = np.asarray(
            whitener.adjoint(np.asarray(observation).reshape(strain_shape)), dtype=np.float64
        ).reshape(-1, 3)
        displacement = unpack_interior(
            operator.solve_stiffness(operator._strain_transpose(dual.reshape(-1))), grid
        )
        strain = np.asarray(operator.kinematics.strain(displacement)).reshape(-1, 3)
        return np.einsum("pi,pij->pj", strain, operator.elasticity).reshape(-1)

    observation_size = int(np.prod(strain_shape))
    action = LinearOperator(
        (observation_size, operator.plastic_size),
        matvec=forward,
        rmatvec=backward,
        dtype=np.float64,
    )
    gauge = np.linalg.inv(PLANE_STRESS_VON_MISES_METRIC)

    results = {}
    for state in arguments.states:
        target = whitener.apply(residual_strain(history[state] - reference)).reshape(-1)
        norm = float(np.linalg.norm(target))
        residual_field = residual_strain(history[state] - reference)
        residual_amplitude = np.sqrt((residual_field**2).sum(axis=2))
        curve = []
        for limit in arguments.iterations:
            solution = lsqr(action, target, iter_lim=limit, atol=0.0, btol=0.0, conlim=0.0)[0]
            plastic = solution.reshape(pixels, pixels, 2, 3)
            equivalent = np.sqrt(
                np.maximum(np.einsum("xyci,ij,xycj->xyc", plastic, gauge, plastic), 0.0)
            ).mean(axis=2)
            model = np.asarray(forward(solution))
            flat_model, flat_residual = equivalent.reshape(-1), residual_amplitude.reshape(-1)
            top = flat_residual >= np.quantile(flat_residual, 0.9)
            curve.append(
                {
                    "iterations": limit,
                    "relative_residual": float(np.linalg.norm(model - target) / norm),
                    "plastic_rms": float(np.sqrt((equivalent**2).mean())),
                    "plastic_peak": float(equivalent.max()),
                    "correlation_with_the_explained_residual": float(
                        np.corrcoef(flat_model, flat_residual)[0, 1]
                    ),
                    "share_inside_the_residual_top_decile": float(
                        flat_model[top].sum() / flat_model.sum()
                    ),
                }
            )
        results[str(state)] = curve
        print(f"\nstate {state}")
        print("  iters | residual | plastic RMS | plastic peak | corr | top decile")
        for entry in curve:
            print(
                f"  {entry['iterations']:5d} | {entry['relative_residual']:8.4f} | "
                f"{entry['plastic_rms']:11.3e} | {entry['plastic_peak']:12.3e} | "
                f"{entry['correlation_with_the_explained_residual']:+.3f} | "
                f"{entry['share_inside_the_residual_top_decile']:.3f}"
            )

    output = {
        "schema_version": 1,
        "pixels": pixels,
        "origin_nodes": [x0, y0],
        "reference_state": arguments.reference_state,
        "method": "LSQR on the whitened strain residual; iteration count is the regularisation",
        "note": (
            "no SVD reordering: the previous attempt ranked the Krylov space by |A phi|, "
            "which reintroduced the observability ordering it was meant to avoid"
        ),
        "results": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
