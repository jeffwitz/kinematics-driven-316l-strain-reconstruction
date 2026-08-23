#!/usr/bin/env python3
"""Arm A4: the free tensor family, per increment, on the measured targets.

The milestone-3 family (192 local tensor coefficients, degree zero, patches 8,
projection after assembly) fitted increment by increment against the measured
DIC targets, through the same linear mechanics and the same `E` scoring as
`learn_flow_direction_p43.py`. A4 is the fit ceiling and is not identifiable —
the preregistration registers it as such; this driver only measures how far
fitting can go.

Per state:

* `target = measured[s] - predictor_s`, the predictor accumulating this arm's
  own previous increments;
* the projection half-space uses the state's own predictor stress, the same
  frame as the learned arms;
* the fit minimises `0.5 |apply_numpy(P_H(B a)) - target|^2` with an exact
  gradient (A is linear and self-adjoint, the projection jacobian is exact);
* the score is `E_s = |measured - (predictor + response)| / defect_s`, the
  script's own metric.

The `moved` flag is reported per the preregistration's dead-ReLU clause.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from fem_inhouse.core.kelvin import KELVIN_SCALE_2D
from fem_inhouse.identification.tensor_local_inverse import (
    DissipativeProjection,
    TensorLocalBasis,
)
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior

ROOT = Path(__file__).resolve().parents[1]
HISTORY = (
    ROOT
    / "validation/reference_data/dic_multistep_history_p0043_repaired_v1"
    / "repaired_history_mm.npy"
)
OUT = ROOT / "validation/_generated/shared_tensor_generator"
PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30
ORIGIN = (1580, 1030)
PIXELS = 100
PATCHES = 8
SUBCELLS = 2
REFERENCE_STATE = 20
STATES = list(range(21, 41))
HELDOUT = (24, 28, 32, 36, 40)


class _Identity:
    def apply(self, values):
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values):
        return np.asarray(values, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--origin", nargs=2, type=int, default=list(ORIGIN))
    parser.add_argument("--pixels", type=int, default=PIXELS)
    parser.add_argument("--output", type=Path, default=OUT / "arm_a4.json")
    arguments = parser.parse_args()
    pixels = arguments.pixels
    x0, y0 = arguments.origin
    out = arguments.output

    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    operator = TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=_Identity(),
        whitener=_Identity(),
    )
    basis = TensorLocalBasis.build(grid.nx, grid.ny, PATCHES)
    points = operator.kinematics.material_point_count

    def kelvin_strain(field) -> np.ndarray:
        return operator.kelvin_strain(field).reshape(-1, 3)

    def stress_of(strain: np.ndarray) -> np.ndarray:
        return np.einsum("pi,pij->pj", strain.reshape(-1, 3), operator.elasticity)

    def divergence(stress_kelvin: np.ndarray) -> np.ndarray:
        voigt = stress_kelvin.reshape(-1, 3) / KELVIN_SCALE_2D
        return pack_interior(
            operator.kinematics.divergence_from_sample_stress(voigt.reshape((pixels, pixels, 2, 3)))
        )

    def elastic_lift(field: np.ndarray) -> np.ndarray:
        forcing = -divergence(stress_of(kelvin_strain(field))) / operator.quadrature_weight
        lifted = field.copy()
        lifted[1:-1, 1:-1, :] -= operator.solve_stiffness(forcing).reshape(
            pixels - 1, pixels - 1, 2
        )
        return lifted

    def apply_numpy(plastic: np.ndarray) -> np.ndarray:
        flat = plastic.reshape(points, 3, -1)
        stressed = np.stack([stress_of(flat[:, :, k]) for k in range(flat.shape[2])], axis=2)
        loads = np.stack(
            [
                operator._strain_transpose(stressed[:, :, k].reshape(-1))
                for k in range(stressed.shape[2])
            ],
            axis=1,
        )
        solved = operator.solve_stiffness(loads)
        solved = solved.reshape(-1, stressed.shape[2]) if solved.ndim == 2 else solved
        return np.stack(
            [
                kelvin_strain(np.asarray(operator.kinematics.unpack_interior(solved[:, k], grid)))
                for k in range(stressed.shape[2])
            ],
            axis=2,
        ).reshape(plastic.shape)

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
    reference = history[REFERENCE_STATE]
    residual_guard = np.linalg.norm(
        divergence(stress_of(kelvin_strain(elastic_lift(history[40] - reference))))
    )
    print(f"elastic lifting residual (guard): {residual_guard:.3e}", flush=True)
    if residual_guard > 1e-8:
        raise RuntimeError(f"replicated elastic lift does not equilibrate ({residual_guard:.3e})")

    measured = {s: kelvin_strain(history[s] - reference) for s in STATES}
    elastic = {s: kelvin_strain(elastic_lift(history[s] - reference)) for s in STATES}
    defect = {s: float(np.linalg.norm(measured[s] - elastic[s])) for s in STATES}
    reference_stress = stress_of(kelvin_strain(reference))

    count = basis.coefficient_count
    cumulative = np.zeros((points, 3))
    scores: dict[int, float] = {}
    moved = False
    per_state: dict[int, dict[str, float]] = {}
    for state in STATES:
        predictor = elastic[state] + apply_numpy(cumulative)
        target = measured[state] - predictor
        stress = reference_stress + stress_of(predictor - cumulative)
        projection = DissipativeProjection(stress=stress)

        def assemble(
            coefficients: np.ndarray,
            active_projection: DissipativeProjection = projection,
        ) -> np.ndarray:
            field = basis.assemble(coefficients)
            raw = np.repeat(field[:, :, None, :], SUBCELLS, axis=2).reshape(-1, 3)
            projected, active = active_projection.apply(raw)
            return projected, active

        def objective_and_gradient(
            coefficients: np.ndarray,
            active_target: np.ndarray = target,
            active_projection: DissipativeProjection = projection,
        ) -> tuple[float, np.ndarray]:
            projected, active = assemble(coefficients)
            response = apply_numpy(projected)
            residual = response - active_target
            dual = apply_numpy(residual)
            dual = active_projection.transpose_action(dual, active)
            dual = np.repeat(
                dual.reshape(pixels, pixels, SUBCELLS, 3).sum(axis=2), 1, axis=0
            )
            gradient = basis.assemble_transpose(
                dual.reshape(pixels, pixels, SUBCELLS, 3).sum(axis=2)
            ).ravel()
            return 0.5 * float(np.sum(residual**2)), gradient

        result = minimize(
            objective_and_gradient,
            np.zeros(count),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": arguments.iterations, "ftol": 1e-16, "gtol": 1e-10},
        )
        projected, active = assemble(result.x)
        response = apply_numpy(projected)
        simulated = predictor + response
        cumulative = cumulative + projected
        score = float(np.linalg.norm(measured[state] - simulated) / defect[state])
        gauge = float(np.sqrt(np.maximum(np.sum(projected**2), 0.0)))
        if gauge > 0.0:
            moved = True
        scores[state] = score
        per_state[state] = {
            "E": score,
            "gauge": gauge,
            "objective": float(result.fun),
            "active_fraction": float(np.mean(active)),
            "iterations": int(result.nit),
            "success": bool(result.success),
        }
        print(
            f"  state {state:2d}: E {score:.4f}  gauge {gauge:.3e}  "
            f"active {100 * float(np.mean(active)):.1f} %  nit {result.nit}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "arm": "A4",
        "origin": [x0, y0],
        "pixels": pixels,
        "moved": moved,
        "heldout_median_E": float(np.median([scores[s] for s in HELDOUT])),
        "final_state_E": scores[STATES[-1]],
        "per_state": per_state,
        "elastic_lifting_residual_guard": residual_guard,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
