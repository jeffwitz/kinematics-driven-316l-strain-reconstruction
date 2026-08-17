#!/usr/bin/env python3
"""The projected-Krylov control: raw modes, free coefficients, P_H after assembly.

The last purely geometric control before crystallography, registered in
`validation/krylov_projected_control_preregistration.md`. Raw Krylov modes
(built from DIC residuals, either from all states — the expressivity oracle —
or from the training states only — the predictive line), free signed
coefficients, and the dissipative half-space applied to the assembled
combination only:

    Delta eps^p = P_H[ Phi_K a ],   never  sum_k a_k P_H(phi_k).

Two projectors: the plastic-gauge `G_p` one (primary) and the Euclidean
ablation; `--projector none` disables the projection and must reproduce the
archived raw-Krylov scores (equivalence check of the replication).

The fit is `min_a 1/2 |A P_H(Phi_K a) - g|^2` per state, sequentially, with
an exact gradient (A's adjoint and the projection transpose are exact on the
frozen active set), L-BFGS-B, multi-start from zero, the unprojected
least-squares solution and one random start.

Reported per rank: E per state and held-out median, chi (mean where active and
work-weighted global), accumulated p_eq, D- and D+ work shares, projection
activity and the multi-start spread.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from fem_inhouse.core.kelvin import KELVIN_SCALE_2D, PLANE_STRESS_PLASTIC_GAUGE
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior

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
REFERENCE_STATE = 20
STATES = list(range(21, 41))
HELDOUT = (24, 28, 32, 36, 40)
RIDGE = 1e-6

GAUGE = np.asarray(PLANE_STRESS_PLASTIC_GAUGE, dtype=np.float64)
INV_GAUGE = np.linalg.inv(GAUGE)


class _Identity:
    def apply(self, values):
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values):
        return np.asarray(values, dtype=np.float64)


def project(
    values: np.ndarray,
    stress: np.ndarray,
    kind: str,
    inverse_metric: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Half-space projection and its per-point active mask.

    `kind == "gp"` corrects along `G_p^{-1} sigma`; `"euclid"` along `sigma`;
    `"none"` is the identity (equivalence check)."""
    if kind == "none":
        return values, np.zeros(values.shape[0], dtype=bool)
    metric = INV_GAUGE if kind == "gp" else None
    dot = np.einsum("pi,pi->p", stress, values)
    if metric is None:
        denom = np.einsum("pi,pi->p", stress, stress)
        correction = np.where(dot < 0.0, -dot, 0.0) / denom
        return values + correction[:, None] * stress, dot < 0.0
    direction = np.einsum("ij,pj->pi", metric, stress)
    denom = np.einsum("pi,pi->p", stress, direction)
    correction = np.where(dot < 0.0, -dot, 0.0) / denom
    return values + correction[:, None] * direction, dot < 0.0


def project_transpose(
    dual: np.ndarray,
    stress: np.ndarray,
    kind: str,
    active: np.ndarray,
) -> np.ndarray:
    """Exact transpose of `project` at the frozen active set."""
    if kind == "none" or not np.any(active):
        return dual
    if kind == "euclid":
        denom = np.einsum("pi,pi->p", stress, stress)
        scalar = np.einsum("pi,pi->p", stress, dual) / denom
        result = dual.copy()
        result[active] -= (scalar[:, None] * stress)[active]
        return result
    metric = INV_GAUGE
    direction = np.einsum("ij,pj->pi", metric, stress)
    denom = np.einsum("pi,pi->p", stress, direction)
    scalar = np.einsum("pi,pi->p", stress, np.einsum("ij,pj->pi", metric, dual)) / denom
    result = dual.copy()
    # The transpose corrects along sigma, not along G_p^{-1} sigma: the
    # forward moves along G_p^{-1} sigma, so its adjoint pulls the dual along
    # sigma with the metric-contracted scalar. The Euclidean case coincides.
    result[active] -= (scalar[:, None] * stress)[active]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projector", choices=("gp", "euclid", "none"), default="gp")
    parser.add_argument("--krylov", choices=("oracle", "predictive"), default="predictive")
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--origin", nargs=2, type=int, default=list(ORIGIN))
    parser.add_argument("--pixels", type=int, default=PIXELS)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--gradient-check", action="store_true",
                        help="FD sweep of the projected chain gradient at one state, then exit")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    pixels = arguments.pixels
    x0, y0 = arguments.origin

    grid = StructuredGrid2D(
        pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels
    )
    operator = TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=_Identity(),
        whitener=_Identity(),
    )
    points = operator.kinematics.material_point_count

    def kelvin_strain(field) -> np.ndarray:
        return operator.kelvin_strain(field).reshape(-1, 3)

    def stress_of(strain: np.ndarray) -> np.ndarray:
        return np.einsum("pi,pij->pj", strain.reshape(-1, 3), operator.elasticity)

    def divergence(stress_kelvin: np.ndarray) -> np.ndarray:
        voigt = stress_kelvin.reshape(-1, 3) / KELVIN_SCALE_2D
        return pack_interior(
            operator.kinematics.divergence_from_sample_stress(
                voigt.reshape((pixels, pixels, 2, 3))
            )
        )

    def elastic_lift(field: np.ndarray) -> np.ndarray:
        forcing = -divergence(stress_of(kelvin_strain(field))) / operator.quadrature_weight
        lifted = field.copy()
        lifted[1:-1, 1:-1, :] -= operator.solve_stiffness(forcing).reshape(
            pixels - 1, pixels - 1, 2
        )
        return lifted

    def batched_green(fields: np.ndarray) -> np.ndarray:
        columns = fields.shape[2]
        loads = np.stack(
            [operator._strain_transpose(fields[:, :, k].reshape(-1)) for k in range(columns)],
            axis=1,
        )
        solved = operator.solve_stiffness(loads if columns > 1 else loads[:, 0])
        solved = solved.reshape(-1, columns) if columns > 1 else solved[:, None]
        return np.stack(
            [kelvin_strain(np.asarray(unpack_interior(solved[:, k], grid)))
             for k in range(columns)],
            axis=2,
        )

    def apply_numpy(plastic: np.ndarray) -> np.ndarray:
        flat = plastic.reshape(points, 3, -1)
        stressed = np.stack([stress_of(flat[:, :, k]) for k in range(flat.shape[2])], axis=2)
        return batched_green(stressed).reshape(plastic.shape)

    def transpose_numpy(values: np.ndarray) -> np.ndarray:
        flat = values.reshape(points, 3, -1)
        strained = batched_green(flat)
        return np.stack([stress_of(strained[:, :, k]) for k in range(strained.shape[2])], axis=2).reshape(values.shape)

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
    training = [s for s in STATES if s not in HELDOUT]
    seed_states = STATES if arguments.krylov == "oracle" else training

    def krylov_basis(size: int) -> np.ndarray:
        seeds = [transpose_numpy(measured[s] - elastic[s]).reshape(-1) for s in seed_states]
        basis, _ = np.linalg.qr(np.asarray(seeds).T)
        columns, total = [basis], basis.shape[1]
        while total < size:
            grown = np.asarray(
                [transpose_numpy(apply_numpy(columns[-1][:, k].reshape(points, 3))).reshape(-1)
                 for k in range(columns[-1].shape[1])]
            ).T
            orthonormal, _ = np.linalg.qr(np.concatenate([*columns, grown], axis=1))
            addition = orthonormal[:, total:]
            if addition.shape[1] == 0:
                break
            columns.append(addition)
            total += addition.shape[1]
        return np.concatenate(columns, axis=1)[:, :size]

    if arguments.gradient_check:
        # Plumbing, not a milestone: central FD against the exact gradient of
        # the projected chain, four decades, at one state of the first rank.
        rank = arguments.ranks[0]
        basis = krylov_basis(rank)
        modes = basis.reshape(points, 3, rank)
        stress = reference_stress.copy()
        target = measured[21] - elastic[21]
        rng = np.random.default_rng(20260817)
        a0 = rng.normal(size=rank)

        def objective_and_gradient_check(a):
            increment = modes @ a
            projected, active = project(increment, stress, arguments.projector)
            response = apply_numpy(projected)
            residual = response - target
            dual = transpose_numpy(residual)
            dual = project_transpose(dual, stress, arguments.projector, active)
            gradient = modes.reshape(-1, rank).T @ dual.ravel()
            return 0.5 * float(np.sum(residual**2)), gradient

        base_value, analytic = objective_and_gradient_check(a0)
        sweep = {}
        for h in (1e-3, 1e-4, 1e-5, 1e-6):
            scale = max(float(np.abs(a0).max()), 1e-9)
            step = h * scale
            worst = 0.0
            for k in range(rank):
                delta = np.zeros(rank)
                delta[k] = step
                plus, _ = objective_and_gradient_check(a0 + delta)
                minus, _ = objective_and_gradient_check(a0 - delta)
                fd = (plus - minus) / 2
                denom = abs(analytic[k] * step)
                if denom > 0:
                    worst = max(worst, abs(analytic[k] * step - fd) / denom)
            sweep[h] = worst
        payload = {
            "schema_version": 1,
            "projector": arguments.projector,
            "rank": rank,
            "sweep": sweep,
            "projection_active_fraction": float(
                np.mean(project(modes @ a0, stress, arguments.projector)[1])
            ),
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    results: dict[str, dict] = {}
    for rank in arguments.ranks:
        basis = krylov_basis(rank)
        modes = basis.reshape(points, 3, rank)
        responses = apply_numpy(modes).reshape(-1, rank)
        gram = responses.T @ responses
        plastic = np.zeros((points, 3))
        previous_stress = reference_stress.copy()
        scores: dict[int, float] = {}
        per_state: dict[int, dict] = {}
        total_power_positive = 0.0
        total_power_negative = 0.0
        total_predictor_negative = 0.0
        total_predictor_positive = 0.0
        accumulated_gauge = 0.0
        chi_active: list[float] = []
        activities: list[float] = []
        starts_spread: list[float] = []
        for state in STATES:
            predictor = elastic[state] + apply_numpy(plastic)
            stress = reference_stress + stress_of(predictor - plastic)
            target = measured[state] - predictor

            def objective_and_gradient(a):
                increment = modes @ a
                projected, active = project(increment, stress, arguments.projector)
                response = apply_numpy(projected)
                residual = response - target
                dual = transpose_numpy(residual)
                dual = project_transpose(dual, stress, arguments.projector, active)
                gradient = modes.reshape(-1, rank).T @ dual.ravel()
                return 0.5 * float(np.sum(residual**2)), gradient

            if arguments.projector == "none":
                # Linear problem: the script's own baseline solve, verbatim.
                best_solution = np.linalg.solve(
                    gram + RIDGE * np.trace(gram) / rank * np.eye(rank),
                    responses.T @ target.reshape(-1),
                )
                best_value = 0.5 * float(
                    np.sum((responses @ best_solution - target.reshape(-1)) ** 2)
                )
                starts_spread.append(0.0)
            else:
                starts = [np.zeros(rank)]
                linear = np.linalg.solve(
                    gram + RIDGE * np.trace(gram) / rank * np.eye(rank),
                    responses.T @ target.reshape(-1),
                )
                starts.append(linear)
                rng = np.random.default_rng(20260817 + rank)
                starts.append(
                    rng.normal(
                        scale=max(float(np.linalg.norm(starts[0])), 1e-6), size=rank
                    )
                )
                best_value, best_solution = np.inf, None
                values = []
                for start in starts:
                    result = minimize(
                        objective_and_gradient,
                        start,
                        method="L-BFGS-B",
                        jac=True,
                        options={"maxiter": arguments.maxiter, "ftol": 1e-14, "gtol": 1e-8},
                    )
                    values.append(float(result.fun))
                    if float(result.fun) < best_value:
                        best_value, best_solution = float(result.fun), result.x
                starts_spread.append(float(np.std(values)) / max(best_value, 1e-300))
            increment = modes @ best_solution
            projected, active = project(increment, stress, arguments.projector)
            increment = projected
            response = apply_numpy(increment)
            simulated = predictor + response
            plastic = plastic + increment
            new_stress = reference_stress + stress_of(simulated - plastic)
            power = (0.5 * (previous_stress + new_stress) * increment).sum(axis=1)
            power_pred = (stress * increment).sum(axis=1)
            frobenius = np.sqrt(np.maximum(
                np.einsum("pi,ij,pj->p", increment, 1.5 * GAUGE, increment), 0.0))
            stress_norm = np.sqrt((new_stress**2).sum(axis=1))
            gauge = np.sqrt(np.maximum(np.einsum("pi,ij,pj->p", increment, GAUGE, increment), 0.0))
            active_points = gauge > 0.1 * max(gauge.mean(), 1e-300)
            cosine = power / np.maximum(stress_norm * frobenius, 1e-300)
            if active_points.any():
                chi_active.append(float(cosine[active_points].mean()))
            total_power_positive += float(np.maximum(power, 0.0).sum())
            total_power_negative += float(np.abs(np.minimum(power, 0.0)).sum())
            total_predictor_positive += float(np.maximum(power_pred, 0.0).sum())
            total_predictor_negative += float(np.abs(np.minimum(power_pred, 0.0)).sum())
            accumulated_gauge += float(gauge.sum())
            activities.append(float(np.mean(active)))
            previous_stress = new_stress
            score = float(np.linalg.norm(measured[state] - simulated) / defect[state])
            scores[state] = score
            per_state[state] = {
                "E": score,
                "objective": best_value,
                "active_fraction": float(np.mean(active)),
                "start_spread": starts_spread[-1],
                "chi_active_mean": chi_active[-1] if chi_active else 0.0,
            }
            print(
                f"  r={rank:2d} state {state:2d}: E {score:.4f}  "
                f"active {100 * float(np.mean(active)):.1f} %  obj {best_value:.3e}",
                flush=True,
            )
        results[f"r{rank}"] = {
            "heldout_median_E": float(np.median([scores[s] for s in HELDOUT])),
            "fitted_mean_E": float(np.mean([scores[s] for s in training])),
            "per_state": per_state,
            "chi_active_mean": float(np.mean(chi_active)),
            "chi_global_work_weighted": float(
                (total_power_positive - total_power_negative)
                / max(total_power_positive + total_power_negative, 1e-300)
            ),
            "accumulated_equivalent_plastic_strain": accumulated_gauge / points,
            "negative_power_share": float(
                total_power_negative / max(total_power_positive + total_power_negative, 1e-300)
            ),
            "positive_work": total_power_positive,
            "negative_work_midpoint": total_power_negative,
            "predictor_negative_share": float(
                total_predictor_negative
                / max(total_predictor_positive + total_predictor_negative, 1e-300)
            ),
            "predictor_negative_work": total_predictor_negative,
            "projected_fraction_mean": float(np.mean(activities)),
            "start_spread_relative_mean": float(np.mean(starts_spread)),
        }
        print(
            f"rank {rank}: held out {results[f'r{rank}']['heldout_median_E']:.4f}  "
            f"chi {results[f'r{rank}']['chi_active_mean']:+.3f} "
            f"(global {results[f'r{rank}']['chi_global_work_weighted']:+.3f})  "
            f"p_eq {results[f'r{rank}']['accumulated_equivalent_plastic_strain']:.3e}  "
            f"neg {100 * results[f'r{rank}']['negative_power_share']:.2f} % of power  "
            f"proj {100 * results[f'r{rank}']['projected_fraction_mean']:.1f} %",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "projector": arguments.projector,
        "krylov": arguments.krylov,
        "origin": [x0, y0],
        "pixels": pixels,
        "ranks": arguments.ranks,
        "elastic_lifting_residual_guard": residual_guard,
        "results": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
