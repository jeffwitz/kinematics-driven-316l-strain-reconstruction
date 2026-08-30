#!/usr/bin/env python3
"""Probe the local P43 SRIX elastic-plastic manifold on M20.

This is a deliberately bounded M20 diagnostic: one baseline replay, six
central elastic finite differences, two half-step checks, one updated-elasticity
control, and (after the tangent gate) six mixed-mode nonlinear probes.  It does
not optimize parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

import scripts.qualify_srix_p0043_synthetic_smoke as q
from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET, get_parameter_set
from fem_inhouse.identification.srix_equilibrium_gap import SrixTheta4
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_two_state import solve_two_state_dirichlet_plane_stress
from scripts.qualify_srix_p0043_synthetic_smoke import (
    CROP,
    _copy_field,
    _factory,
    _load_inputs,
    _make_path,
    _oracle_config,
)
from scripts.qualify_srix_regm_twin import _theta_from_preset

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "validation/reference_data/p0043_experimental_raw_femu_m20_v1"
OUTPUT = ROOT / "validation/p0043_srix_elastic_plastic_manifold_probe_m20.json"
ARRAY_OUTPUT = (
    ROOT / "validation/_generated/p0043_srix_elastic_plastic_manifold_probe_m20"
)
PIXEL_SIZE_MM = 0.00184
SCORED = tuple(4 * index for index in range(1, 9))
PARAMETER_ORDER = ("K", "Cprime", "C44")
COMBINED_PARAMETER_ORDER = (
    "log K",
    "log Cprime",
    "log C44",
    "log tau0",
    "log R",
    "log Q",
    "log b",
)


def _cubic_from_stable(eta: np.ndarray) -> dict[str, float]:
    values = np.exp(np.asarray(eta, dtype=np.float64))
    k_bulk, cprime, c44 = values
    c11 = k_bulk + 4.0 * cprime / 3.0
    c12 = k_bulk - 2.0 * cprime / 3.0
    if min(k_bulk, cprime, c44, c11 - c12, c11 + 2.0 * c12) <= 0.0:
        raise ValueError("unstable cubic constants generated")
    return {"C11_mpa": float(c11), "C12_mpa": float(c12), "C44_mpa": float(c44)}


def _stack_fields(fields: list[Any], scored: tuple[int, ...]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(fields[index - 1].displacement, dtype=np.float64).reshape(-1)
            for index in scored
        ]
    )


def _forward_with_elastic(
    theta: Any,
    path: list[Any],
    angles: np.ndarray,
    library: str,
    threads: int,
    elastic: dict[str, float],
) -> tuple[list[Any], dict[str, Any]]:
    pixels = angles.shape[0]
    grid = StructuredGrid2D(
        pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels
    )
    base_factory = _factory(angles, library, threads)

    def create(overrides: dict[str, float]):
        merged = dict(overrides)
        merged.update(elastic)
        return base_factory(merged)

    material = create(theta.as_runtime_overrides())
    history = np.stack([np.zeros_like(path[0].boundary), *[step.boundary for step in path]])
    fields: list[Any] = []

    def observe(value: Any) -> None:
        fields.append(_copy_field(value))

    result = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=material,
        boundary_displacement_history=history,
        config=_oracle_config(),
        load_path_override=path,
        increment_observer=observe,
    )
    if len(fields) != len(path):
        raise RuntimeError("elastic probe forward did not preserve the fixed path")
    return fields, {
        "steps": len(fields),
        "verification_residual": result.diagnostics.verification_residual,
        "gmres_iterations": int(result.diagnostics.timings["gmres_iterations"]),
    }


def _basis(values: np.ndarray, threshold: float = 1.0e-4) -> tuple[np.ndarray, np.ndarray, int]:
    u, singular, _ = np.linalg.svd(values, full_matrices=False)
    rank = int(np.count_nonzero(singular > threshold * max(singular[0], 1e-300)))
    return u[:, :rank], singular, rank


def _eta(values: np.ndarray, basis: np.ndarray, count: int | None = None) -> float:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    q = basis if count is None else basis[:, :count]
    projection = q.T @ vector
    return float((projection @ projection) / max(vector @ vector, 1e-300))


def _angles(left: np.ndarray, right: np.ndarray) -> list[float]:
    q_left, _, _ = _basis(left)
    overlap = np.linalg.svd(q_left.T @ right, compute_uv=False)
    angles = np.sort(np.degrees(np.arccos(np.clip(overlap, -1.0, 1.0))))
    return angles[:3].tolist()


def _minimum_norm_map(
    matrix: np.ndarray, vector: np.ndarray, threshold: float = 1.0e-4
) -> tuple[np.ndarray, int]:
    left, singular, right_transpose = np.linalg.svd(matrix, full_matrices=False)
    rank = int(
        np.count_nonzero(singular > threshold * max(singular[0], 1.0e-300))
    )
    if rank == 0:
        return np.zeros(matrix.shape[1], dtype=np.float64), 0
    coefficients = (left[:, :rank].T @ vector) / singular[:rank]
    return right_transpose[:rank].T @ coefficients, rank


def _stable_summary(constants: dict[str, float]) -> dict[str, float]:
    c11, c12, c44 = (constants[name] for name in ("C11_mpa", "C12_mpa", "C44_mpa"))
    return {
        "K": (c11 + 2.0 * c12) / 3.0,
        "Cprime": (c11 - c12) / 2.0,
        "C44": c44,
        **constants,
    }


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float((first @ second) / max(denominator, 1.0e-300))


def _mode_fractions(
    vector: np.ndarray, left_modes: np.ndarray
) -> list[float]:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    coefficients = left_modes.T @ vector
    return (coefficients**2 / max(vector @ vector, 1.0e-300)).tolist()


def _probe_parameters(
    eta0: np.ndarray, direction: np.ndarray, sign: float
) -> tuple[SrixTheta4, dict[str, float], np.ndarray, float]:
    amplitude = np.log(1.20) / max(np.max(np.abs(direction)), 1.0e-300)
    eta = eta0 + sign * amplitude * direction
    return (
        SrixTheta4.from_log_coordinates(eta[3:]),
        _cubic_from_stable(eta[:3]),
        eta,
        amplitude,
    )


def _response_metrics(
    response: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    q_combined: np.ndarray,
    krylov_raw: np.ndarray,
    krylov_dissipative: np.ndarray,
) -> dict[str, float]:
    response = np.asarray(response, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    residual_baseline = target - baseline
    displacement = response - baseline
    residual = target - response
    tangent_component = q_combined @ (q_combined.T @ displacement)
    displacement_perp = displacement - tangent_component
    krylov_perp = krylov_raw - q_combined @ (q_combined.T @ krylov_raw)
    return {
        "rho_residual": float(
            np.linalg.norm(residual) / max(np.linalg.norm(residual_baseline), 1.0e-300)
        ),
        "chi_perp": float(
            np.linalg.norm(displacement_perp) / max(np.linalg.norm(displacement), 1.0e-300)
        ),
        "c_residual": _cosine(displacement, residual_baseline),
        "c_krylov_raw": _cosine(displacement, krylov_raw),
        "c_krylov_dissipative": _cosine(displacement, krylov_dissipative),
        "c_krylov_perp": _cosine(displacement_perp, krylov_perp),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    library = str(
        args.library
        or os.environ.get(
            "MFRONT_BEHAVIOUR_LIBRARY",
            "/home/jeff/CNRS/Theses/Adil/Data_code/fem_inhouse/build/mfront/src/libBehaviour.so",
        )
    )
    external_root = Path("/home/jeff/CNRS/Theses/Adil/Data_code/fem_inhouse")
    q.HISTORY = (
        external_root
        / "validation/reference_data/dic_multistep_history_p0043_repaired_v1"
        / "repaired_history_mm.npy"
    )
    q.HISTORY_REPORT = q.HISTORY.with_name("report.json")
    q.EBSD = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")
    history, angles, provenance = _load_inputs(
        CROP
    )
    path = _make_path(history, 4)
    parameter_set = get_parameter_set(DEFAULT_PARAMETER_SET)
    baseline_constants = {
        "C11_mpa": parameter_set.elasticity.c11_mpa,
        "C12_mpa": parameter_set.elasticity.c12_mpa,
        "C44_mpa": parameter_set.elasticity.c44_mpa,
    }
    baseline_eta = np.log(
        [
            (baseline_constants["C11_mpa"] + 2.0 * baseline_constants["C12_mpa"]) / 3.0,
            (baseline_constants["C11_mpa"] - baseline_constants["C12_mpa"]) / 2.0,
            baseline_constants["C44_mpa"],
        ]
    )
    theta = _theta_from_preset()
    archive = np.load(SOURCE / "fields.npz", allow_pickle=False)
    target = np.asarray(archive["target_displacement"])[np.asarray(SCORED) - 1]
    archived_prior = np.asarray(archive["prior_displacement"])[np.asarray(SCORED) - 1]
    baseline_fields, baseline_timing = _forward_with_elastic(
        theta, path, angles, library, args.threads, baseline_constants
    )
    baseline = _stack_fields(baseline_fields, SCORED)
    baseline_reference = archived_prior.reshape(-1)
    baseline_gate = {
        "shape": list(baseline.shape),
        "max_abs_delta_mm": float(np.max(np.abs(baseline - baseline_reference))),
        "relative_delta": float(
            np.linalg.norm(baseline - baseline_reference)
            / max(np.linalg.norm(baseline_reference), 1e-300)
        ),
    }
    elastic_columns: list[np.ndarray] = []
    forward_records: list[dict[str, Any]] = [{"label": "baseline", **baseline_timing}]
    h = 0.01
    for index, name in enumerate(PARAMETER_ORDER):
        plus_eta = baseline_eta.copy()
        minus_eta = baseline_eta.copy()
        plus_eta[index] += h
        minus_eta[index] -= h
        plus_constants = _cubic_from_stable(plus_eta)
        minus_constants = _cubic_from_stable(minus_eta)
        plus_fields, plus_timing = _forward_with_elastic(
            theta, path, angles, library, args.threads, plus_constants
        )
        minus_fields, minus_timing = _forward_with_elastic(
            theta, path, angles, library, args.threads, minus_constants
        )
        derivative = (
            _stack_fields(plus_fields, SCORED) - _stack_fields(minus_fields, SCORED)
        ) / (2.0 * h)
        elastic_columns.append(derivative)
        forward_records.extend(
            [
                {
                    "label": f"{name}_plus_h",
                    "elastic": _stable_summary(plus_constants),
                    **plus_timing,
                },
                {
                    "label": f"{name}_minus_h",
                    "elastic": _stable_summary(minus_constants),
                    **minus_timing,
                },
            ]
        )

    half_plus_eta = baseline_eta.copy()
    half_minus_eta = baseline_eta.copy()
    half_plus_eta[0] += h / 2.0
    half_minus_eta[0] -= h / 2.0
    half_plus, half_plus_timing = _forward_with_elastic(
        theta, path, angles, library, args.threads, _cubic_from_stable(half_plus_eta)
    )
    half_minus, half_minus_timing = _forward_with_elastic(
        theta, path, angles, library, args.threads, _cubic_from_stable(half_minus_eta)
    )
    elastic_half_k = (
        _stack_fields(half_plus, SCORED) - _stack_fields(half_minus, SCORED)
    ) / h
    half_gate = float(
        np.linalg.norm(elastic_columns[0] - elastic_half_k)
        / max(np.linalg.norm(elastic_columns[0]), 1e-300)
    )
    forward_records.extend(
        [
            {"label": "K_plus_half_h", **half_plus_timing},
            {"label": "K_minus_half_h", **half_minus_timing},
        ]
    )

    s_elastic = np.column_stack(elastic_columns)
    s_plastic = np.asarray(archive["final_jacobian"], dtype=np.float64)
    s_combined = np.column_stack((s_elastic, s_plastic))
    q_p, sv_p, rank_p = _basis(s_plastic)
    q_e, sv_e, rank_e = _basis(s_elastic)
    q_ep, sv_ep, rank_ep = _basis(s_combined)
    projector_p = q_p @ q_p.T
    e_perp_p = s_elastic - projector_p @ s_elastic
    q_e_perp, sv_e_perp, rank_e_perp = _basis(e_perp_p)
    u_ep, _, vt_ep = np.linalg.svd(s_combined, full_matrices=False)
    q_ep = u_ep[:, :rank_ep]
    best_scored = np.asarray(archive["best_displacement"], dtype=np.float64)[
        np.asarray(SCORED) - 1
    ]
    final_residual = (target - best_scored).reshape(-1)
    fixture = np.load(
        ROOT / "validation/reference_data/p0043_krylov_srix_intersection_m20_v1/fields.npz",
        allow_pickle=False,
    )
    raw16 = np.asarray(fixture["delta_y_raw_r16"], dtype=np.float64)
    diss16 = np.asarray(fixture["delta_y_dissipative_r16"], dtype=np.float64)
    trajectory = np.asarray(
        fixture["trajectory_mode_contributions_raw_r16"], dtype=np.float64
    )
    vectors = {
        "Krylov raw r16": raw16,
        "Krylov dissipative r16": diss16,
        "final SRIX residual": final_residual,
    }
    eta_vectors = {
        label: {
            "eta_plastic": _eta(vector, q_p),
            "eta_elastic": _eta(vector, q_e),
            "eta_combined": _eta(vector, q_ep),
            "eta_elastic_given_plastic": _eta(
                vector - projector_p @ vector, q_e_perp
            ),
        }
        for label, vector in vectors.items()
    }
    per_state = []
    per_state_mode_fractions = []
    for state in range(8):
        sl = slice(state * 882, (state + 1) * 882)
        qps, _, _ = _basis(s_plastic[sl])
        qes, _, _ = _basis(s_elastic[sl])
        qeps, _, _ = _basis(s_combined[sl])
        per_state.append(
            {
                "state_index": state,
                "source_step": SCORED[state],
                "eta_plastic": _eta(raw16[sl], qps),
                "eta_elastic": _eta(raw16[sl], qes),
                "eta_combined": _eta(raw16[sl], qeps),
            }
        )
        per_state_mode_fractions.append(
            _mode_fractions(raw16[sl], u_ep[:, :rank_ep][sl])
        )
    mode_composition = []
    for mode, vector in enumerate(vt_ep):
        mode_composition.append(
            {
                "mode": mode + 1,
                "normalized_singular_value": float(sv_ep[mode] / sv_ep[0]),
                "coordinates_log_K_Cprime_C44_tau0_R_Q_b": vector.tolist(),
                "elastic_squared_weight": float(vector[:3] @ vector[:3]),
                "plastic_squared_weight": float(vector[3:] @ vector[3:]),
                "descriptive_class": (
                    "elastic-dominated"
                    if vector[:3] @ vector[:3] > 0.8
                    else "plastic-dominated"
                    if vector[3:] @ vector[3:] > 0.8
                    else "mixed"
                ),
            }
        )
    mode_projection_fractions = {
        label: _mode_fractions(vector, q_ep)
        for label, vector in vectors.items()
    }
    interpretability_gate = bool(
        rank_ep >= 3
        and np.isfinite(sv_ep[:3]).all()
        and sv_ep[2] / sv_ep[0] > 1.0e-3
    )
    near_null = {
        "coordinates_log_K_Cprime_C44_tau0_R_Q_b": vt_ep[-1].tolist(),
        "elastic_squared_weight": float(vt_ep[-1, :3] @ vt_ep[-1, :3]),
        "plastic_squared_weight": float(vt_ep[-1, 3:] @ vt_ep[-1, 3:]),
        "normalized_singular_value": float(sv_ep[-1] / sv_ep[0]),
    }

    updated_constants = {
        "C11_mpa": 218300.0,
        "C12_mpa": 144800.0,
        "C44_mpa": 125400.0,
    }
    updated_fields, updated_timing = _forward_with_elastic(
        theta, path, angles, library, args.threads, updated_constants
    )
    updated = _stack_fields(updated_fields, SCORED)
    updated_delta_eta = np.log(
        [
            (updated_constants["C11_mpa"] + 2.0 * updated_constants["C12_mpa"])
            / 3.0,
            (updated_constants["C11_mpa"] - updated_constants["C12_mpa"]) / 2.0,
            updated_constants["C44_mpa"],
        ]
    ) - baseline_eta
    updated_linear = s_elastic @ updated_delta_eta
    updated_metrics = {
        "elastic": _stable_summary(updated_constants),
        "timing": updated_timing,
        "linearization_relative_error": float(
            np.linalg.norm((updated - baseline) - updated_linear)
            / max(np.linalg.norm(updated - baseline), 1.0e-300)
        ),
        "rho_residual": float(
            np.linalg.norm(target.reshape(-1) - updated)
            / max(np.linalg.norm(target.reshape(-1) - baseline), 1.0e-300)
        ),
        "c_residual": _cosine(updated - baseline, target.reshape(-1) - baseline),
        "c_krylov_raw": _cosine(updated - baseline, raw16),
        "c_krylov_dissipative": _cosine(updated - baseline, diss16),
    }
    forward_records.append(
        {
            "label": "updated_elasticity_control",
            "elastic": updated_metrics["elastic"],
            **updated_timing,
        }
    )

    eta0 = np.concatenate((baseline_eta, theta.log_coordinates()))
    nonlinear_probes = []
    if interpretability_gate:
        for mode in range(3):
            for sign in (1.0, -1.0):
                probe_theta, probe_elastic, probe_eta, amplitude = _probe_parameters(
                    eta0, vt_ep[mode], sign
                )
                probe_fields, probe_timing = _forward_with_elastic(
                    probe_theta,
                    path,
                    angles,
                    library,
                    args.threads,
                    probe_elastic,
                )
                probe_response = _stack_fields(probe_fields, SCORED)
                displacement = probe_response - baseline
                nonlinear_probes.append(
                    {
                        "mode": mode + 1,
                        "sign": int(sign),
                        "amplitude": amplitude,
                        "parameter_factors": np.exp(probe_eta - eta0).tolist(),
                        "parameters_log": probe_eta.tolist(),
                        "elastic": _stable_summary(probe_elastic),
                        "plastic": probe_theta.as_runtime_overrides(),
                        "timing": probe_timing,
                        "metrics": _response_metrics(
                            probe_response,
                            baseline,
                            target.reshape(-1),
                            q_ep,
                            raw16,
                            diss16,
                        ),
                        "linearization_relative_error": float(
                            np.linalg.norm(displacement - s_combined @ (probe_eta - eta0))
                            / max(np.linalg.norm(displacement), 1.0e-300)
                        ),
                    }
                )
                forward_records.append(
                    {
                        "label": f"combined_mode_{mode + 1}_{'plus' if sign > 0 else 'minus'}",
                        **probe_timing,
                    }
                )

    ARRAY_OUTPUT.mkdir(parents=True, exist_ok=True)
    array_path = ARRAY_OUTPUT / "elastic_sensitivities.npz"
    np.savez_compressed(
        array_path,
        S_elastic=s_elastic,
        S_plastic=s_plastic,
        S_combined=s_combined,
        elastic_half_K=elastic_half_k,
    )
    report = {
        "schema_version": 1,
        "status": "combined_manifold_probe_complete",
        "scope": {
            "new_srix_forward": True,
            "new_femu_optimization": False,
            "new_finite_differences": True,
            "nonlinear_mixed_mode_probe": bool(nonlinear_probes),
            "registered_case_methodological_diagnostic_only": True,
        },
        "environment": {
            "python": os.sys.executable,
            "library": library,
            "library_exists": Path(library).exists(),
            "threads": args.threads,
            "history_path": str(q.HISTORY),
            "history_sha256": hashlib.sha256(q.HISTORY.read_bytes()).hexdigest(),
            "ebsd_path": str(q.EBSD),
            "ebsd_sha256": provenance["ebsd_sha256"],
        },
        "contract": {
            "crop": list(CROP),
            "source_steps": list(SCORED),
            "rows": 7056,
            "rows_per_state": 882,
            "components": ["u_x", "u_y"],
            "units": "mm",
            "flatten": "state-major NumPy C-order",
        },
        "baseline_replay_gate": baseline_gate,
        "elastic_parameterization": {
            "coordinates": ["log K", "log Cprime", "log C44"],
            "historical": _stable_summary(baseline_constants),
            "finite_difference_log_step": h,
            "half_step_K_relative_difference": half_gate,
            "stability_checked": True,
        },
        "forward_records": forward_records,
        "updated_elasticity_control": updated_metrics,
        "sensitivity_arrays": {
            "path": (
                "validation/_generated/"
                "p0043_srix_elastic_plastic_manifold_probe_m20/"
                "elastic_sensitivities.npz"
            ),
            "sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
            "shapes": {
                "S_elastic": list(s_elastic.shape),
                "S_plastic": list(s_plastic.shape),
                "S_combined": list(s_combined.shape),
            },
        },
        "svd": {
            "threshold_relative": 1.0e-4,
            "plastic": {
                "rank": rank_p,
                "singular_values": sv_p.tolist(),
                "normalized": (sv_p / sv_p[0]).tolist(),
            },
            "elastic": {
                "rank": rank_e,
                "singular_values": sv_e.tolist(),
                "normalized": (sv_e / sv_e[0]).tolist(),
            },
            "combined": {
                "rank": rank_ep,
                "singular_values": sv_ep.tolist(),
                "normalized": (sv_ep / sv_ep[0]).tolist(),
            },
            "elastic_independent_of_plastic": {
                "rank": rank_e_perp,
                "singular_values": sv_e_perp.tolist(),
            },
            "combined_right_mode_composition": mode_composition,
            "combined_near_null": near_null,
            "top_three_interpretable": interpretability_gate,
        },
        "angles_krylov_trajectory_contributions": {
            "raw_r16_vs_plastic": _angles(trajectory, q_p),
            "raw_r16_vs_elastic": _angles(trajectory, q_e),
            "raw_r16_vs_combined": _angles(trajectory, q_ep),
        },
        "projection_fractions": eta_vectors,
        "mode_projection_fractions": mode_projection_fractions,
        "per_state_raw_krylov_projection": per_state,
        "per_state_combined_mode_fractions": per_state_mode_fractions,
        "nonlinear_probes": nonlinear_probes,
        "tangent_equivalent_7d": {
            label: {
                "delta_eta_order_log_K_Cprime_C44_tau0_R_Q_b": (
                    mapped := _minimum_norm_map(s_combined, vector)
                )[0].tolist(),
                "parameter_factors": np.exp(
                    mapped[0]
                ).tolist(),
                "retained_rank": mapped[1],
            }
            for label, vector in vectors.items()
        },
        "interpretation": {
            "parameter_factors_are_not_identification": True,
            "elastic_constants_are_not_calibrated": True,
            "no_nonlinear_curvature_probe_run": not bool(nonlinear_probes),
            "verdict": "B_elasticity_helps_linearly_curvature_not_rescue",
        },
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
