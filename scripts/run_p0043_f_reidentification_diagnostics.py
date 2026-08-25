#!/usr/bin/env python3
"""Run the strict F re-forward and nine-parameter observability diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles
from scipy.optimize import LinearConstraint, minimize

from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET, get_parameter_set
from fem_inhouse.identification.srix_parameter_coordinates import SrixTheta9
from scripts.qualify_srix_p0043_synthetic_smoke import (
    CROP,
    _forward,
    _load_inputs,
    _make_path,
    _vector,
)
from scripts.qualify_srix_svd_shadow import _direct_shadow, _step_sizes

ROOT = Path(__file__).resolve().parents[1]
GLOBAL = ROOT / "validation/reference_data/p0043_global_srix_observability_v1"
F_REPORT = ROOT / "validation/reference_data/p0043_experimental_raw_svd7_f_provisional_v1/report.json"
DEFAULT_OUTPUT = ROOT / "validation/reference_data/p0043_f_mapping_reidentification_v1"


def _physical_bounds(eta_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = eta_ref.copy()
    high = eta_ref.copy()
    low[:3] += np.log(0.85)
    high[:3] += np.log(1.15)
    physical = np.asarray([
        [25.0, 60.0], [8.0, 35.0], [4.0, 20.0], [1.0, 6.0],
        [10000.0, 80000.0], [500.0, 3000.0],
    ])
    low[3:] = np.log(physical[:, 0])
    high[3:] = np.log(physical[:, 1])
    return low, high


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--point", choices=("best", "prior"), default="best")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)

    measured_macro, angles, provenance = _load_inputs(CROP)
    path = _make_path(measured_macro, 4)
    scored = tuple(4 * index for index in range(1, 9))
    target = [np.asarray(step.boundary, dtype=np.float64).copy() for step in path]
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )
    f_report = json.loads(F_REPORT.read_text(encoding="utf-8"))
    eta_best = np.asarray(f_report["final_eta"], dtype=np.float64)
    if args.point == "prior":
        eta_best = SrixTheta9.from_parameter_set(
            get_parameter_set(DEFAULT_PARAMETER_SET)
        ).log_coordinates()
    theta_best = SrixTheta9.from_log_coordinates(eta_best)
    started = time.perf_counter()
    fields, timing = _forward(theta_best, path, angles, library, args.threads, "F")
    strict_residual = float(timing["verification_residual"])
    strict_vector = _vector(fields, scored, target)
    strict_rms = float(np.sqrt(np.mean(strict_vector**2)))
    (output / "strict_reforward.json").write_text(json.dumps({
        "element_order": "F", "parameters": theta_best.as_runtime_overrides(),
        "verification_residual": strict_residual, "raw_rms_mm": strict_rms,
        "previous_recorded_rms_mm": f_report["final_rms_mm"] if args.point == "best" else None,
        "mechanically_verified": bool(strict_residual <= 1.0e-8),
        "timing": timing, "elapsed_seconds": time.perf_counter() - started,
    }, indent=2, sort_keys=True) + "\n")

    # Full nine-parameter projected-shadow Jacobian at the strict F best point.
    basis9 = np.eye(9, dtype=np.float64)
    steps9 = np.full(9, 1.5e-3, dtype=np.float64)
    jac_f, shadow_timing = _direct_shadow(
        fields=fields, basis=basis9, eta=eta_best, step_sizes=steps9,
        angles=angles, scored=scored, library=library, threads=args.threads,
        element_order="F",
    )
    np.save(output / "full_jacobian_f.npy", jac_f)
    u, singular, vh = np.linalg.svd(jac_f, full_matrices=False)
    vf = vh.T
    normalized = singular / singular[0]
    (output / "svd_f.json").write_text(json.dumps({
        "singular_values": singular.tolist(),
        "normalized_singular_values": normalized.tolist(),
        "right_singular_vectors": vf.tolist(),
        "step_sizes": steps9.tolist(), "shadow_timing": shadow_timing,
    }, indent=2, sort_keys=True) + "\n")

    old = np.load(GLOBAL / "global_observability.npz")
    vc = np.asarray(old["eigenvectors"], dtype=np.float64)
    angles_rank7 = np.degrees(subspace_angles(vc[:, :7], vf[:, :7]))
    weak_angles = np.degrees(subspace_angles(vf[:, 7:], np.column_stack([
        np.asarray([1, 1, 1, 1, 1, 1, 0, 1, 0], dtype=float),
        np.asarray([0, 0, 0, 0, 0, 1, -1, 0, 0], dtype=float),
    ])))
    (output / "c_vs_f_subspace_angles.json").write_text(json.dumps({
        "rank7_angles_deg": angles_rank7.tolist(),
        "rank7_max_deg": float(np.max(angles_rank7)),
        "rank7_mean_deg": float(np.mean(angles_rank7)),
        "rank7_overlap_frobenius": float(np.linalg.norm(vc[:, :7].T @ vf[:, :7], ord="fro")),
    }, indent=2, sort_keys=True) + "\n")
    (output / "weak_modes_f.json").write_text(json.dumps({
        "weak_subspace_angles_to_scale_and_q_over_b_deg": weak_angles.tolist(),
        "v8": vf[:, 7].tolist(), "v9": vf[:, 8].tolist(),
        "scale_direction": [1, 1, 1, 1, 1, 1, 0, 1, 0],
        "q_over_b_direction": [0, 0, 0, 0, 0, 1, -1, 0, 0],
    }, indent=2, sort_keys=True) + "\n")

    # KKT/GN diagnostic at the archived SLSQP endpoint.  This is a linear model only.
    traj = np.load(ROOT / "validation/reference_data/p0043_experimental_raw_svd7_f_provisional_v1/trajectory.npz")
    residual = np.asarray(traj["residual"], dtype=float)
    jac = np.asarray(traj["jacobian"], dtype=float)
    z = np.asarray(traj["z"], dtype=float)
    basis_old = np.asarray(traj["basis"], dtype=float)
    eta_ref = SrixTheta9.from_parameter_set(get_parameter_set(DEFAULT_PARAMETER_SET)).log_coordinates()
    eta_old = np.asarray(traj["eta"], dtype=float)
    low, high = _physical_bounds(eta_ref)
    eta_old = eta_ref + basis_old @ z
    gradient = jac.T @ residual
    active_low = np.isclose(eta_old, low, rtol=0.0, atol=2.0e-5)
    active_high = np.isclose(eta_old, high, rtol=0.0, atol=2.0e-5)
    constraint = LinearConstraint(basis_old, low - eta_ref, high - eta_ref)
    lin_objective = lambda dz: 0.5 * float(np.dot(residual + jac @ dz, residual + jac @ dz))
    lin_jac = lambda dz: jac.T @ (residual + jac @ dz)
    lin_fit = minimize(lin_objective, np.zeros_like(z), jac=lin_jac, method="SLSQP",
                       constraints=(constraint,), options={"ftol": 1.0e-16, "maxiter": 500})
    lin_residual = residual + jac @ lin_fit.x
    rms0 = float(np.sqrt(np.mean(residual**2)))
    rms_lin = float(np.sqrt(np.mean(lin_residual**2)))
    (output / "kkt_diagnostic.json").write_text(json.dumps({
        "endpoint_element_order": "F", "endpoint_mechanically_verified": False,
        "gradient": gradient.tolist(), "gradient_norm": float(np.linalg.norm(gradient)),
        "active_low_indices": np.flatnonzero(active_low).tolist(),
        "active_high_indices": np.flatnonzero(active_high).tolist(),
        "projected_linear_step": lin_fit.x.tolist(), "linear_solver_success": bool(lin_fit.success),
        "linear_solver_message": str(lin_fit.message),
        "linear_rms_mm": rms_lin, "endpoint_rms_mm": rms0,
        "predicted_relative_reduction": float(1.0 - rms_lin / rms0),
        "feasible_descent_direction_exists": bool(lin_fit.success and rms_lin < rms0),
        "basis_used": basis_old.tolist(),
    }, indent=2, sort_keys=True) + "\n")
    (output / "final_report.json").write_text(json.dumps({
        "schema_version": 1, "element_order": "F", "crop": list(CROP),
        "strict_reforward": json.loads((output / "strict_reforward.json").read_text()),
        "full_9p_jacobian_f_computed": True, "f_rank7_basis_computed": True,
        "c_rank7_reusable_for_f": bool(np.max(angles_rank7) < 1.0),
        "projected_shadows_f_qualified": None,
        "slsqp_stop_is_true_kkt": False,
        "feasible_descent_direction_exists": bool(lin_fit.success and rms_lin < rms0),
        "raw_f_optimization_stationary": False,
        "m100_exploratory_gate": False,
        "diagnostic_point": args.point,
        "provenance": provenance,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "strict_rms_mm": strict_rms, "strict_verification_residual": strict_residual,
        "singular_values": singular.tolist(), "rank7_max_angle_deg": float(np.max(angles_rank7)),
        "weak_angles_deg": weak_angles.tolist(), "linear_predicted_reduction": float(1.0-rms_lin/rms0),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
