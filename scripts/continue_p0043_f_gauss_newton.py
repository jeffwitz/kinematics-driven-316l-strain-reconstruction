#!/usr/bin/env python3
"""Constrained raw F-mapping Gauss-Newton continuation from the valid prior."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET, get_parameter_set
from fem_inhouse.identification.srix_parameter_coordinates import SrixTheta9
from scripts.qualify_srix_p0043_synthetic_smoke import CROP, _forward, _load_inputs, _make_path, _vector
from scripts.qualify_srix_svd_shadow import _direct_shadow, _step_sizes

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "validation/reference_data/p0043_f_mapping_reidentification_prior_v1"
DEFAULT_OUTPUT = ROOT / "validation/reference_data/p0043_f_mapping_reidentification_shadow_v1"


def bounds(eta_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low, high = eta_ref.copy(), eta_ref.copy()
    low[:3] += np.log(0.85)
    high[:3] += np.log(1.15)
    physical = np.asarray([[25, 60], [8, 35], [4, 20], [1, 6], [10000, 80000], [500, 3000]], float)
    low[3:], high[3:] = np.log(physical[:, 0]), np.log(physical[:, 1])
    return low, high


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-accepted", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    measured, angles, provenance = _load_inputs(CROP)
    path, scored = _make_path(measured, 4), tuple(4 * i for i in range(1, 9))
    target = [np.asarray(step.boundary, float).copy() for step in path]
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so"))
    eta_ref = SrixTheta9.from_parameter_set(get_parameter_set(DEFAULT_PARAMETER_SET)).log_coordinates()
    low, high = bounds(eta_ref)
    svd = json.loads((PRIOR / "svd_f.json").read_text())
    vf = np.asarray(svd["right_singular_vectors"], float)
    basis = vf[:, :7]
    weak = vf[:, 7:]
    weak0 = weak.T @ (eta_ref - eta_ref)
    if "step_sizes" in svd:
        steps = np.asarray(svd["step_sizes"], float)[:7]
    else:
        steps = _step_sizes(np.asarray(svd["singular_values"], float), 7)
    eta = eta_ref.copy()
    z = np.zeros(7)
    cache: dict[bytes, tuple[float, np.ndarray, np.ndarray, float]] = {}
    history: list[dict[str, object]] = []
    starts = time.perf_counter()

    def evaluate(zv: np.ndarray, need_jacobian: bool = True):
        nonlocal eta
        key = np.ascontiguousarray(zv).tobytes()
        if key in cache and (not need_jacobian or cache[key][2] is not None):
            return cache[key]
        etav = eta_ref + basis @ zv + weak @ weak0
        theta = SrixTheta9.from_log_coordinates(etav)
        fields, timing = _forward(theta, path, angles, library, args.threads, "F")
        verification = float(timing["verification_residual"])
        valid = verification <= 1.0e-8
        residual = _vector(fields, scored, target)
        rms = float(np.sqrt(np.mean(residual**2)))
        jac = None
        if need_jacobian and valid:
            jac, shadow = _direct_shadow(fields=fields, basis=basis, eta=etav, step_sizes=steps,
                angles=angles, scored=scored, library=library, threads=args.threads, element_order="F")
            timing = {**timing, "shadow_seconds": shadow["elapsed_seconds"]}
        value = (rms, residual, jac, verification)
        cache[key] = value
        history.append({"evaluation": len(history) + 1, "z": zv.tolist(), "eta": etav.tolist(),
                        "parameters": theta.as_runtime_overrides(), "rms_mm": rms,
                        "verification_residual": verification, "objective_valid": valid,
                        "timing": timing})
        return value

    # The prior F point was already re-forwarded and its full nine-parameter
    # shadow Jacobian archived by the preceding diagnostic.  Reuse that exact
    # valid matrix for the first GN step; subsequent accepted points are
    # differentiated afresh.
    theta0 = SrixTheta9.from_log_coordinates(eta_ref)
    fields0, timing0 = _forward(theta0, path, angles, library, args.threads, "F")
    residual = _vector(fields0, scored, target)
    rms = float(np.sqrt(np.mean(residual**2)))
    jac = np.load(PRIOR / "full_jacobian_f.npy") @ basis
    verification = float(timing0["verification_residual"])
    cache[np.ascontiguousarray(z).tobytes()] = (rms, residual, jac, verification)
    history.append({"evaluation": 1, "z": z.tolist(), "eta": eta_ref.tolist(),
                    "parameters": theta0.as_runtime_overrides(), "rms_mm": rms,
                    "verification_residual": verification, "objective_valid": verification <= 1.0e-8,
                    "timing": timing0, "jacobian_source": "archived_prior_full_jacobian_f"})
    if jac is None:
        raise RuntimeError("valid prior F Jacobian unavailable")
    initial_rms = rms
    constraint = LinearConstraint(basis, low - eta_ref - weak @ weak0, high - eta_ref - weak @ weak0)
    accepted = 0
    while accepted < args.max_accepted:
        def fun(d):
            rr = residual + jac @ d
            return 0.5 * float(rr @ rr)
        def grad(d):
            return jac.T @ (residual + jac @ d)
        gn = minimize(fun, np.zeros(7), jac=grad, method="SLSQP", constraints=(constraint,),
                      options={"ftol": 1e-18, "maxiter": 500})
        direction = np.asarray(gn.x, float)
        accepted_this = False
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            trial_z = z + alpha * direction
            trial_eta = eta_ref + basis @ trial_z + weak @ weak0
            if np.any(trial_eta < low - 1.0e-10) or np.any(trial_eta > high + 1.0e-10):
                history.append({"accepted": False, "reason": "physical_constraint_violation",
                                "alpha": alpha, "z": trial_z.tolist(),
                                "eta": trial_eta.tolist()})
                continue
            trial_rms, trial_residual, trial_jac, trial_verification = evaluate(trial_z, need_jacobian=False)
            if trial_verification <= 1e-8 and trial_rms < rms:
                z, rms, residual, verification = trial_z, trial_rms, trial_residual, trial_verification
                trial_rms, trial_residual, trial_jac, trial_verification = evaluate(z, need_jacobian=True)
                jac = trial_jac
                accepted += 1
                accepted_this = True
                history[-1]["accepted"] = True
                history[-1]["alpha"] = alpha
                history[-1]["linear_predicted_rms_mm"] = float(np.sqrt(2.0 * fun(direction) / residual.size))
                break
        if not accepted_this:
            history.append({"accepted": False, "reason": "no_valid_decreasing_backtracking_step",
                            "linear_step": direction.tolist(), "linear_solver_success": bool(gn.success),
                            "linear_solver_message": str(gn.message)})
            break
    eta_final = eta_ref + basis @ z + weak @ weak0
    final_theta = SrixTheta9.from_log_coordinates(eta_final)
    with (output / "optimization_history.csv").open("w", newline="") as handle:
        keys = ["evaluation", "rms_mm", "verification_residual", "objective_valid", "accepted", "alpha"]
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(history)
    report = {"schema_version": 1, "method": "constrained raw F Gauss-Newton with projected rank-7 basis",
              "element_order": "F", "crop": list(CROP), "path_steps": len(path),
              "accepted_evaluations": accepted, "max_accepted": args.max_accepted,
              "prior_rms_mm": initial_rms, "final_rms_mm": rms,
              "relative_reduction": 1.0 - rms / initial_rms, "final_verification_residual": verification,
              "final_eta": eta_final.tolist(), "final_z": z.tolist(),
              "final_parameters": final_theta.as_runtime_overrides(), "weak_coordinates": weak0.tolist(),
              "history": history, "provenance": provenance,
              "stationary": not any(x.get("accepted", False) for x in history[-1:]),
              "claims": {"raw_f_optimization_stationary": False, "m100_exploratory_gate": False}}
    (output / "optimization_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted": accepted, "prior_rms_mm": initial_rms, "final_rms_mm": rms,
                      "verification": verification, "elapsed_s": time.perf_counter() - starts}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
