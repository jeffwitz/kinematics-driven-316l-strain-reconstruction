#!/usr/bin/env python3
"""Run the RAW F-mapping P43 M100 identification from the qualified M20 point."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.optimize import LinearConstraint, minimize

from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET, get_parameter_set
from fem_inhouse.identification.srix_parameter_coordinates import SrixTheta9
from scripts.plot_p0043_raw_svd7_evm_maps import (
    _evm,
    _plot_historical_style,
    _plot_summary,
)
from scripts.qualify_srix_p0043_synthetic_smoke import (
    _forward,
    _load_inputs,
    _make_path,
    _vector,
)
from scripts.qualify_srix_svd_shadow import _direct_shadow

ROOT = Path(__file__).resolve().parents[1]
M20_REPORT = ROOT / (
    "validation/reference_data/p0043_f_mapping_reidentification_shadow_v1/"
    "optimization_report.json"
)
FULL_CROP = (1580, 1680, 1030, 1130)
DEFAULT_OUTPUT = ROOT / (
    "validation/reference_data/p0043_f_mapping_reidentification_m100_shadow_v1"
)
H = 1.5e-3


def _bounds(eta_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low, high = eta_ref.copy(), eta_ref.copy()
    low[:3] += np.log(0.85)
    high[:3] += np.log(1.15)
    physical = np.asarray(
        [[25.0, 60.0], [8.0, 35.0], [4.0, 20.0], [1.0, 6.0],
         [10000.0, 80000.0], [500.0, 3000.0]],
        dtype=float,
    )
    low[3:] = np.log(physical[:, 0])
    high[3:] = np.log(physical[:, 1])
    return low, high


def _plot_evm(
    output: Path,
    measured: np.ndarray,
    initial: list,
    final: list,
    scored: tuple[int, ...],
) -> None:
    dic = np.stack([_evm(measured[index - 1]) for index in scored])
    initial_evm = np.stack([_evm(initial[index - 1].displacement) for index in scored])
    final_evm = np.stack([_evm(final[index - 1].displacement) for index in scored])
    labels = [str(index) for index in range(1, len(scored) + 1)]
    _plot_historical_style(output, dic, initial_evm, final_evm, labels)
    _plot_summary(output, dic, initial_evm, final_evm, labels)
    np.savez_compressed(
        output / "evm_fields.npz", dic=dic, initial=initial_evm, final=final_evm
    )
    (output / "evm_metrics.json").write_text(json.dumps({
        "states": labels,
        "dic_rms": np.sqrt(np.mean(dic**2, axis=(1, 2))).tolist(),
        "initial_rms": np.sqrt(np.mean(initial_evm**2, axis=(1, 2))).tolist(),
        "final_rms": np.sqrt(np.mean(final_evm**2, axis=(1, 2))).tolist(),
        "initial_minus_dic_rms": np.sqrt(np.mean((initial_evm - dic)**2, axis=(1, 2))).tolist(),
        "final_minus_dic_rms": np.sqrt(np.mean((final_evm - dic)**2, axis=(1, 2))).tolist(),
    }, indent=2) + "\n")


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

    m20_report = json.loads(M20_REPORT.read_text())
    eta_m20 = np.asarray(m20_report["final_eta"], dtype=float)
    measured, angles, provenance = _load_inputs(FULL_CROP)
    path = _make_path(measured, 4)
    scored = tuple(4 * index for index in range(1, 9))
    target = [np.asarray(step.boundary, dtype=float).copy() for step in path]
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )
    prior_eta = SrixTheta9.from_parameter_set(
        get_parameter_set(DEFAULT_PARAMETER_SET)
    ).log_coordinates()
    prior = SrixTheta9.from_log_coordinates(prior_eta)
    low, high = _bounds(prior_eta)

    started = time.perf_counter()
    initial_theta = SrixTheta9.from_log_coordinates(eta_m20)
    initial_fields, initial_timing = _forward(
        initial_theta, path, angles, library, args.threads, "F"
    )
    initial_residual = _vector(initial_fields, scored, target)
    initial_rms = float(np.sqrt(np.mean(initial_residual**2)))

    # Build the M100 basis at the actual M20-initialized point. This is a
    # direct-shadow exploratory basis; its complete FD oracle is intentionally
    # not launched at M100.
    full_basis = np.eye(9, dtype=float)
    full_steps = np.full(9, H, dtype=float)
    full_jacobian, full_shadow_timing = _direct_shadow(
        fields=initial_fields,
        basis=full_basis,
        eta=eta_m20,
        step_sizes=full_steps,
        angles=angles,
        scored=scored,
        library=library,
        threads=args.threads,
        element_order="F",
    )
    _, singular, vh = np.linalg.svd(full_jacobian, full_matrices=False)
    vectors = vh.T
    basis = vectors[:, :7]
    weak = vectors[:, 7:]
    z_all = vectors.T @ (eta_m20 - prior_eta)
    z = z_all[:7].copy()
    weak0 = z_all[7:].copy()
    steps = np.full(7, H, dtype=float)
    cache: dict[bytes, tuple[float, np.ndarray, np.ndarray | None, float]] = {}
    history: list[dict[str, object]] = []

    def evaluate(z_value: np.ndarray, need_jacobian: bool = True):
        key = np.ascontiguousarray(z_value).tobytes()
        if key in cache and (not need_jacobian or cache[key][2] is not None):
            return cache[key]
        eta = prior_eta + basis @ z_value + weak @ weak0
        fields, timing = _forward(
            SrixTheta9.from_log_coordinates(eta), path, angles, library, args.threads, "F"
        )
        verification = float(timing["verification_residual"])
        residual = _vector(fields, scored, target)
        rms = float(np.sqrt(np.mean(residual**2)))
        jac = None
        if need_jacobian and verification <= 1.0e-8:
            jac, shadow_timing = _direct_shadow(
                fields=fields,
                basis=basis,
                eta=eta,
                step_sizes=steps,
                angles=angles,
                scored=scored,
                library=library,
                threads=args.threads,
                element_order="F",
            )
            timing = {**timing, "shadow_seconds": shadow_timing["elapsed_seconds"]}
        value = (rms, residual, jac, verification)
        cache[key] = value
        history.append({
            "evaluation": len(history) + 1,
            "z": z_value.tolist(),
            "eta": eta.tolist(),
            "parameters": SrixTheta9.from_log_coordinates(eta).as_runtime_overrides(),
            "rms_mm": rms,
            "verification_residual": verification,
            "objective_valid": verification <= 1.0e-8,
            "timing": timing,
        })
        return value

    initial_jacobian = full_jacobian @ basis
    cache[np.ascontiguousarray(z).tobytes()] = (
        initial_rms,
        initial_residual,
        initial_jacobian,
        float(initial_timing["verification_residual"]),
    )
    history.append({
        "evaluation": 1,
        "z": z.tolist(),
        "eta": eta_m20.tolist(),
        "parameters": initial_theta.as_runtime_overrides(),
        "rms_mm": initial_rms,
        "verification_residual": float(initial_timing["verification_residual"]),
        "objective_valid": bool(initial_timing["verification_residual"] <= 1.0e-8),
        "timing": initial_timing,
        "jacobian_source": "M100 direct shadow at M20 initialization",
    })
    rms, residual, jac, _verification = cache[np.ascontiguousarray(z).tobytes()]
    accepted = 0
    while accepted < args.max_accepted:
        def objective(delta, residual0=residual, jac0=jac):
            value = residual0 + jac0 @ delta
            return 0.5 * float(value @ value)

        def gradient(delta, residual0=residual, jac0=jac):
            return jac0.T @ (residual0 + jac0 @ delta)

        constraint = LinearConstraint(
            basis,
            low - prior_eta - weak @ weak0,
            high - prior_eta - weak @ weak0,
        )
        gn = minimize(
            objective,
            np.zeros(7),
            jac=gradient,
            method="SLSQP",
            constraints=(constraint,),
            options={"ftol": 1.0e-18, "maxiter": 500},
        )
        direction = np.asarray(gn.x, dtype=float)
        accepted_this = False
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            trial_z = z + alpha * direction
            trial_eta = prior_eta + basis @ trial_z + weak @ weak0
            if np.any(trial_eta < low - 1.0e-10) or np.any(trial_eta > high + 1.0e-10):
                continue
            trial_rms, trial_residual, _, trial_verification = evaluate(
                trial_z, need_jacobian=False
            )
            if trial_verification <= 1.0e-8 and trial_rms < rms:
                z, rms, residual = trial_z, trial_rms, trial_residual
                _, _, jac, _ = evaluate(z, need_jacobian=True)
                accepted += 1
                accepted_this = True
                history[-1]["accepted"] = True
                history[-1]["alpha"] = alpha
                break
        if not accepted_this:
            history.append({
                "accepted": False,
                "reason": "no_valid_decreasing_backtracking_step",
                "linear_solver_success": bool(gn.success),
                "linear_solver_message": str(gn.message),
            })
            break

    eta_final = prior_eta + basis @ z + weak @ weak0
    final_theta = SrixTheta9.from_log_coordinates(eta_final)
    final_fields, final_timing = _forward(
        final_theta, path, angles, library, args.threads, "F"
    )
    final_residual = _vector(final_fields, scored, target)
    final_rms = float(np.sqrt(np.mean(final_residual**2)))
    np.savez_compressed(
        output / "fields.npz",
        dic_displacement=np.asarray(target),
        initial_displacement=np.asarray([field.displacement for field in initial_fields]),
        final_displacement=np.asarray([field.displacement for field in final_fields]),
        initial_residual=initial_residual,
        final_residual=final_residual,
    )
    _plot_evm(output, measured, initial_fields, final_fields, scored)
    report = {
        "schema_version": 1,
        "method": "RAW P43 M100 F-mapping rank-7 direct-shadow identification",
        "git_sha": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "dirty": bool(__import__("subprocess").run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()),
        "machine": platform.node(),
        "crop": list(FULL_CROP),
        "mesh": list(angles.shape[:2]),
        "path_steps": len(path),
        "scored_steps": list(scored),
        "element_order": "F",
        "spectral_batch_order": "C",
        "observation_weighting": "none",
        "m20_initialization_report": str(M20_REPORT),
        "m20_initialization_eta": eta_m20.tolist(),
        "m20_initialization_parameters": initial_theta.as_runtime_overrides(),
        "prior": prior.as_runtime_overrides(),
        "provenance": provenance,
        "initial_forward": initial_timing,
        "final_forward": final_timing,
        "initial_rms_mm": initial_rms,
        "final_rms_mm": final_rms,
        "relative_rms_reduction": 1.0 - final_rms / initial_rms,
        "accepted_evaluations": accepted,
        "history": history,
        "m100_svd": {
            "singular_values": singular.tolist(),
            "normalized_singular_values": (singular / singular[0]).tolist(),
            "right_singular_vectors": vectors.tolist(),
            "full_shadow_timing": full_shadow_timing,
        },
        "final_parameters": final_theta.as_runtime_overrides(),
        "elapsed_seconds": time.perf_counter() - started,
        "claims": {
            "m100_forward_completed": True,
            "m100_raw_optimization_stationary": False,
            "experimental_parameters_identified": False,
        },
    }
    with (output / "optimization_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["evaluation", "rms_mm", "verification_residual", "accepted", "alpha"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(history)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "accepted": accepted,
        "initial_rms_mm": initial_rms,
        "final_rms_mm": final_rms,
        "verification_residual": float(final_timing["verification_residual"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
