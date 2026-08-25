#!/usr/bin/env python3
"""Exploratory raw P43 M20 fit in the provisional global SVD rank-7 basis."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET, get_parameter_set
from fem_inhouse.identification.srix_parameter_coordinates import SrixTheta9
from scripts.qualify_srix_p0043_synthetic_smoke import (
    CROP,
    _forward,
    _git,
    _load_inputs,
    _make_path,
    _vector,
)
from scripts.qualify_srix_svd_shadow import _direct_shadow, _step_sizes

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "validation/reference_data/p0043_global_srix_observability_v1"
DEFAULT_OUTPUT = ROOT / "validation/reference_data/p0043_experimental_raw_svd7_provisional_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--element-order", choices=("C", "F"), default="C")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)

    archive = np.load(SMOKE / "global_observability.npz")
    eigenvectors = np.asarray(archive["eigenvectors"], dtype=np.float64)
    eigenvalues = np.asarray(archive["eigenvalues"], dtype=np.float64)
    basis = eigenvectors[:, :7]
    discarded = eigenvectors[:, 7:]
    eta_ref = SrixTheta9.from_parameter_set(
        get_parameter_set(DEFAULT_PARAMETER_SET)
    ).log_coordinates()
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
    steps = _step_sizes(np.sqrt(np.maximum(eigenvalues, 0.0)), 7)

    measured_macro, angles, provenance = _load_inputs(CROP)
    path = _make_path(measured_macro, 4)
    scored = tuple(4 * index for index in range(1, 9))
    target = [np.asarray(step.boundary, dtype=np.float64).copy() for step in path]
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )
    cache: dict[bytes, tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}
    history: list[dict[str, object]] = []
    residual_scale: float | None = None

    def evaluate(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        eta = eta_ref + basis @ np.asarray(z, dtype=np.float64)
        key = eta.tobytes()
        if key not in cache:
            theta = SrixTheta9.from_log_coordinates(eta)
            fields, timing = _forward(
                theta, path, angles, library, args.threads, args.element_order
            )
            residual = _vector(fields, scored, target)
            jacobian, shadow_timing = _direct_shadow(
                fields=fields, basis=basis, eta=eta, step_sizes=steps,
                angles=angles, scored=scored, library=library, threads=args.threads,
                element_order=args.element_order,
            )
            timing = {**timing, "shadow_seconds": shadow_timing["elapsed_seconds"]}
            nonlocal residual_scale
            if residual_scale is None:
                residual_scale = float(np.sqrt(np.mean(residual**2)))
            cache[key] = (residual, jacobian, timing)
            history.append({
                "evaluation": len(history) + 1,
                "z": np.asarray(z).tolist(),
                "eta": eta.tolist(),
                "parameters": theta.as_runtime_overrides(),
                "rms_mm": float(np.sqrt(np.mean(residual**2))),
                "timing": timing,
            })
        return cache[key]

    def residual(z: np.ndarray) -> np.ndarray:
        values = evaluate(z)[0]
        if residual_scale is None:
            raise RuntimeError("residual scale was not initialized")
        return values / residual_scale

    def jacobian(z: np.ndarray) -> np.ndarray:
        values = evaluate(z)[1]
        if residual_scale is None:
            raise RuntimeError("residual scale was not initialized")
        return values / residual_scale

    initial_z = np.zeros(7, dtype=np.float64)
    initial_residual, initial_jacobian, _ = evaluate(initial_z)
    if residual_scale is None:
        raise RuntimeError("residual scale was not initialized")
    gn_step, *_ = np.linalg.lstsq(initial_jacobian, -initial_residual, rcond=None)
    predicted_residual = initial_residual + initial_jacobian @ gn_step
    predicted_reduction = float(
        1.0 - np.linalg.norm(predicted_residual) / np.linalg.norm(initial_residual)
    )
    started = time.perf_counter()
    eta_constraint = LinearConstraint(basis, low - eta_ref, high - eta_ref)

    def objective(z: np.ndarray) -> tuple[float, np.ndarray]:
        values = residual(z)
        matrix = jacobian(z)
        return 0.5 * float(np.dot(values, values)), matrix.T @ values

    fit = minimize(
        objective,
        initial_z,
        jac=True,
        method="SLSQP",
        constraints=(eta_constraint,),
        options={"maxiter": args.max_nfev, "ftol": 1.0e-12, "disp": False},
    )
    final_residual, final_jacobian, final_timing = evaluate(fit.x)
    final_eta = eta_ref + basis @ fit.x
    final_theta = SrixTheta9.from_log_coordinates(final_eta)
    report = {
        "schema_version": 1,
        "method": "exploratory raw P43 M20 FEMU in provisional global SVD rank-7 basis",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "crop": list(CROP),
        "path_steps": len(path),
        "element_order": args.element_order,
        "scored_steps": list(scored),
        "observation_weighting": "none",
        "noise_model_used": False,
        "covariance_used": False,
        "residual_units": "mm",
        "parameterization": "admissible nine-parameter logarithmic coordinates",
        "rank": 7,
        "eta_reference": eta_ref.tolist(),
        "retained_basis": basis.tolist(),
        "discarded_basis": discarded.tolist(),
        "weak_coordinates_initial": (discarded.T @ (eta_ref - eta_ref)).tolist(),
        "singular_values_basis": np.sqrt(np.maximum(eigenvalues, 0.0)).tolist(),
        "shadow_step_sizes": steps.tolist(),
        "eta_bounds": {"low": low.tolist(), "high": high.tolist()},
        "constraint_type": "linear polytope: low <= eta_ref + V7 z <= high",
        "initial_z": initial_z.tolist(),
        "final_z": fit.x.tolist(),
        "final_eta": final_eta.tolist(),
        "final_parameters": final_theta.as_runtime_overrides(),
        "prior_rms_mm": history[0]["rms_mm"],
        "optimizer_residual_scale_mm": residual_scale,
        "gauss_newton_linear_step": gn_step.tolist(),
        "gauss_newton_predicted_relative_reduction": predicted_reduction,
        "final_rms_mm": float(np.sqrt(np.mean(final_residual**2))),
        "relative_rms_reduction": float(
            1.0 - np.sqrt(np.mean(final_residual**2)) / history[0]["rms_mm"]
        ),
        "optimizer": {
            "success": bool(fit.success), "status": int(fit.status),
            "message": str(fit.message), "nfev": int(fit.nfev),
            "njev": int(getattr(fit, "njev", -1)), "cost": float(fit.fun),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "evaluation_history": history,
        "final_timing": final_timing,
        "provenance": provenance,
        "claims": {
            "exploratory_only": True,
            "global_svd_qualified": False,
            "experimental_parameters_identified": False,
            "experimental_m100_authorized": False,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output / "trajectory.npz", residual=final_residual, jacobian=final_jacobian,
        eta=final_eta, z=fit.x, basis=basis, discarded_basis=discarded,
    )
    print(
        json.dumps(
            {
                "prior_rms_mm": report["prior_rms_mm"],
                "final_rms_mm": report["final_rms_mm"],
                "nfev": fit.nfev,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
