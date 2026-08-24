#!/usr/bin/env python3
"""Run deterministic multi-start synthetic P43 SRIX identification."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET
from fem_inhouse.identification.srix_equilibrium_gap import SrixTheta4
from scripts.qualify_srix_femu_direct_sensitivity import _direct_jacobian, _geometry
from scripts.qualify_srix_p0043_synthetic_smoke import (
    CROP,
    ROOT,
    H,
    _factory,
    _forward,
    _git,
    _Identity,
    _load_inputs,
    _make_path,
    _vector,
)
from scripts.qualify_srix_regm_twin import _theta_from_preset

DEFAULT_OUTPUT = ROOT / "validation/reference_data/p0043_synthetic_multistart_v1"
START_FACTORS = {
    "B1": (1.20, 0.80, 1.25, 0.75),
    "B2": (0.80, 1.20, 0.75, 1.25),
    "B3": (1.25, 1.25, 0.80, 0.80),
    "B4": (0.75, 0.75, 1.20, 1.20),
}


def _theta_from_factors(truth: SrixTheta4, factors: tuple[float, ...]) -> SrixTheta4:
    return SrixTheta4(
        tau0_mpa=truth.tau0_mpa * factors[0],
        r_mpa=truth.r_mpa * factors[1],
        q_mpa=truth.q_mpa * factors[2],
        b=truth.b * factors[3],
    )


def _run_start(
    name: str,
    initial: SrixTheta4,
    truth: SrixTheta4,
    target: list[np.ndarray],
    path: list[Any],
    angles: np.ndarray,
    library: str,
    threads: int,
    max_nfev: int,
) -> dict[str, Any]:
    scored = tuple(4 * index for index in range(1, 9))
    factory = _factory(angles, library, threads)
    identity = _Identity()
    cache: dict[bytes, tuple[list[Any], np.ndarray]] = {}
    forwards: list[dict[str, Any]] = []
    jacobians: list[dict[str, Any]] = []

    def evaluate(eta: np.ndarray) -> tuple[list[Any], np.ndarray]:
        key = np.asarray(eta, dtype=np.float64).tobytes()
        if key not in cache:
            theta = SrixTheta4.from_log_coordinates(eta)
            fields, timing = _forward(theta, path, angles, library, threads)
            residual = _vector(fields, scored, target)
            cache[key] = (fields, residual)
            forwards.append({"theta": theta.as_runtime_overrides(), **timing})
        return cache[key]

    initial_fields, initial_residual = evaluate(initial.log_coordinates())
    scale = max(float(np.linalg.norm(initial_residual)), 1.0e-30)

    def residual(eta: np.ndarray) -> np.ndarray:
        return evaluate(eta)[1] / scale

    def jacobian(eta: np.ndarray) -> np.ndarray:
        fields, _ = evaluate(eta)
        started = time.perf_counter()
        matrix, timing = _direct_jacobian(
            fields=fields,
            scored=scored,
            orientations=angles,
            theta=SrixTheta4.from_log_coordinates(eta),
            library=library,
            threads=threads,
            transfer=identity,
            h=H,
            material_factory=factory,
        )
        jacobians.append({"seconds": time.perf_counter() - started, **timing})
        return matrix / scale

    started = time.perf_counter()
    fit = least_squares(
        residual,
        initial.log_coordinates(),
        jac=jacobian,
        bounds=(truth.log_coordinates() - np.log(4.0), truth.log_coordinates() + np.log(4.0)),
        max_nfev=max_nfev,
        x_scale="jac",
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=1.0e-8,
    )
    final = SrixTheta4.from_log_coordinates(fit.x)
    final_fields, final_residual = evaluate(fit.x)
    matrix, timing = _direct_jacobian(
        fields=final_fields,
        scored=scored,
        orientations=angles,
        theta=final,
        library=library,
        threads=threads,
        transfer=identity,
        h=H,
        material_factory=factory,
    )
    geometry = _geometry(matrix)
    error = fit.x - truth.log_coordinates()
    right = np.asarray(geometry["right_singular_vectors"], dtype=float)
    projection = (right @ error).tolist()
    return {
        "name": name,
        "initial": initial.as_runtime_overrides(),
        "identified": final.as_runtime_overrides(),
        "log_error_initial": (initial.log_coordinates() - truth.log_coordinates()).tolist(),
        "log_error_identified": error.tolist(),
        "svd_projection_of_log_error": projection,
        "cost": {
            "initial_rms": float(np.sqrt(np.mean(_vector(initial_fields, scored, target) ** 2))),
            "identified_rms": float(np.sqrt(np.mean(final_residual**2))),
        },
        "optimizer": {
            "success": bool(fit.success),
            "message": str(fit.message),
            "nfev": int(fit.nfev),
            "njev": int(fit.njev or 0),
            "seconds": time.perf_counter() - started,
            "forward_evaluations": len(forwards),
            "jacobian_evaluations": len(jacobians) + 1,
        },
        "sensitivity": geometry,
        "forward_records": forwards,
        "jacobian_records": jacobians,
        "final_jacobian_timing": timing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=15)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    history, angles, provenance = _load_inputs()
    path = _make_path(history, 4)
    truth = _theta_from_preset()
    library = __import__("os").environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )
    truth_fields, truth_timing = _forward(truth, path, angles, library, args.threads)
    target = [np.asarray(field.displacement).copy() for field in truth_fields]
    results = []
    for name, factors in START_FACTORS.items():
        results.append(
            _run_start(
                name,
                _theta_from_factors(truth, factors),
                truth,
                target,
                path,
                angles,
                library,
                args.threads,
                args.max_nfev,
            )
        )
    report = {
        "schema_version": 1,
        "method": "P43 M20 synthetic direct FEMU identification multi-start",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "crop": list(CROP),
        "mesh": list(angles.shape[:2]),
        "path_steps": len(path),
        "scored_steps": [4 * index for index in range(1, 9)],
        "parameter_preset": DEFAULT_PARAMETER_SET,
        "shadow_fd_step": H,
        "max_nfev": args.max_nfev,
        "truth": truth.as_runtime_overrides(),
        "provenance": provenance,
        "truth_forward": truth_timing,
        "starts": results,
        "claims": {
            "synthetic_multistart_completed": True,
            "experimental_p43_authorized": False,
            "four_parameter_recovery_claimed": False,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {
        "starts": [
            {
                "name": item["name"],
                "rms": item["cost"]["identified_rms"],
                "success": item["optimizer"]["success"],
            }
            for item in results
        ]
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
