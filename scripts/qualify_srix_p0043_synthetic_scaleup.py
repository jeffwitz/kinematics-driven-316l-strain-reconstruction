#!/usr/bin/env python3
"""Scale the synthetic P43 identification from M20 to the registered M100 crop."""

from __future__ import annotations

import argparse
import json
import os
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

FULL_CROP = (1580, 1680, 1030, 1130)
DEFAULT_OUTPUT = ROOT / "validation/reference_data/p0043_synthetic_scaleup_v1"


def _theta_from_report(path: Path, truth: SrixTheta4) -> SrixTheta4:
    report = json.loads(path.read_text(encoding="utf-8"))
    starts = report.get("starts", [])
    if not starts:
        raise ValueError("multi-start report contains no starts")
    chosen = min(starts, key=lambda item: item["cost"]["identified_rms"])
    values = chosen["identified"]
    return SrixTheta4(
        tau0_mpa=float(values["tau0_mpa"]),
        r_mpa=float(values["R_mpa"]),
        q_mpa=float(values["Q_mpa"]),
        b=float(values["b"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial-report",
        type=Path,
        default=ROOT / "validation/reference_data/p0043_synthetic_multistart_v1/report.json",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--maximum-evaluations", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    history, angles, provenance = _load_inputs(FULL_CROP)
    path = _make_path(history, 4)
    truth = _theta_from_preset()
    initial = _theta_from_report(args.initial_report, truth)
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )
    truth_fields, truth_timing = _forward(truth, path, angles, library, args.threads)
    target = [np.asarray(field.displacement).copy() for field in truth_fields]
    factory = _factory(angles, library, args.threads)
    identity = _Identity()
    scored = tuple(4 * index for index in range(1, 9))
    cache: dict[bytes, tuple[list[Any], np.ndarray]] = {}
    forwards: list[dict[str, Any]] = []
    jacobians: list[dict[str, Any]] = []

    def evaluate(eta: np.ndarray) -> tuple[list[Any], np.ndarray]:
        key = np.asarray(eta, dtype=np.float64).tobytes()
        if key not in cache:
            theta = SrixTheta4.from_log_coordinates(eta)
            fields, timing = _forward(theta, path, angles, library, args.threads)
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
            threads=args.threads,
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
        max_nfev=args.maximum_evaluations,
        x_scale="jac",
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=1.0e-8,
    )
    identified = SrixTheta4.from_log_coordinates(fit.x)
    identified_fields, identified_residual = evaluate(fit.x)
    matrix, final_timing = _direct_jacobian(
        fields=identified_fields,
        scored=scored,
        orientations=angles,
        theta=identified,
        library=library,
        threads=args.threads,
        transfer=identity,
        h=H,
        material_factory=factory,
    )
    report = {
        "schema_version": 1,
        "method": "P43 M100 synthetic direct FEMU scale-up initialized from M20",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "crop": list(FULL_CROP),
        "mesh": list(angles.shape[:2]),
        "path_steps": len(path),
        "scored_steps": list(scored),
        "parameter_preset": DEFAULT_PARAMETER_SET,
        "shadow_fd_step": H,
        "initial_report": str(args.initial_report),
        "truth": truth.as_runtime_overrides(),
        "initial_from_m20": initial.as_runtime_overrides(),
        "identified": identified.as_runtime_overrides(),
        "provenance": provenance,
        "truth_forward": truth_timing,
        "forward_records": forwards,
        "jacobian_records": jacobians,
        "final_jacobian_timing": final_timing,
        "optimizer": {
            "success": bool(fit.success),
            "message": str(fit.message),
            "nfev": int(fit.nfev),
            "njev": int(fit.njev or 0),
            "seconds": time.perf_counter() - started,
        },
        "cost": {
            "initial_rms": float(np.sqrt(np.mean(initial_residual**2))),
            "identified_rms": float(np.sqrt(np.mean(identified_residual**2))),
        },
        "sensitivity": _geometry(matrix),
        "claims": {
            "scaleup_completed": True,
            "experimental_p43_authorized": False,
            "four_parameter_recovery_claimed": False,
        },
    }
    np.savez_compressed(
        output / "fields.npz",
        truth_displacement=np.asarray([field.displacement for field in truth_fields]),
        initial_displacement=np.asarray([field.displacement for field in initial_fields]),
        identified_displacement=np.asarray([field.displacement for field in identified_fields]),
        initial_residual=initial_residual,
        identified_residual=identified_residual,
        jacobian=matrix,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "cost": report["cost"],
                "identified": report["identified"],
                "optimizer": report["optimizer"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
