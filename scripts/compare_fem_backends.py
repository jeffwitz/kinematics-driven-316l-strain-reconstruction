#!/usr/bin/env python3
"""Run and preserve a DIC-driven FEM comparison of Python and MFront."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.config import CaseStudyConfig, MaterialConfig, MeshConfig, SolverConfig
from fem_inhouse.results import FEMResult
from fem_inhouse.solver import run_case_study

FIELD_NAMES = (
    "displacement_mm",
    "stress_mpa",
    "total_strain",
    "plastic_strain",
    "equivalent_plastic_strain",
    "reaction_force",
)
INPUT_NAMES = (
    "displacement_x_mm",
    "displacement_y_mm",
    "yield_stress_mpa",
    "hardening_coefficient_mpa",
)
THRESHOLDS = {
    "displacement_mm": 1e-6,
    "stress_mpa": 5e-4,
    "total_strain": 5e-4,
    "plastic_strain": 1e-3,
    "equivalent_plastic_strain": 1e-3,
    "reaction_force": 5e-4,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _load_inputs(directory: Path) -> dict[str, NDArray]:
    inputs = {name: np.load(directory / f"{name}.npy") for name in INPUT_NAMES}
    element_shape = inputs["yield_stress_mpa"].shape
    if inputs["hardening_coefficient_mpa"].shape != element_shape:
        raise ValueError("material maps do not share the same element shape")
    nodal_shape = (element_shape[0] + 1, element_shape[1] + 1)
    if inputs["displacement_x_mm"].shape != nodal_shape:
        raise ValueError("displacement_x_mm is incompatible with the material maps")
    if inputs["displacement_y_mm"].shape != nodal_shape:
        raise ValueError("displacement_y_mm is incompatible with the material maps")
    return inputs


def _arrays(result: FEMResult) -> dict[str, NDArray]:
    return {name: np.asarray(getattr(result, name)) for name in FIELD_NAMES}


def _field_metrics(reference: NDArray, prediction: NDArray) -> dict[str, float]:
    difference = prediction - reference
    reference_scale = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)
    reference_norm = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return {
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "relative_linf": float(np.max(np.abs(difference)) / reference_scale),
        "relative_l2": float(np.linalg.norm(difference) / reference_norm),
        "rmse": float(np.sqrt(np.mean(difference**2))),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("build/mfront/src/libBehaviour.so"),
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--increments", type=int, default=20)
    parser.add_argument("--max-newton-iterations", type=int, default=25)
    parser.add_argument("--residual-tolerance", type=float, default=1e-7)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.library.is_file():
        raise FileNotFoundError(f"MFront behaviour library not found: {args.library}")
    report_path = args.output / "report.json"
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty campaign: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    inputs = _load_inputs(args.input)
    nx, ny = inputs["yield_stress_mpa"].shape
    base_solver = SolverConfig(
        increments=args.increments,
        max_newton_iterations=args.max_newton_iterations,
        residual_tolerance=args.residual_tolerance,
        hardening_mode="ludwik",
        mfront_library=str(args.library),
        mfront_threads=args.threads,
    )
    base_config = CaseStudyConfig(
        mesh=MeshConfig(nx=nx, ny=ny),
        material=MaterialConfig(),
        solver=base_solver,
    )

    results: dict[str, FEMResult] = {}
    arrays: dict[str, dict[str, NDArray]] = {}
    for backend in ("python", "mfront"):
        config = replace(
            base_config,
            solver=replace(base_solver, constitutive_backend=backend),
        )
        results[backend] = run_case_study(config, **inputs)
        arrays[backend] = _arrays(results[backend])
        np.savez_compressed(
            args.output / f"{backend}_fields.npz",
            **arrays[backend],
        )
        diagnostics = results[backend].diagnostics
        (args.output / f"{backend}_diagnostics.json").write_text(
            json.dumps(asdict(diagnostics) if diagnostics is not None else None, indent=2) + "\n",
            encoding="utf-8",
        )

    metrics = {
        name: _field_metrics(arrays["python"][name], arrays["mfront"][name]) for name in FIELD_NAMES
    }
    passed = all(metrics[name]["relative_linf"] <= THRESHOLDS[name] for name in FIELD_NAMES)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "configuration": asdict(base_config),
        "comparison": {
            "reference_backend": "python",
            "prediction_backend": "mfront",
        },
        "inputs": {
            name: {
                "path": str(args.input / f"{name}.npy"),
                "sha256": _sha256(args.input / f"{name}.npy"),
                "shape": list(inputs[name].shape),
            }
            for name in INPUT_NAMES
        },
        "mfront_library": {
            "path": str(args.library),
            "sha256": _sha256(args.library),
        },
        "diagnostics": {
            backend: asdict(result.diagnostics) if result.diagnostics is not None else None
            for backend, result in results.items()
        },
        "thresholds": {
            name: {"metric": "relative_linf", "maximum": threshold}
            for name, threshold in THRESHOLDS.items()
        },
        "metrics": metrics,
        "passed": passed,
        "artifacts": {
            backend: {
                "filename": f"{backend}_fields.npz",
                "sha256": _sha256(args.output / f"{backend}_fields.npz"),
            }
            for backend in ("python", "mfront")
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
