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
from fem_inhouse.core.tensor_reconstruction import reconstruct_python_plane_stress_state
from fem_inhouse.postprocessing import (
    instantaneous_equivalent_plastic_strain,
    plane_stress_equivalent_strain,
    reconstructed_equivalent_strain,
    von_mises_from_stress_tensor,
    von_mises_stress,
)
from fem_inhouse.results import FEMResult
from fem_inhouse.solver import run_case_study

HISTORICAL_FIELD_NAMES = (
    "displacement_mm",
    "stress_mpa",
    "total_strain",
    "plastic_strain",
    "equivalent_plastic_strain",
    "reaction_force",
)
TENSOR_FIELD_NAMES = (
    "stress_tensor_mpa",
    "total_strain_tensor",
    "elastic_strain_tensor",
    "plastic_strain_tensor",
    "plane_stress_residual_mpa",
)
DERIVED_FIELD_NAMES = (
    "EVM_HISTORICAL",
    "EVM_RECONSTRUCTED_3D",
    "PEEQ_TENSOR_INSTANTANEOUS",
    "MISES_3D_MPA",
)
FIELD_NAMES = HISTORICAL_FIELD_NAMES + TENSOR_FIELD_NAMES + DERIVED_FIELD_NAMES
COMPARISON_FIELD_NAMES = HISTORICAL_FIELD_NAMES + TENSOR_FIELD_NAMES[:-1] + DERIVED_FIELD_NAMES
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
    "stress_tensor_mpa": 5e-4,
    "total_strain_tensor": 5e-4,
    "elastic_strain_tensor": 5e-4,
    "plastic_strain_tensor": 1e-3,
    "EVM_HISTORICAL": 5e-4,
    "EVM_RECONSTRUCTED_3D": 5e-4,
    "PEEQ_TENSOR_INSTANTANEOUS": 1e-3,
    "MISES_3D_MPA": 5e-4,
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


def _full_tensor_arrays(result: FEMResult) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    values = (
        result.stress_tensor_mpa,
        result.total_strain_tensor,
        result.elastic_strain_tensor,
        result.plastic_strain_tensor,
        result.plane_stress_residual_mpa,
    )
    if any(value is None for value in values):
        raise ValueError("the solver result does not contain a complete reconstructed state")
    stress, total, elastic, plastic, residual = values
    assert stress is not None
    assert total is not None
    assert elastic is not None
    assert plastic is not None
    assert residual is not None
    return stress, total, elastic, plastic, residual


def _arrays(result: FEMResult, *, poisson_ratio: float) -> dict[str, NDArray]:
    stress, total, elastic, plastic, residual = _full_tensor_arrays(result)
    arrays = {name: np.asarray(getattr(result, name)) for name in HISTORICAL_FIELD_NAMES}
    arrays.update(
        {
            "stress_tensor_mpa": stress,
            "total_strain_tensor": total,
            "elastic_strain_tensor": elastic,
            "plastic_strain_tensor": plastic,
            "plane_stress_residual_mpa": residual,
        }
    )
    arrays["EVM_HISTORICAL"] = plane_stress_equivalent_strain(
        result.total_strain[..., 0],
        result.total_strain[..., 1],
        result.total_strain[..., 2],
        poisson_ratio=poisson_ratio,
        shear_convention="engineering",
    )
    arrays["EVM_RECONSTRUCTED_3D"] = reconstructed_equivalent_strain(total)
    arrays["PEEQ_TENSOR_INSTANTANEOUS"] = instantaneous_equivalent_plastic_strain(plastic)
    arrays["MISES_3D_MPA"] = von_mises_from_stress_tensor(stress)
    return arrays


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


def _state_consistency(
    result: FEMResult,
    arrays: dict[str, NDArray],
    *,
    poisson_ratio: float,
) -> dict[str, float]:
    _, total, elastic, plastic, plane_stress_residual = _full_tensor_arrays(result)
    additive_residual = total - elastic - plastic
    analytical = reconstruct_python_plane_stress_state(
        result.total_strain,
        result.plastic_strain,
        result.stress_mpa,
        poisson_ratio,
    )
    historical_mises = von_mises_stress(
        result.stress_mpa[..., 0],
        result.stress_mpa[..., 1],
        result.stress_mpa[..., 2],
    )
    return {
        "maximum_abs_plane_stress_residual_mpa": float(np.max(np.abs(plane_stress_residual))),
        "maximum_abs_plastic_trace": float(np.max(np.abs(np.trace(plastic, axis1=-2, axis2=-1)))),
        "maximum_abs_additive_residual": float(np.max(np.abs(additive_residual))),
        "maximum_abs_mises_3d_minus_historical_mpa": float(
            np.max(np.abs(arrays["MISES_3D_MPA"] - historical_mises))
        ),
        "maximum_abs_native_minus_analytical_total_strain": float(
            np.max(np.abs(total - analytical.total_strain_tensor))
        ),
        "maximum_abs_native_minus_analytical_elastic_strain": float(
            np.max(np.abs(elastic - analytical.elastic_strain_tensor))
        ),
        "maximum_abs_native_minus_analytical_plastic_strain": float(
            np.max(np.abs(plastic - analytical.plastic_strain_tensor))
        ),
    }


def _historical_regression(
    reference_directory: Path,
    arrays: dict[str, dict[str, NDArray]],
) -> dict[str, dict[str, dict[str, float]]]:
    regression: dict[str, dict[str, dict[str, float]]] = {}
    for backend in ("python", "mfront"):
        reference_path = reference_directory / f"{backend}_fields.npz"
        if not reference_path.is_file():
            raise FileNotFoundError(f"historical reference not found: {reference_path}")
        with np.load(reference_path) as reference:
            missing = [name for name in HISTORICAL_FIELD_NAMES if name not in reference]
            if missing:
                raise ValueError(
                    f"historical reference {reference_path} is missing fields: {missing}"
                )
            regression[backend] = {
                name: _field_metrics(reference[name], arrays[backend][name])
                for name in HISTORICAL_FIELD_NAMES
            }
    return regression


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
    parser.add_argument(
        "--historical-reference",
        type=Path,
        help="optional campaign containing legacy Python/MFront fields",
    )
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
        results[backend] = run_case_study(
            config,
            displacement_x_mm=inputs["displacement_x_mm"],
            displacement_y_mm=inputs["displacement_y_mm"],
            yield_stress_mpa=inputs["yield_stress_mpa"],
            hardening_coefficient_mpa=inputs["hardening_coefficient_mpa"],
        )
        arrays[backend] = _arrays(
            results[backend],
            poisson_ratio=base_config.material.poisson_ratio,
        )
        np.savez_compressed(
            args.output / f"{backend}_fields.npz",
            **arrays[backend],  # type: ignore[arg-type]
        )
        diagnostics = results[backend].diagnostics
        (args.output / f"{backend}_diagnostics.json").write_text(
            json.dumps(asdict(diagnostics) if diagnostics is not None else None, indent=2) + "\n",
            encoding="utf-8",
        )

    metrics = {
        name: _field_metrics(arrays["python"][name], arrays["mfront"][name])
        for name in COMPARISON_FIELD_NAMES
    }
    consistency = {
        backend: _state_consistency(
            results[backend],
            arrays[backend],
            poisson_ratio=base_config.material.poisson_ratio,
        )
        for backend in ("python", "mfront")
    }
    historical_regression = (
        _historical_regression(args.historical_reference, arrays)
        if args.historical_reference is not None
        else None
    )
    field_comparison_passed = all(
        metrics[name]["relative_linf"] <= THRESHOLDS[name] for name in COMPARISON_FIELD_NAMES
    )
    invariants_passed = all(
        values["maximum_abs_plastic_trace"] <= 1e-10
        and values["maximum_abs_additive_residual"] <= 1e-10
        and values["maximum_abs_plane_stress_residual_mpa"]
        <= 1e-6 * max(1.0, float(np.max(arrays[backend]["MISES_3D_MPA"])))
        for backend, values in consistency.items()
    )
    historical_regression_passed = historical_regression is None or all(
        field["maximum_absolute_error"] <= 1e-12
        for backend in historical_regression.values()
        for field in backend.values()
    )
    passed = field_comparison_passed and invariants_passed and historical_regression_passed

    report = {
        "schema_version": 2,
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
        "consistency": consistency,
        "historical_regression": {
            "reference_directory": (
                str(args.historical_reference) if args.historical_reference is not None else None
            ),
            "maximum_absolute_error_limit": 1e-12,
            "metrics": historical_regression,
            "passed": historical_regression_passed,
        },
        "checks": {
            "field_comparison_passed": field_comparison_passed,
            "invariants_passed": invariants_passed,
            "historical_regression_passed": historical_regression_passed,
        },
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
