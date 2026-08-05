"""Qualify the selected TRI2 Python-J2 policy on the registered P43 100x100 crop."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from benchmark_tri2_j2_krylov import (  # type: ignore[import-not-found]
    DATA_ROOT,
    DEFAULT_CROP,
    PIXEL_SIZE_MM,
    _load_case,
)

from fem_inhouse.core.plane_stress_material import PythonJ2PlaneStressBatch
from fem_inhouse.spectral2d import EBISpectralSolverConfig
from fem_inhouse.spectral2d.newton_two_state import solve_two_state_dirichlet_plane_stress
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig

REFERENCE_COMMIT = "4c14f744caf8f673024e293afd2e521dea12e371"
REGISTERED_REFERENCE = {
    "elapsed_seconds": 2084.386,
    "newton_iterations": 83,
    "gmres_iterations": 10677,
    "final_residual": 6.431849320206941e-09,
    "jacobian_seconds": 1393.954,
    "preconditioner_seconds": 11.804,
}


def _hash_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _solve(
    *,
    mesh: int,
    crop: tuple[int, int, int, int],
    increments: int,
    tolerance: float,
    restart: int,
    linear_mode: str,
    reference_update: str,
    output_directory: Path,
    name: str,
) -> dict[str, Any]:
    grid, history, yield_stress, coefficient, _ = _load_case(mesh, crop)
    material = PythonJ2PlaneStressBatch(
        np.repeat(yield_stress, 2),
        np.repeat(coefficient, 2),
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    config = EBISpectralSolverConfig(
        relative_equilibrium_tolerance=tolerance,
        gmres_restart=restart,
        linear_tolerance_mode=linear_mode,  # type: ignore[arg-type]
        reference_update_mode=reference_update,  # type: ignore[arg-type]
        transform=SpectralTransformConfig(
            backend="fftw",
            workers=1,
            fftw_planner_effort="measure",
            fftw_planning_time_limit_s=2.0,
            fftw_use_wisdom=False,
        ),
    )
    started = time.perf_counter()
    result = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=material,
        boundary_displacement_history=history[: increments + 1],
        config=config,
    )
    elapsed = time.perf_counter() - started
    assert result.plastic_strain_tensor is not None
    fields: dict[str, np.ndarray] = {
        "displacement": result.displacement,
        "stress_in_plane_mpa": result.stress_in_plane_mpa,
        "plastic_strain_tensor": result.plastic_strain_tensor,
        "equivalent_plastic_strain": result.observables["equivalent_plastic_strain"],
        "reaction_forces": result.reaction_forces,
    }
    field_path = output_directory / f"tri2_j2_p43_m{mesh}_{name}_fields.npz"
    np.savez_compressed(field_path, **fields)  # type: ignore[arg-type]
    diagnostics = result.diagnostics
    return {
        "name": name,
        "elapsed_seconds": elapsed,
        "newton_iterations": int(sum(diagnostics.iterations_per_increment)),
        "iterations_per_increment": list(diagnostics.iterations_per_increment),
        "gmres_iterations": int(diagnostics.timings["gmres_iterations"]),
        "final_residual": diagnostics.verification_residual,
        "timings": diagnostics.timings,
        "linear_solves": [
            {
                "increment": entry.increment,
                "newton_iteration": entry.newton_iteration,
                "nonlinear_residual_before": entry.nonlinear_residual_before,
                "requested_relative_tolerance": entry.requested_relative_tolerance,
                "gmres_iterations": entry.gmres_iterations,
                "jacobian_calls": entry.jacobian_calls,
                "preconditioner_calls": entry.preconditioner_calls,
                "gmres_seconds": entry.gmres_seconds,
                "jacobian_seconds": entry.jacobian_seconds,
                "preconditioner_seconds": entry.preconditioner_seconds,
                "krylov_overhead_seconds": entry.krylov_overhead_seconds,
                "line_search_factor": entry.line_search_factor,
                "linear_residual_ratio": entry.linear_residual_ratio,
            }
            for entry in diagnostics.linear_solves
        ],
        "reference_updates": list(diagnostics.reference_updates),
        "provenance": diagnostics.provenance,
        "field_file": str(field_path),
        "field_sha256": {name: _hash_array(values) for name, values in fields.items()},
    }


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(float(np.linalg.norm(reference)), 1.0e-30)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=DEFAULT_CROP)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    crop = tuple(arguments.crop_nodes)
    mesh = crop[1] - crop[0]
    if mesh != crop[3] - crop[2]:
        raise SystemExit("P43 crop must be square")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    baseline = _solve(
        mesh=mesh,
        crop=crop,
        increments=arguments.increments,
        tolerance=arguments.tolerance,
        restart=50,
        linear_mode="fixed",
        reference_update="initial",
        output_directory=arguments.output.parent,
        name="baseline",
    )
    optimized = _solve(
        mesh=mesh,
        crop=crop,
        increments=arguments.increments,
        tolerance=arguments.tolerance,
        restart=20,
        linear_mode="eisenstat_walker",
        reference_update="per_newton",
        output_directory=arguments.output.parent,
        name="optimized",
    )
    baseline_fields = np.load(baseline["field_file"])
    optimized_fields = np.load(optimized["field_file"])
    correctness = {
        key: _relative_error(optimized_fields[key], baseline_fields[key])
        for key in baseline_fields.files
    }
    report = {
        "status": "completed_tri2_j2_p43_performance_qualification",
        "reference_commit": REFERENCE_COMMIT,
        "data_root": str(DATA_ROOT),
        "crop_nodes": list(crop),
        "mesh": [mesh, mesh],
        "increments": arguments.increments,
        "tolerance": arguments.tolerance,
        "pixel_size_mm": PIXEL_SIZE_MM,
        "cpu": platform.processor(),
        "baseline": baseline,
        "optimized": optimized,
        "correctness_optimized_vs_current_baseline": correctness,
        "performance": {
            "total_speedup": baseline["elapsed_seconds"] / optimized["elapsed_seconds"],
            "jacobian_speedup": baseline["timings"]["jacobian_seconds"]
            / optimized["timings"]["jacobian_seconds"],
            "gmres_iteration_reduction": 1.0
            - optimized["gmres_iterations"] / baseline["gmres_iterations"],
            "optimized_vs_registered_reference": {
                "total_speedup": REGISTERED_REFERENCE["elapsed_seconds"]
                / optimized["elapsed_seconds"],
                "gmres_iteration_reduction": 1.0
                - optimized["gmres_iterations"]
                / REGISTERED_REFERENCE["gmres_iterations"],
                "newton_iteration_change": optimized["newton_iterations"]
                - REGISTERED_REFERENCE["newton_iterations"],
            },
        },
        "registered_reference_metrics": REGISTERED_REFERENCE,
    }
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    comparison_path = arguments.output.with_name(
        "tri2_j2_p43_m100_comparison.json"
    )
    comparison_path.write_text(
        json.dumps(
            {
                "reference": REFERENCE_COMMIT,
                "candidate": optimized["name"],
                "correctness_vs_current_baseline": correctness,
                "performance": report["performance"],
                "final_residual": optimized["final_residual"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
