"""Benchmark TRI2 Python-J2 Newton-GMRES policies on the registered P43 crop."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from fem_inhouse.core.plane_stress_material import PythonJ2PlaneStressBatch
from fem_inhouse.spectral2d import EBISpectralSolverConfig, StructuredGrid2D
from fem_inhouse.spectral2d.newton_two_state import solve_two_state_dirichlet_plane_stress
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig

DATA_ROOT = Path("data/processed/case_study")
PIXEL_SIZE_MM = 0.00184
DEFAULT_CROP = (1570, 1670, 1035, 1135)


def _load_case(
    mesh: int,
    crop: tuple[int, int, int, int],
) -> tuple[StructuredGrid2D, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x0, x1, y0, y1 = crop
    if x1 - x0 != mesh or y1 - y0 != mesh:
        raise ValueError("crop extent must equal --mesh")
    ux = np.load(DATA_ROOT / "displacement_x_mm.npy", mmap_mode="r")
    uy = np.load(DATA_ROOT / "displacement_y_mm.npy", mmap_mode="r")
    yield_stress = np.load(DATA_ROOT / "yield_stress_mpa.npy", mmap_mode="r")
    coefficient = np.load(DATA_ROOT / "hardening_coefficient_mpa.npy", mmap_mode="r")
    boundary = np.stack((ux[x0 : x1 + 1, y0 : y1 + 1], uy[x0 : x1 + 1, y0 : y1 + 1]), axis=-1)
    history = np.stack([fraction * boundary for fraction in np.linspace(0.0, 1.0, 9)])
    grid = StructuredGrid2D(mesh, mesh, mesh * PIXEL_SIZE_MM, mesh * PIXEL_SIZE_MM)
    return (
        grid,
        history,
        np.asarray(yield_stress[x0:x1, y0:y1]).reshape(-1),
        np.asarray(coefficient[x0:x1, y0:y1]).reshape(-1),
        boundary,
    )


def _run(
    *,
    mesh: int,
    crop: tuple[int, int, int, int],
    increments: int,
    tolerance: float,
    backend: str,
    workers: int,
    restart: int,
    linear_mode: str,
    reference_update: str,
) -> dict[str, object]:
    grid, history, yield_stress, coefficient, _ = _load_case(mesh, crop)
    material = PythonJ2PlaneStressBatch(
        np.repeat(yield_stress, 2),
        np.repeat(coefficient, 2),
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    history = history[: increments + 1]
    config = EBISpectralSolverConfig(
        relative_equilibrium_tolerance=tolerance,
        gmres_restart=restart,
        linear_tolerance_mode=linear_mode,  # type: ignore[arg-type]
        reference_update_mode=reference_update,  # type: ignore[arg-type]
        transform=SpectralTransformConfig(
            backend=backend,  # type: ignore[arg-type]
            workers=workers,
            fftw_planner_effort="estimate",
            fftw_use_wisdom=False,
        ),
    )
    started = time.perf_counter()
    result = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=material,
        boundary_displacement_history=history,
        config=config,
    )
    elapsed = time.perf_counter() - started
    diagnostics = result.diagnostics
    return {
        "mesh": mesh,
        "increments": increments,
        "tolerance": tolerance,
        "backend": backend,
        "workers": workers,
        "restart": restart,
        "linear_mode": linear_mode,
        "reference_update": reference_update,
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
        "provenance": diagnostics.provenance,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=int, default=50)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=DEFAULT_CROP)
    parser.add_argument("--restarts", nargs="+", type=int, default=[20, 30, 50, 80, 120])
    parser.add_argument("--linear-mode", choices=("fixed", "eisenstat_walker"), default="fixed")
    parser.add_argument(
        "--reference-update",
        choices=("initial", "per_increment", "per_newton"),
        default="initial",
    )
    parser.add_argument("--transform-backend", choices=("scipy", "fftw"), default="fftw")
    parser.add_argument("--transform-workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.mesh < 2 or arguments.increments < 1:
        raise SystemExit("mesh and increments must be positive")
    crop = tuple(arguments.crop_nodes)
    if crop[1] - crop[0] != arguments.mesh or crop[3] - crop[2] != arguments.mesh:
        raise SystemExit("crop extent must equal --mesh")
    runs = [
        _run(
            mesh=arguments.mesh,
            crop=crop,
            increments=arguments.increments,
            tolerance=arguments.tolerance,
            backend=arguments.transform_backend,
            workers=arguments.transform_workers,
            restart=restart,
            linear_mode=arguments.linear_mode,
            reference_update=arguments.reference_update,
        )
        for restart in arguments.restarts
    ]
    report = {
        "status": "completed_tri2_j2_krylov_sweep",
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "crop_nodes": list(crop),
        "runs": runs,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
