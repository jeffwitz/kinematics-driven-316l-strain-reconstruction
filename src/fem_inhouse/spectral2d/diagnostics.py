"""Diagnostics records for spectral plane-stress calculations."""

from __future__ import annotations

import datetime
import importlib.metadata
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field

from fem_inhouse.spectral2d.transforms import TransformDiagnostics


@dataclass(slots=True)
class JacobianActionDiagnostics:
    """Detailed timing counters for one matrix-free Jacobian action."""

    calls: int = 0
    total_seconds: float = 0.0
    unpack_seconds: float = 0.0
    gradient_seconds: float = 0.0
    tangent_seconds: float = 0.0
    divergence_seconds: float = 0.0
    pack_seconds: float = 0.0


@dataclass(slots=True)
class PreconditionerActionDiagnostics:
    """Detailed timing counters for one preconditioner action family."""

    calls: int = 0
    total_seconds: float = 0.0
    reshape_seconds: float = 0.0
    forward_transform_seconds: float = 0.0
    green_seconds: float = 0.0
    inverse_transform_seconds: float = 0.0
    output_copy_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class LinearSolveDiagnostics:
    """Cost and call counts for one Newton linear solve."""

    increment: int
    newton_iteration: int
    nonlinear_residual_before: float
    requested_relative_tolerance: float
    gmres_info: int
    gmres_iterations: int
    jacobian_calls: int
    preconditioner_calls: int
    gmres_seconds: float
    jacobian_seconds: float
    preconditioner_seconds: float
    krylov_overhead_seconds: float
    restart: int
    line_search_factor: float | None
    linear_residual_ratio: float | None = None


def collect_runtime_provenance(
    transform: TransformDiagnostics,
    *,
    gmres_restart: int,
    gmres_maximum_iterations: int,
    gmres_relative_tolerance: float,
    linear_tolerance_mode: str = "fixed",
    forcing_initial: float | None = None,
    forcing_minimum: float | None = None,
    forcing_maximum: float | None = None,
    forcing_gamma: float | None = None,
    forcing_alpha: float | None = None,
) -> dict[str, str | int | float | bool | None]:
    """Collect reproducibility metadata for a spectral solve."""

    try:
        import pyfftw  # type: ignore[import-untyped]
    except ImportError:
        pyfftw_version: str | None = None
        fftw_version: str | None = None
    else:
        pyfftw_version = str(getattr(pyfftw, "__version__", "unknown"))
        fftw_version = str(getattr(pyfftw, "fftw_version", "unknown"))

    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        cpu_physical = None
    else:
        cpu_physical = psutil.cpu_count(logical=False)

    try:
        commit_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except OSError:
        commit_sha = None

    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "commit_sha": commit_sha,
        "date_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "python": sys.version.split()[0],
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
        "pyfftw": pyfftw_version,
        "fftw": fftw_version,
        "cpu_logical": os.cpu_count(),
        "cpu_physical": cpu_physical,
        "fftw_threads": transform.workers if transform.backend == "fftw" else None,
        "fftw_planner": transform.planner_effort,
        "fftw_wisdom_loaded": transform.wisdom_loaded,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "transform_backend": transform.backend,
        "gmres_restart": gmres_restart,
        "gmres_maximum_iterations": gmres_maximum_iterations,
        "gmres_relative_tolerance": gmres_relative_tolerance,
        "linear_tolerance_mode": linear_tolerance_mode,
        "forcing_initial": forcing_initial,
        "forcing_minimum": forcing_minimum,
        "forcing_maximum": forcing_maximum,
        "forcing_gamma": forcing_gamma,
        "forcing_alpha": forcing_alpha,
    }


@dataclass(frozen=True, slots=True)
class Spectral2DDiagnostics:
    spatial_scheme: str
    green_operator: str
    pixels: tuple[int, int]
    material_points: int
    points_per_pixel: int
    spacing_x: float
    spacing_y: float
    relative_residual_history: tuple[float, ...] = ()
    dimensionless_equilibrium_history: tuple[float, ...] = ()
    absolute_residual_history: tuple[float, ...] = ()
    iterations_per_increment: tuple[int, ...] = ()
    cutbacks: int = 0
    anderson_proposals: int = 0
    anderson_accelerated_proposals: int = 0
    anderson_resets: int = 0
    minimum_relaxation: float = 1.0
    maximum_plane_stress_residual_mpa: float = 0.0
    total_seconds: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    reference_lambda_0: float = 0.0
    reference_mu_0: float = 0.0
    reference_projection_error: float = 0.0
    iteration_diagnostics: tuple[dict[str, float | int | bool], ...] = ()
    residual_ratios: tuple[float, ...] = ()
    verification_residual: float = 0.0
    verification_residual_history: tuple[float, ...] = ()
    verification_relative_mismatch_history: tuple[float, ...] = ()
    highest_mode_energy_history: tuple[float, ...] = ()
    high_frequency_energy_fraction_history: tuple[float, ...] = ()
    fluctuation_norm_history: tuple[float, ...] = ()
    highest_mode_residual_history: tuple[float, ...] = ()
    active_slip_systems_history: tuple[int, ...] = ()
    transform_backend: str = "scipy"
    transform_implementation: str = "scipy.fft.dstn"
    transform_interior_shape: tuple[int, int] = (0, 0)
    transform_batch_components: int = 2
    transform_dtype: str = "float64"
    transform_workers: int = 1
    transform_planner_effort: str | None = None
    transform_wisdom_loaded: bool = False
    transform_planning_seconds: float = 0.0
    linear_solves: tuple[LinearSolveDiagnostics, ...] = ()
    reference_updates: tuple[dict[str, str | int | float | bool], ...] = ()
    provenance: dict[str, str | int | float | bool | None] = field(default_factory=dict)
