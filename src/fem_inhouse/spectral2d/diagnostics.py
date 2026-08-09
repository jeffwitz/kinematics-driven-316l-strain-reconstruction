"""Diagnostics records for spectral plane-stress calculations."""

from __future__ import annotations

import datetime
import importlib.metadata
import json
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
    krylov_method: str = "gmres"
    krylov_recycling: bool = False


@dataclass(frozen=True, slots=True)
class LoadStepAttemptDiagnostics:
    """Exclusive cost attribution for one accepted or rejected load-step attempt."""

    attempt_index: int
    load_fraction_start: float
    load_fraction_end: float
    accepted: bool
    failure_reason: str | None
    newton_iterations: int
    linear_solves: int
    krylov_outer_callbacks: int
    jacobian_matvec_calls: int
    preconditioner_calls: int
    krylov_seconds: float
    jacobian_seconds: float
    preconditioner_seconds: float
    krylov_overhead_seconds: float
    material_seconds: float
    material_evaluations: int
    material_integration_seconds: float
    material_condensation_seconds: float
    mgis_integrations: int
    line_search_rejections: int
    minimum_line_search_factor: float


def summarize_load_step_attempts(
    attempts: tuple[LoadStepAttemptDiagnostics, ...] | list[LoadStepAttemptDiagnostics],
) -> dict[str, dict[str, int | float]]:
    """Aggregate mutually exclusive attempt costs for reporting."""

    result: dict[str, dict[str, int | float]] = {}
    for label, selected in (
        ("accepted", [item for item in attempts if item.accepted]),
        ("rejected", [item for item in attempts if not item.accepted]),
        ("total", list(attempts)),
    ):
        result[label] = {
            "attempts": len(selected),
            "newton_iterations": sum(item.newton_iterations for item in selected),
            "linear_solves": sum(item.linear_solves for item in selected),
            "krylov_outer_callbacks": sum(
                item.krylov_outer_callbacks for item in selected
            ),
            "jacobian_matvec_calls": sum(
                item.jacobian_matvec_calls for item in selected
            ),
            "preconditioner_calls": sum(item.preconditioner_calls for item in selected),
            "krylov_seconds": sum(item.krylov_seconds for item in selected),
            "jacobian_seconds": sum(item.jacobian_seconds for item in selected),
            "preconditioner_seconds": sum(
                item.preconditioner_seconds for item in selected
            ),
            "krylov_overhead_seconds": sum(
                item.krylov_overhead_seconds for item in selected
            ),
            "material_seconds": sum(item.material_seconds for item in selected),
            "material_evaluations": sum(
                item.material_evaluations for item in selected
            ),
            "material_integration_seconds": sum(
                item.material_integration_seconds for item in selected
            ),
            "material_condensation_seconds": sum(
                item.material_condensation_seconds for item in selected
            ),
            "mgis_integrations": sum(item.mgis_integrations for item in selected),
            "line_search_rejections": sum(
                item.line_search_rejections for item in selected
            ),
        }
    return result


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
    krylov_method: str = "gmres",
    krylov_recycling: bool = False,
    lgmres_inner_m: int | None = None,
    lgmres_outer_k: int | None = None,
    gcrotmk_m: int | None = None,
    gcrotmk_k: int | None = None,
    reference_update_mode: str | None = None,
    krylov_blas_threads: int | None = 1,
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

    try:
        from threadpoolctl import threadpool_info  # type: ignore[import-untyped]
    except ImportError:
        blas_threadpools = []
    else:
        blas_threadpools = [
            {
                "filepath": item.get("filepath"),
                "internal_api": item.get("internal_api"),
                "internal_api_num_threads": item.get("num_threads"),
                "prefix": item.get("prefix"),
                "user_api": item.get("user_api"),
            }
            for item in threadpool_info()
            if item.get("user_api") == "blas"
        ]
    blas_backends = ",".join(
        str(item["internal_api"])
        for item in blas_threadpools
        if item.get("internal_api") is not None
    ) or None
    blas_effective_threads = ",".join(
        str(item["internal_api_num_threads"])
        for item in blas_threadpools
        if item.get("internal_api_num_threads") is not None
    ) or None

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
        "krylov_blas_threads": krylov_blas_threads,
        "blas_backend": blas_backends,
        "blas_library_threads": blas_effective_threads,
        "krylov_blas_limit_applied": krylov_blas_threads is not None,
        "blas_threadpools_json": json.dumps(blas_threadpools, sort_keys=True),
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
        "krylov_method": krylov_method,
        "krylov_recycling": krylov_recycling,
        "lgmres_inner_m": lgmres_inner_m,
        "lgmres_outer_k": lgmres_outer_k,
        "gcrotmk_m": gcrotmk_m,
        "gcrotmk_k": gcrotmk_k,
        "reference_update_mode": reference_update_mode,
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
    material_local_iteration_histogram: tuple[int, ...] = ()
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
    adaptive_stepping_enabled: bool = False
    adaptive_step_history: tuple[dict[str, object], ...] = ()
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
    load_step_attempts: tuple[LoadStepAttemptDiagnostics, ...] = ()
    reference_updates: tuple[dict[str, str | int | float | bool], ...] = ()
    provenance: dict[str, str | int | float | bool | None] = field(default_factory=dict)
