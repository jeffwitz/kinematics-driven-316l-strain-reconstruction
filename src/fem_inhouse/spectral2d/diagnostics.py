"""Diagnostics records for spectral plane-stress calculations."""

from __future__ import annotations

from dataclasses import dataclass, field


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
