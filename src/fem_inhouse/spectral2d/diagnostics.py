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
    iteration_diagnostics: tuple[dict[str, float | int | bool], ...] = ()
    residual_ratios: tuple[float, ...] = ()
    verification_residual: float = 0.0
