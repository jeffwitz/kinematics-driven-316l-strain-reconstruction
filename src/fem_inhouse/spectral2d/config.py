"""Configuration for the full-Dirichlet spectral solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Spectral2DConfig:
    """Pre-registered numerical choices for the first spectral solver."""

    spatial_scheme: Literal["quad1", "tri2"] = "quad1"
    green_operator: Literal["b0", "two_mu", "c0"] = "b0"
    relative_equilibrium_tolerance: float = 1.0e-6
    maximum_fixed_point_iterations: int = 200
    anderson_enabled: bool = True
    anderson_memory: int = 4
    anderson_start_iteration: int = 2
    anderson_regularization: float = 1.0e-12
    relaxation_reduction: float = 0.5
    minimum_relaxation: float = 1.0 / 16.0
    armijo_coefficient: float = 1.0e-4
    maximum_cutbacks_per_increment: int = 8
    minimum_increment_fraction: float = 1.0 / 256.0
    reference_projection_tolerance: float = 1.0e-12
    symbol_null_tolerance: float = 1.0e-12
    reference_lambda_0: float = 1.0
    reference_mu_0: float = 1.0

    def __post_init__(self) -> None:
        if self.spatial_scheme not in {"quad1", "tri2"}:
            raise ValueError("spatial_scheme must be 'quad1' or 'tri2'")
        if self.green_operator not in {"b0", "two_mu", "c0"}:
            raise ValueError("green_operator must be 'b0', 'two_mu' or 'c0'")
        if self.relative_equilibrium_tolerance <= 0.0:
            raise ValueError("relative_equilibrium_tolerance must be positive")
        if self.maximum_fixed_point_iterations < 1:
            raise ValueError("maximum_fixed_point_iterations must be positive")
        if self.anderson_memory < 1 or self.anderson_start_iteration < 1:
            raise ValueError("Anderson memory and start iteration must be positive")
        if not 0.0 < self.relaxation_reduction < 1.0:
            raise ValueError("relaxation_reduction must lie in (0, 1)")
        if not 0.0 < self.minimum_relaxation <= 1.0:
            raise ValueError("minimum_relaxation must lie in (0, 1]")
        if self.maximum_cutbacks_per_increment < 0:
            raise ValueError("maximum_cutbacks_per_increment cannot be negative")
        if not 0.0 < self.minimum_increment_fraction <= 1.0:
            raise ValueError("minimum_increment_fraction must lie in (0, 1]")
        if self.reference_mu_0 <= 0.0 or self.reference_lambda_0 + self.reference_mu_0 <= 0.0:
            raise ValueError("reference parameters must satisfy mu>0 and lambda+mu>0")
