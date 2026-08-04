"""Configuration for the full-Dirichlet spectral solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Spectral2DConfig:
    """Pre-registered numerical choices for the first spectral solver."""

    spatial_scheme: Literal["one_point", "two_subcell"] = "one_point"
    green_operator: Literal["b0", "two_mu", "c0"] = "b0"
    relative_equilibrium_tolerance: float = 1.0e-6
    maximum_fixed_point_iterations: int = 200
    anderson_enabled: bool = True
    anderson_target: Literal["none", "displacement", "polarization"] = "polarization"
    anderson_memory: int = 4
    anderson_start_iteration: int = 4
    anderson_period: int = 2
    anderson_regularization: float = 1.0e-12
    update_safeguard: Literal["published_none", "monotone_armijo", "nonmonotone"] = (
        "published_none"
    )
    catastrophic_residual_growth_factor: float = 1.0e3
    relaxation_reduction: float = 0.5
    minimum_relaxation: float = 1.0 / 16.0
    armijo_coefficient: float = 1.0e-4
    maximum_cutbacks_per_increment: int = 8
    minimum_increment_fraction: float = 1.0 / 256.0
    reference_projection_tolerance: float = 1.0e-12
    symbol_null_tolerance: float = 1.0e-12
    record_high_frequency_diagnostics: bool = True
    reference_parameter_mode: Literal["explicit", "projected"] = "projected"
    reference_parameter_scale: float = 1.0
    reference_lambda_0: float | None = None
    reference_mu_0: float | None = None

    def __post_init__(self) -> None:
        if self.spatial_scheme not in {"one_point", "two_subcell"}:
            raise ValueError("spatial_scheme must be 'one_point' or 'two_subcell'")
        if self.green_operator not in {"b0", "two_mu", "c0"}:
            raise ValueError("green_operator must be 'b0', 'two_mu' or 'c0'")
        if self.relative_equilibrium_tolerance <= 0.0:
            raise ValueError("relative_equilibrium_tolerance must be positive")
        if self.maximum_fixed_point_iterations < 1:
            raise ValueError("maximum_fixed_point_iterations must be positive")
        if self.anderson_memory < 1 or self.anderson_start_iteration < 1:
            raise ValueError("Anderson memory and start iteration must be positive")
        if self.anderson_period < 1:
            raise ValueError("Anderson period must be positive")
        if self.anderson_target not in {"none", "displacement", "polarization"}:
            raise ValueError("unsupported Anderson target")
        if self.update_safeguard not in {"published_none", "monotone_armijo", "nonmonotone"}:
            raise ValueError("unsupported update safeguard")
        if self.catastrophic_residual_growth_factor <= 1.0:
            raise ValueError("catastrophic residual growth factor must exceed one")
        if not 0.0 < self.relaxation_reduction < 1.0:
            raise ValueError("relaxation_reduction must lie in (0, 1)")
        if not 0.0 < self.minimum_relaxation <= 1.0:
            raise ValueError("minimum_relaxation must lie in (0, 1]")
        if self.maximum_cutbacks_per_increment < 0:
            raise ValueError("maximum_cutbacks_per_increment cannot be negative")
        if not 0.0 < self.minimum_increment_fraction <= 1.0:
            raise ValueError("minimum_increment_fraction must lie in (0, 1]")
        if self.reference_parameter_mode not in {"explicit", "projected"}:
            raise ValueError("unsupported reference parameter mode")
        if self.reference_parameter_scale <= 0.0:
            raise ValueError("reference parameter scale must be positive")
        if self.reference_parameter_mode == "explicit":
            if self.reference_lambda_0 is None or self.reference_mu_0 is None:
                raise ValueError("explicit reference parameters require lambda_0 and mu_0")
            if self.reference_mu_0 <= 0.0 or self.reference_lambda_0 + self.reference_mu_0 <= 0.0:
                raise ValueError("reference parameters must satisfy mu>0 and lambda+mu>0")
        elif self.reference_lambda_0 is not None or self.reference_mu_0 is not None:
            raise ValueError("projected reference parameters reject explicit values")
