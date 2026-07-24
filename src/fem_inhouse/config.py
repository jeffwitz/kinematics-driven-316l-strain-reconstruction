"""Validated configuration objects for the supported 316L case study."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MaterialConfig:
    """Homogeneous material constants and Abaqus table settings.

    Stress-like quantities use MPa and strains are dimensionless.
    """

    young_modulus_mpa: float = 205_000.0
    poisson_ratio: float = 0.30
    hardening_exponent: float = 0.245
    plastic_strain_max: float = 0.2
    plastic_table_points: int = 1_000
    first_positive_plastic_strain: float = 1e-6

    def __post_init__(self) -> None:
        if self.young_modulus_mpa <= 0:
            raise ValueError("young_modulus_mpa must be positive")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must satisfy -1 < nu < 0.5")
        if self.hardening_exponent <= 0:
            raise ValueError("hardening_exponent must be positive")
        if self.plastic_strain_max <= 0:
            raise ValueError("plastic_strain_max must be positive")
        if self.plastic_table_points < 3:
            raise ValueError("plastic_table_points must be at least 3")
        if not 0 < self.first_positive_plastic_strain < self.plastic_strain_max:
            raise ValueError(
                "first_positive_plastic_strain must lie inside the plastic-strain range"
            )


@dataclass(frozen=True, slots=True)
class MeshConfig:
    """Structured pixel mesh settings."""

    nx: int
    ny: int
    base_pixel_size_mm: float = 0.001
    scale_factor: float = 1.84

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx and ny must be positive")
        if self.base_pixel_size_mm <= 0:
            raise ValueError("base_pixel_size_mm must be positive")
        if self.scale_factor <= 0:
            raise ValueError("scale_factor must be positive")

    @property
    def element_size_mm(self) -> float:
        return self.base_pixel_size_mm * self.scale_factor

    @property
    def physical_size_mm(self) -> tuple[float, float]:
        spacing = self.element_size_mm
        return self.nx * spacing, self.ny * spacing


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Nonlinear solution controls."""

    increments: int = 20
    max_newton_iterations: int = 15
    residual_tolerance: float = 1e-6
    minimum_step_divisor: int = 1_024
    require_pypardiso: bool = True

    def __post_init__(self) -> None:
        if self.increments < 1:
            raise ValueError("increments must be positive")
        if self.max_newton_iterations < 1:
            raise ValueError("max_newton_iterations must be positive")
        if not 0 < self.residual_tolerance < 1:
            raise ValueError("residual_tolerance must lie in (0, 1)")
        if self.minimum_step_divisor < 2:
            raise ValueError("minimum_step_divisor must be at least 2")


@dataclass(frozen=True, slots=True)
class CaseStudyConfig:
    """Complete configuration for one structured case-study solve."""

    mesh: MeshConfig
    material: MaterialConfig = field(default_factory=MaterialConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
