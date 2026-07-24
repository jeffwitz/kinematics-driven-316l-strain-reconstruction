"""Typed result containers for finite-element case-study solves."""

from __future__ import annotations

from dataclasses import dataclass, field

from numpy.typing import NDArray

FloatArray = NDArray


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Fields recorded at one pseudo-time."""

    stress_mpa: FloatArray
    total_strain: FloatArray
    equivalent_plastic_strain: FloatArray
    displacement_mm: FloatArray


@dataclass(frozen=True, slots=True)
class SolverDiagnostics:
    """Convergence and timing data for one completed nonlinear solve."""

    backend: str
    elapsed_seconds: float
    initialization_seconds: float
    elastic_assembly_seconds: float
    constitutive_seconds: float
    tangent_assembly_seconds: float
    linear_solve_seconds: float
    output_seconds: float
    attempted_increments: int
    converged_increments: int
    cutbacks: int
    total_newton_iterations: int
    maximum_newton_iterations: int
    final_residual_norm: float
    final_relative_residual: float
    final_convergence_criterion: str


@dataclass(frozen=True, slots=True)
class FEMResult:
    """Final fields returned by the supported structured CPS4 solve."""

    displacement_mm: FloatArray
    stress_mpa: FloatArray
    total_strain: FloatArray
    plastic_strain: FloatArray
    equivalent_plastic_strain: FloatArray
    reaction_force: FloatArray
    frames: dict[float, FrameResult] = field(default_factory=dict)
    diagnostics: SolverDiagnostics | None = None

    def arrays(self) -> tuple[FloatArray, ...]:
        """Return every final-state array for common validation operations."""

        return (
            self.displacement_mm,
            self.stress_mpa,
            self.total_strain,
            self.plastic_strain,
            self.equivalent_plastic_strain,
            self.reaction_force,
        )
