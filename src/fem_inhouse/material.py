"""Ludwik-Hollomon material utilities used by the supported case study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LudwikLaw:
    """Scalar Ludwik-Hollomon hardening law."""

    yield_stress_mpa: float
    hardening_coefficient_mpa: float
    hardening_exponent: float

    def __post_init__(self) -> None:
        if self.yield_stress_mpa <= 0:
            raise ValueError("yield_stress_mpa must be positive")
        if self.hardening_coefficient_mpa < 0:
            raise ValueError("hardening_coefficient_mpa must be non-negative")
        if self.hardening_exponent <= 0:
            raise ValueError("hardening_exponent must be positive")

    def stress(self, equivalent_plastic_strain: ArrayLike) -> FloatArray:
        strain = np.asarray(equivalent_plastic_strain, dtype=float)
        if np.any(strain < 0):
            raise ValueError("equivalent plastic strain cannot be negative")
        return self.yield_stress_mpa + self.hardening_coefficient_mpa * np.power(
            strain, self.hardening_exponent
        )

    def tangent(self, equivalent_plastic_strain: ArrayLike) -> FloatArray:
        strain = np.asarray(equivalent_plastic_strain, dtype=float)
        if np.any(strain < 0):
            raise ValueError("equivalent plastic strain cannot be negative")
        positive = strain > 0
        tangent = np.zeros_like(strain, dtype=float)
        tangent[positive] = (
            self.hardening_coefficient_mpa
            * self.hardening_exponent
            * np.power(strain[positive], self.hardening_exponent - 1.0)
        )
        return tangent


def abaqus_plastic_table(
    law: LudwikLaw,
    *,
    plastic_strain_max: float = 0.2,
    n_points: int = 1_000,
    first_positive_strain: float | None = 1e-6,
) -> FloatArray:
    """Return ``[stress, plastic_strain]`` rows for the Abaqus material table.

    The article specifies 1000 points over ``[0, 0.2]`` and mentions a
    minimum positive increment of ``1e-6``. Until the original input generator
    is recovered, the grid is explicit: zero is the first point, the requested
    minimum positive strain is the second point, and the remaining points are
    linearly spaced up to ``plastic_strain_max``.
    """

    if plastic_strain_max <= 0:
        raise ValueError("plastic_strain_max must be positive")
    if n_points < 3:
        raise ValueError("n_points must be at least 3")

    if first_positive_strain is None:
        plastic_strain = np.linspace(0.0, plastic_strain_max, n_points)
    else:
        if not 0 < first_positive_strain < plastic_strain_max:
            raise ValueError("first_positive_strain must lie in (0, plastic_strain_max)")
        plastic_strain = np.concatenate(
            (
                np.array([0.0]),
                np.linspace(first_positive_strain, plastic_strain_max, n_points - 1),
            )
        )

    return np.column_stack((law.stress(plastic_strain), plastic_strain))
