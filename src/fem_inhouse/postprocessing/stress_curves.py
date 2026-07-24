"""Direct and strain-reconstructed stress curves kept scientifically distinct."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def von_mises_stress(sigma_xx: ArrayLike, sigma_yy: ArrayLike, sigma_xy: ArrayLike) -> FloatArray:
    """Plane-stress von Mises equivalent stress."""

    sxx, syy, sxy = np.broadcast_arrays(
        np.asarray(sigma_xx, dtype=float),
        np.asarray(sigma_yy, dtype=float),
        np.asarray(sigma_xy, dtype=float),
    )
    value = np.square(sxx) - sxx * syy + np.square(syy) + 3.0 * np.square(sxy)
    return np.sqrt(np.maximum(value, 0.0))


def reconstructed_equivalent_stress(
    equivalent_strain: ArrayLike,
    *,
    young_modulus_mpa: float,
    poisson_ratio: float,
    yield_stress_mpa: float,
    hardening_coefficient_mpa: float,
    hardening_exponent: float,
) -> FloatArray:
    """Map equivalent strain through the scalar relation used in the article."""

    if young_modulus_mpa <= 0:
        raise ValueError("young_modulus_mpa must be positive")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must satisfy -1 < nu < 0.5")
    if yield_stress_mpa <= 0:
        raise ValueError("yield_stress_mpa must be positive")
    if hardening_coefficient_mpa < 0 or hardening_exponent <= 0:
        raise ValueError("invalid Ludwik hardening parameters")

    strain = np.asarray(equivalent_strain, dtype=float)
    if np.any(strain < 0):
        raise ValueError("equivalent strain cannot be negative")

    shear_modulus = young_modulus_mpa / (2.0 * (1.0 + poisson_ratio))
    yield_strain = yield_stress_mpa / (3.0 * shear_modulus)
    elastic = 3.0 * shear_modulus * strain
    plastic_strain = np.maximum(strain - yield_strain, 0.0)
    plastic = yield_stress_mpa + hardening_coefficient_mpa * np.power(
        plastic_strain, hardening_exponent
    )
    return np.where(strain <= yield_strain, elastic, plastic)


def direct_fe_equivalent_stress_curve(
    sigma_xx_fields: ArrayLike,
    sigma_yy_fields: ArrayLike,
    sigma_xy_fields: ArrayLike,
    *,
    spatial_axes: tuple[int, ...] = (-2, -1),
) -> FloatArray:
    """Average tensor components spatially, then compute direct FE von Mises."""

    sxx, syy, sxy = np.broadcast_arrays(
        np.asarray(sigma_xx_fields, dtype=float),
        np.asarray(sigma_yy_fields, dtype=float),
        np.asarray(sigma_xy_fields, dtype=float),
    )
    return von_mises_stress(
        np.nanmean(sxx, axis=spatial_axes),
        np.nanmean(syy, axis=spatial_axes),
        np.nanmean(sxy, axis=spatial_axes),
    )


def reconstructed_stress_curve_from_strain(
    equivalent_strain_fields: ArrayLike,
    *,
    spatial_axes: tuple[int, ...] = (-2, -1),
    young_modulus_mpa: float,
    poisson_ratio: float,
    yield_stress_mpa: float,
    hardening_coefficient_mpa: float,
    hardening_exponent: float,
) -> FloatArray:
    """Average equivalent strain spatially, then apply the scalar reconstruction."""

    mean_strain = np.nanmean(np.asarray(equivalent_strain_fields, dtype=float), axis=spatial_axes)
    return reconstructed_equivalent_stress(
        mean_strain,
        young_modulus_mpa=young_modulus_mpa,
        poisson_ratio=poisson_ratio,
        yield_stress_mpa=yield_stress_mpa,
        hardening_coefficient_mpa=hardening_coefficient_mpa,
        hardening_exponent=hardening_exponent,
    )
