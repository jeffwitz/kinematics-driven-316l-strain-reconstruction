"""Integrated equilibrium diagnostics for cell-centred plane-stress fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SectionEquilibriumResult:
    """Section resultants and the conservative balance between adjacent rows."""

    y_mm: FloatArray
    section_force_n: FloatArray
    lateral_shear_flux_n_per_mm: FloatArray
    interval_balance_residual_n: FloatArray
    section_force_mean_n: float
    section_force_relative_dispersion: float
    naive_force_increment_rms_n: float
    balance_residual_rms_n: float
    balance_residual_relative_l2: float
    balance_residual_relative_to_mean_force: float
    boundary_flux_closure_gain: float
    spacing_x_mm: float
    spacing_y_mm: float
    thickness_mm: float


def integrated_section_equilibrium(
    stress_mpa: np.ndarray,
    *,
    spacing_x_mm: float,
    spacing_y_mm: float,
    thickness_mm: float,
    y_origin_mm: float = 0.0,
) -> SectionEquilibriumResult:
    r"""Evaluate integrated vertical equilibrium on a rectangular cell grid.

    The historical stress convention is ``[..., (s11, s22, s12)]`` and array
    axes are ``(x, y)``.  For an interior subdomain the section force need not
    be constant because shear can cross its artificial lateral boundaries.
    The conservative interval residual therefore discretises

    .. math::

       N(y_{j+1})-N(y_j)
       + \int_{y_j}^{y_{j+1}} t\,[\sigma_{12}(x_R,y)
       -\sigma_{12}(x_L,y)]\,dy = 0.

    Boundary tractions are approximated from the first and last cell-centred
    stresses.  The result is a post-processing baseline, not an FE residual at
    quadrature-point accuracy.
    """

    values = np.asarray(stress_mpa, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("stress_mpa must have shape (nx, ny, 3)")
    if values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("stress_mpa must contain at least two cells per direction")
    if not np.isfinite(values).all():
        raise ValueError("stress_mpa contains non-finite values")
    for name, value in (
        ("spacing_x_mm", spacing_x_mm),
        ("spacing_y_mm", spacing_y_mm),
        ("thickness_mm", thickness_mm),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive")
    if not np.isfinite(y_origin_mm):
        raise ValueError("y_origin_mm must be finite")

    sigma_yy = values[..., 1]
    sigma_xy = values[..., 2]
    section_force = thickness_mm * spacing_x_mm * np.sum(sigma_yy, axis=0)
    lateral_flux = thickness_mm * (sigma_xy[-1, :] - sigma_xy[0, :])
    flux_midpoint = 0.5 * (lateral_flux[:-1] + lateral_flux[1:])
    force_increment = np.diff(section_force)
    flux_increment = spacing_y_mm * flux_midpoint
    balance_residual = force_increment + flux_increment

    force_mean = float(np.mean(section_force))
    force_scale = max(abs(force_mean), np.finfo(np.float64).tiny)
    force_dispersion = float(np.std(section_force) / force_scale)
    residual_scale = max(
        float(np.linalg.norm(force_increment) + np.linalg.norm(flux_increment)),
        np.finfo(np.float64).tiny,
    )
    residual_relative = float(np.linalg.norm(balance_residual) / residual_scale)
    naive_rms = float(np.sqrt(np.mean(np.square(force_increment))))
    residual_rms = float(np.sqrt(np.mean(np.square(balance_residual))))
    closure_gain = (
        float(1.0 - residual_rms / naive_rms)
        if naive_rms > np.finfo(np.float64).tiny
        else 0.0
    )
    y = y_origin_mm + spacing_y_mm * (np.arange(values.shape[1], dtype=np.float64) + 0.5)

    return SectionEquilibriumResult(
        y_mm=y,
        section_force_n=np.asarray(section_force, dtype=np.float64),
        lateral_shear_flux_n_per_mm=np.asarray(lateral_flux, dtype=np.float64),
        interval_balance_residual_n=np.asarray(balance_residual, dtype=np.float64),
        section_force_mean_n=force_mean,
        section_force_relative_dispersion=force_dispersion,
        naive_force_increment_rms_n=naive_rms,
        balance_residual_rms_n=residual_rms,
        balance_residual_relative_l2=residual_relative,
        balance_residual_relative_to_mean_force=residual_rms / force_scale,
        boundary_flux_closure_gain=closure_gain,
        spacing_x_mm=float(spacing_x_mm),
        spacing_y_mm=float(spacing_y_mm),
        thickness_mm=float(thickness_mm),
    )
