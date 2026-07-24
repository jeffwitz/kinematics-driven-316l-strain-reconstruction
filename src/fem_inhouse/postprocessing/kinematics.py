"""Shared DIC/FE kinematic post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
ShearConvention = Literal["tensorial", "engineering"]


@dataclass(frozen=True, slots=True)
class StrainComponents:
    """In-plane small-strain components on the nodal grid."""

    epsilon_xx: FloatArray
    epsilon_yy: FloatArray
    gamma_xy: FloatArray

    @property
    def epsilon_xy(self) -> FloatArray:
        return 0.5 * self.gamma_xy


def _as_matching_2d(first: ArrayLike, second: ArrayLike) -> tuple[FloatArray, FloatArray]:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("displacement fields must be two-dimensional")
    if a.shape != b.shape:
        raise ValueError("displacement fields must have the same shape")
    if min(a.shape) < 2:
        raise ValueError("each displacement-field axis must contain at least two nodes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("displacement fields must contain only finite values")
    return a, b


def strain_from_displacement(
    u_x: ArrayLike,
    u_y: ArrayLike,
    *,
    spacing_x: float,
    spacing_y: float,
) -> StrainComponents:
    """Compute small strain with array axis 0 = x and axis 1 = y."""

    if spacing_x <= 0 or spacing_y <= 0:
        raise ValueError("grid spacings must be positive")
    ux, uy = _as_matching_2d(u_x, u_y)

    dux_dx, dux_dy = np.gradient(ux, spacing_x, spacing_y)
    duy_dx, duy_dy = np.gradient(uy, spacing_x, spacing_y)
    return StrainComponents(
        epsilon_xx=dux_dx,
        epsilon_yy=duy_dy,
        gamma_xy=dux_dy + duy_dx,
    )


def cell_average(field: ArrayLike) -> FloatArray:
    """Average a nodal scalar field to structured element centres."""

    values = np.asarray(field, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("field must be a two-dimensional nodal grid")
    return 0.25 * (values[:-1, :-1] + values[1:, :-1] + values[:-1, 1:] + values[1:, 1:])


def plane_stress_equivalent_strain(
    epsilon_xx: ArrayLike,
    epsilon_yy: ArrayLike,
    shear: ArrayLike,
    *,
    poisson_ratio: float,
    shear_convention: ShearConvention = "tensorial",
) -> FloatArray:
    """Evaluate the 3D deviatoric equivalent strain under plane stress."""

    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must satisfy -1 < nu < 0.5")
    exx, eyy, shear_values = np.broadcast_arrays(
        np.asarray(epsilon_xx, dtype=float),
        np.asarray(epsilon_yy, dtype=float),
        np.asarray(shear, dtype=float),
    )
    if shear_convention == "tensorial":
        exy = shear_values
    elif shear_convention == "engineering":
        exy = 0.5 * shear_values
    else:
        raise ValueError("shear_convention must be 'tensorial' or 'engineering'")

    ezz = -poisson_ratio / (1.0 - poisson_ratio) * (exx + eyy)
    mean = (exx + eyy + ezz) / 3.0
    invariant = (
        np.square(exx - mean) + np.square(eyy - mean) + np.square(ezz - mean) + 2.0 * np.square(exy)
    )
    return np.sqrt(np.maximum((2.0 / 3.0) * invariant, 0.0))
