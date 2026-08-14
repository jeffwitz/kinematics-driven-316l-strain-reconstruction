"""Synthetic qualification cases for the DIC-compatible mechanical oracle."""

from __future__ import annotations

from math import isfinite

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.workflows.experimental_mechanical_oracle import (
    FixedIncrementEquilibriumResult,
    solve_fixed_plastic_increment_equilibrium,
)

FloatArray = NDArray[np.float64]


def diagonal_localised_plastic_increment(
    grid: StructuredGrid2D,
    *,
    points_per_pixel: int,
    background: float = 2.0e-4,
    amplitude: float = 1.5e-3,
    width_pixels: float = 1.5,
    slope: float = 0.45,
    offset: float = 0.0,
) -> FloatArray:
    """Create a smooth diagonal band without injecting checkerboard modes."""

    if points_per_pixel < 1:
        raise ValueError("points_per_pixel must be positive")
    for name, value in (("background", background), ("amplitude", amplitude)):
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not isfinite(width_pixels) or width_pixels <= 0.0:
        raise ValueError("width_pixels must be finite and positive")
    if not isfinite(slope) or not isfinite(offset):
        raise ValueError("band slope and offset must be finite")
    x = np.arange(grid.nx, dtype=np.float64)[:, None] + 0.5
    y = np.arange(grid.ny, dtype=np.float64)[None, :] + 0.5
    centre_x = 0.5 * grid.nx
    centre_y = 0.5 * grid.ny + offset
    distance = (y - centre_y) - slope * (x - centre_x)
    distance /= np.sqrt(1.0 + slope**2)
    band = background + amplitude * np.exp(
        -0.5 * (distance / width_pixels) ** 2
    )
    return np.repeat(band[..., None], points_per_pixel, axis=-1)



__all__ = [
    "FixedIncrementEquilibriumResult",
    "diagonal_localised_plastic_increment",
    "solve_fixed_plastic_increment_equilibrium",
]
