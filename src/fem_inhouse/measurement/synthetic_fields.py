"""Controlled displacement fields for image-chain metrology."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def gaussian_gradient_band(
    shape: tuple[int, int],
    *,
    fwhm_pixels: float,
    orientation: str,
    peak_gradient: float,
    maximum_displacement_pixels: float = 2.0,
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Return an integrated Gaussian band at a declared peak gradient."""

    if len(shape) != 2 or any(size < 3 for size in shape):
        raise ValueError("shape must contain two dimensions >= 3")
    if not np.isfinite(fwhm_pixels) or fwhm_pixels <= 0.0:
        raise ValueError("fwhm_pixels must be finite and positive")
    if not np.isfinite(peak_gradient) or peak_gradient <= 0.0:
        raise ValueError("peak_gradient must be finite and positive")
    if not np.isfinite(maximum_displacement_pixels) or maximum_displacement_pixels <= 0:
        raise ValueError("maximum_displacement_pixels must be finite and positive")
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("orientation must be horizontal or vertical")

    size = shape[1] if orientation == "horizontal" else shape[0]
    coordinate = np.arange(size, dtype=np.float64)
    centre = 0.5 * (size - 1)
    sigma = fwhm_pixels / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gradient = np.exp(-0.5 * np.square((coordinate - centre) / sigma))
    gradient *= peak_gradient / np.max(gradient)
    cumulative = np.cumsum(gradient)
    cumulative -= cumulative[0]
    if cumulative[-1] >= maximum_displacement_pixels:
        raise ValueError(
            f"integrated displacement {cumulative[-1]:.6g} px is not below "
            f"{maximum_displacement_pixels:.6g} px"
        )
    flow = np.zeros((*shape, 2), dtype=np.float32)
    if orientation == "horizontal":
        flow[..., 0] = cumulative[None, :]
    else:
        flow[..., 1] = cumulative[:, None]
    return flow, gradient
