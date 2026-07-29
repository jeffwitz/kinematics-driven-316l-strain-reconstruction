"""Pure conversions between image, historical and canonical FEM conventions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def image_flow_to_canonical(
    flow_pixels: NDArray[np.generic],
    *,
    pixel_size_mm: float,
) -> FloatArray:
    """Convert image flow to this experiment's canonical transverse/tensile frame."""

    flow = np.asarray(flow_pixels, dtype=np.float64)
    if flow.ndim != 3 or flow.shape[-1] != 2 or not np.isfinite(flow).all():
        raise ValueError("flow_pixels must have finite shape (rows, columns, 2)")
    if not np.isfinite(pixel_size_mm) or pixel_size_mm <= 0.0:
        raise ValueError("pixel_size_mm must be finite and positive")
    return np.ascontiguousarray(
        np.stack((flow[..., 1], flow[..., 0]), axis=-1) * pixel_size_mm
    )


def canonical_to_image_flow(
    displacement_mm: NDArray[np.generic],
    *,
    pixel_size_mm: float,
) -> FloatArray:
    """Convert canonical ``(x,y,[ux,uy])`` to OpenCV image flow in pixels."""

    displacement = np.asarray(displacement_mm, dtype=np.float64)
    if (
        displacement.ndim != 3
        or displacement.shape[-1] != 2
        or not np.isfinite(displacement).all()
    ):
        raise ValueError("displacement_mm must have finite shape (x, y, 2)")
    if not np.isfinite(pixel_size_mm) or pixel_size_mm <= 0.0:
        raise ValueError("pixel_size_mm must be finite and positive")
    scaled = displacement / pixel_size_mm
    return np.ascontiguousarray(np.stack((scaled[..., 1], scaled[..., 0]), axis=-1))


def historical_uv_to_canonical(
    historical_u_pixels: NDArray[np.generic],
    historical_v_pixels: NDArray[np.generic],
    *,
    pixel_size_mm: float,
) -> FloatArray:
    """Convert received ``U=u_y, V=u_x`` arrays to canonical displacement."""

    historical_u = np.asarray(historical_u_pixels, dtype=np.float64)
    historical_v = np.asarray(historical_v_pixels, dtype=np.float64)
    if historical_u.ndim != 2 or historical_v.shape != historical_u.shape:
        raise ValueError("historical U and V must be matching two-dimensional arrays")
    if not np.isfinite(historical_u).all() or not np.isfinite(historical_v).all():
        raise ValueError("historical U and V must contain finite values")
    if not np.isfinite(pixel_size_mm) or pixel_size_mm <= 0.0:
        raise ValueError("pixel_size_mm must be finite and positive")
    return np.stack((historical_v, historical_u), axis=-1) * pixel_size_mm
