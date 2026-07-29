"""Photometric consistency diagnostics for direct DIC displacement fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

ByteImage = NDArray[np.uint8]
FloatArray = NDArray[np.float64]
BooleanArray = NDArray[np.bool_]


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("photometric diagnostics require the measurement dependency") from error
    return cv2


@dataclass(frozen=True, slots=True)
class PhotometricResidualResult:
    """Brightness-constancy residual and its geometrically valid support."""

    absolute_residual_grey_levels: FloatArray
    valid_mask: BooleanArray


def direct_photometric_residual(
    reference: NDArray[np.generic],
    current: NDArray[np.generic],
    flow_pixels: NDArray[np.generic],
) -> PhotometricResidualResult:
    """Evaluate ``abs(current(x + flow(x)) - reference(x))``.

    OpenCV flow ordering is used: component 0 is column displacement and
    component 1 is row displacement.
    """

    cv2 = _cv2()
    reference_image = np.asarray(reference)
    current_image = np.asarray(current)
    if (
        reference_image.ndim != 2
        or current_image.shape != reference_image.shape
        or reference_image.dtype != np.uint8
        or current_image.dtype != np.uint8
    ):
        raise TypeError("reference and current must be same-shape two-dimensional uint8 images")
    flow = np.asarray(flow_pixels, dtype=np.float64)
    if flow.shape != (*reference_image.shape, 2):
        raise ValueError("flow_pixels must have shape (*reference.shape, 2)")
    if not np.isfinite(flow).all():
        raise ValueError("flow_pixels must contain finite values")

    rows, columns = np.indices(reference_image.shape, dtype=np.float64)
    destination_columns = columns + flow[..., 0]
    destination_rows = rows + flow[..., 1]
    valid = (
        (destination_columns >= 0.0)
        & (destination_columns <= reference_image.shape[1] - 1.0)
        & (destination_rows >= 0.0)
        & (destination_rows <= reference_image.shape[0] - 1.0)
    )
    mapped = cv2.remap(
        current_image,
        destination_columns.astype(np.float32),
        destination_rows.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    residual = np.abs(
        np.asarray(mapped, dtype=np.float64)
        - np.asarray(reference_image, dtype=np.float64)
    )
    residual[~valid] = 0.0
    return PhotometricResidualResult(
        absolute_residual_grey_levels=np.ascontiguousarray(residual),
        valid_mask=np.ascontiguousarray(valid),
    )
