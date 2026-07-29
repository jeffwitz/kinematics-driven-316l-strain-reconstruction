"""Explicit image-mask semantics for historical and maintained paths."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

ByteImage = NDArray[np.uint8]
MaskMode = Literal["binary_mask", "legacy_uint8_multiply"]


def declared_all_valid_mask(shape: tuple[int, int]) -> NDArray[np.bool_]:
    """Return the deterministic fallback authorised when no mask is archived."""

    if len(shape) != 2 or any(size < 1 for size in shape):
        raise ValueError("mask shape must contain two positive dimensions")
    return np.ones(shape, dtype=np.bool_)


def binary_mask(mask: NDArray[np.generic], *, shape: tuple[int, int]) -> NDArray[np.bool_]:
    """Validate and normalise a bool, 0/1 or 0/255 mask."""

    values = np.asarray(mask)
    if values.shape != shape or values.ndim != 2:
        raise ValueError(f"mask must have shape {shape}")
    if values.dtype == np.bool_:
        return np.array(values, dtype=np.bool_, copy=True)
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("binary mask must have boolean or integer dtype")
    unique = set(int(value) for value in np.unique(values))
    if unique <= {0, 1} or unique <= {0, 255}:
        return values != 0
    raise ValueError("binary mask values must be 0/1 or 0/255")


def apply_image_mask(
    image: NDArray[np.generic],
    mask: NDArray[np.generic],
    *,
    mode: MaskMode,
) -> ByteImage:
    """Apply maintained binary semantics or exact historical uint8 multiply."""

    values = np.asarray(image)
    if values.ndim != 2 or values.dtype != np.uint8:
        raise TypeError("image must be a two-dimensional uint8 array")
    if mode == "binary_mask":
        valid = binary_mask(mask, shape=values.shape)
        return np.where(valid, values, 0).astype(np.uint8, copy=False)
    if mode == "legacy_uint8_multiply":
        historical = np.asarray(mask)
        if historical.shape != values.shape or historical.dtype != np.uint8:
            raise TypeError("legacy mask must be uint8 with the image shape")
        return np.multiply(values, historical, dtype=np.uint8)
    raise ValueError(f"unsupported mask mode: {mode}")
