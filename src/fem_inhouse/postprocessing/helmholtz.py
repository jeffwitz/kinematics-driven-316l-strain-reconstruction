"""Scalar Helmholtz filtering on structured element-centred fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import dctn, idctn

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HelmholtzFilterResult:
    """Source, solution and numerical checks for one Helmholtz filter."""

    source_element_field: FloatArray
    filtered_element_field: FloatArray
    length_scale_mm: float
    spacing_x_mm: float
    spacing_y_mm: float
    mean_drift: float
    residual_relative: float


def _positive_neumann_laplacian(
    field: FloatArray,
    *,
    spacing_x_mm: float,
    spacing_y_mm: float,
) -> FloatArray:
    """Apply the positive cell-centred ``-Laplacian`` with omitted boundary faces."""

    result = np.zeros_like(field)
    if field.shape[0] > 1:
        jumps_x = (field[:-1, :] - field[1:, :]) / spacing_x_mm**2
        result[:-1, :] += jumps_x
        result[1:, :] -= jumps_x
    if field.shape[1] > 1:
        jumps_y = (field[:, :-1] - field[:, 1:]) / spacing_y_mm**2
        result[:, :-1] += jumps_y
        result[:, 1:] -= jumps_y
    return result


def helmholtz_filter_element_field(
    field: ArrayLike,
    *,
    length_scale_mm: float,
    spacing_x_mm: float,
    spacing_y_mm: float,
) -> HelmholtzFilterResult:
    """Filter a two-dimensional element field with homogeneous Neumann flux.

    Array axis 0 is the physical x direction and axis 1 is the physical y
    direction. All lengths are expressed in millimetres.
    """

    source = np.array(field, dtype=np.float64, copy=True)
    if source.ndim != 2:
        raise ValueError(f"field must be strictly two-dimensional, got shape {source.shape}")
    if source.size == 0:
        raise ValueError("field must not be empty")
    if not np.isfinite(source).all():
        raise ValueError("field must contain only finite values")

    parameters = {
        "length_scale_mm": length_scale_mm,
        "spacing_x_mm": spacing_x_mm,
        "spacing_y_mm": spacing_y_mm,
    }
    for name, value in parameters.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if length_scale_mm < 0:
        raise ValueError("length_scale_mm must be nonnegative")
    if spacing_x_mm <= 0 or spacing_y_mm <= 0:
        raise ValueError("spacing_x_mm and spacing_y_mm must be strictly positive")

    if length_scale_mm == 0.0:
        filtered = source.copy()
        return HelmholtzFilterResult(
            source_element_field=source,
            filtered_element_field=filtered,
            length_scale_mm=0.0,
            spacing_x_mm=float(spacing_x_mm),
            spacing_y_mm=float(spacing_y_mm),
            mean_drift=0.0,
            residual_relative=0.0,
        )

    nx, ny = source.shape
    wave_x = np.arange(nx, dtype=np.float64)
    wave_y = np.arange(ny, dtype=np.float64)
    eigenvalues_x = (2.0 - 2.0 * np.cos(np.pi * wave_x / nx)) / spacing_x_mm**2
    eigenvalues_y = (2.0 - 2.0 * np.cos(np.pi * wave_y / ny)) / spacing_y_mm**2
    denominator = 1.0 + length_scale_mm**2 * (
        eigenvalues_x[:, np.newaxis] + eigenvalues_y[np.newaxis, :]
    )
    transformed = dctn(source, type=2, norm="ortho")
    filtered = np.asarray(
        idctn(transformed / denominator, type=2, norm="ortho"),
        dtype=np.float64,
    )

    residual = (
        filtered
        + length_scale_mm**2
        * _positive_neumann_laplacian(
            filtered,
            spacing_x_mm=spacing_x_mm,
            spacing_y_mm=spacing_y_mm,
        )
        - source
    )
    source_norm = float(np.linalg.norm(source))
    residual_norm = float(np.linalg.norm(residual))
    if source_norm == 0.0:
        residual_relative = 0.0 if residual_norm == 0.0 else float("inf")
    else:
        residual_relative = residual_norm / source_norm

    return HelmholtzFilterResult(
        source_element_field=source,
        filtered_element_field=filtered,
        length_scale_mm=float(length_scale_mm),
        spacing_x_mm=float(spacing_x_mm),
        spacing_y_mm=float(spacing_y_mm),
        mean_drift=float(np.mean(filtered) - np.mean(source)),
        residual_relative=residual_relative,
    )
