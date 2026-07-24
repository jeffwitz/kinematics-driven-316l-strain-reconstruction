"""Quantitative full-field and partition-interface comparison metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from fem_inhouse.partitioning import PartitionLayout


@dataclass(frozen=True, slots=True)
class FieldErrorMetrics:
    """Scalar metrics computed on the same valid pixels of two fields."""

    count: int
    rmse: float
    mae: float
    signed_mean_error: float
    maximum_absolute_error: float
    relative_l2_error: float
    pearson_correlation: float


def field_error_metrics(
    reference: ArrayLike,
    prediction: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> FieldErrorMetrics:
    """Compare equal-shaped fields after excluding non-finite or masked values."""

    reference_values = np.asarray(reference, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    if reference_values.shape != prediction_values.shape:
        raise ValueError("reference and prediction must have the same shape")
    valid = np.isfinite(reference_values) & np.isfinite(prediction_values)
    if mask is not None:
        mask_values = np.asarray(mask, dtype=bool)
        if mask_values.shape != reference_values.shape:
            raise ValueError("mask must have the same shape as the compared fields")
        valid &= mask_values
    if not valid.any():
        raise ValueError("no valid values remain for comparison")

    reference_valid = reference_values[valid]
    prediction_valid = prediction_values[valid]
    difference = prediction_valid - reference_valid
    reference_norm = float(np.linalg.norm(reference_valid))
    difference_norm = float(np.linalg.norm(difference))
    if reference_norm == 0:
        relative_l2 = 0.0 if difference_norm == 0 else float("inf")
    else:
        relative_l2 = difference_norm / reference_norm

    reference_centered = reference_valid - reference_valid.mean()
    prediction_centered = prediction_valid - prediction_valid.mean()
    correlation_denominator = float(
        np.linalg.norm(reference_centered) * np.linalg.norm(prediction_centered)
    )
    if correlation_denominator == 0:
        correlation = 1.0 if np.array_equal(reference_valid, prediction_valid) else float("nan")
    else:
        correlation = float(
            np.dot(reference_centered, prediction_centered) / correlation_denominator
        )
    return FieldErrorMetrics(
        count=int(valid.sum()),
        rmse=float(np.sqrt(np.mean(difference**2))),
        mae=float(np.mean(np.abs(difference))),
        signed_mean_error=float(np.mean(difference)),
        maximum_absolute_error=float(np.max(np.abs(difference))),
        relative_l2_error=relative_l2,
        pearson_correlation=correlation,
    )


def interface_gradient_ratio(field: ArrayLike, layout: PartitionLayout) -> float:
    """Return mean interface gradient divided by the whole-field mean gradient.

    A value near one means gradients at stitched core boundaries are typical of
    the field. Values above one indicate enhanced jumps at partition seams.
    This explicit definition matches the qualitative BGE description in the
    article; it must not be labelled as the article's exact BGE until the
    original analysis script or formula is recovered.
    """

    values = np.asarray(field, dtype=float)
    element_shape = layout.global_shape
    node_shape = (element_shape[0] + 1, element_shape[1] + 1)
    if values.shape == element_shape:
        boundary_offset = -1
    elif values.shape == node_shape:
        boundary_offset = 0
    else:
        raise ValueError(
            f"field has shape {values.shape}, expected element shape {element_shape} "
            f"or node shape {node_shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("field contains non-finite values")

    gradient_x = np.abs(np.diff(values, axis=0))
    gradient_y = np.abs(np.diff(values, axis=1))
    boundaries_x = sorted(
        {partition.core_bounds[0] for partition in layout if partition.core_bounds[0] > 0}
    )
    boundaries_y = sorted(
        {partition.core_bounds[2] for partition in layout if partition.core_bounds[2] > 0}
    )
    noise_floor = np.finfo(float).eps * max(float(np.max(np.abs(values))), 1.0) * 100

    def normalized(
        values_at_interface: np.ndarray,
        directional_gradients: np.ndarray,
    ) -> np.ndarray:
        if float(directional_gradients.max()) <= noise_floor:
            if np.any(values_at_interface > noise_floor):
                return np.full(values_at_interface.size, np.inf)
            return np.ones(values_at_interface.size)
        return values_at_interface.ravel() / float(directional_gradients.mean())

    interface_pieces = [
        *(
            normalized(gradient_x[boundary + boundary_offset, :], gradient_x)
            for boundary in boundaries_x
        ),
        *(
            normalized(gradient_y[:, boundary + boundary_offset], gradient_y)
            for boundary in boundaries_y
        ),
    ]
    if not interface_pieces:
        raise ValueError("layout has no internal partition interface")
    return float(np.concatenate(interface_pieces).mean())
