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


@dataclass(frozen=True, slots=True)
class LocalizationOverlapMetrics:
    """Overlap of high-localization pixels selected independently per field."""

    top_fraction: float
    reference_threshold: float
    prediction_threshold: float
    reference_count: int
    prediction_count: int
    intersection_count: int
    intersection_over_union: float
    dice_coefficient: float
    reference_recall: float
    prediction_precision: float


@dataclass(frozen=True, slots=True)
class FieldAcceptanceThresholds:
    """Pre-declared scalar acceptance thresholds for one field comparison."""

    maximum_rmse: float
    maximum_mae: float
    minimum_correlation: float
    minimum_localization_iou: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.maximum_rmse) or self.maximum_rmse < 0:
            raise ValueError("maximum_rmse must be finite and nonnegative")
        if not np.isfinite(self.maximum_mae) or self.maximum_mae < 0:
            raise ValueError("maximum_mae must be finite and nonnegative")
        if not -1 <= self.minimum_correlation <= 1:
            raise ValueError("minimum_correlation must lie in [-1, 1]")
        if not 0 <= self.minimum_localization_iou <= 1:
            raise ValueError("minimum_localization_iou must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class FieldComparisonReport:
    """Metrics, thresholds and decision for one co-registered field pair."""

    errors: FieldErrorMetrics
    localization: LocalizationOverlapMetrics
    thresholds: FieldAcceptanceThresholds
    passed: bool


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


def localization_overlap_metrics(
    reference: ArrayLike,
    prediction: ArrayLike,
    *,
    top_fraction: float = 0.1,
    mask: ArrayLike | None = None,
) -> LocalizationOverlapMetrics:
    """Compare independently thresholded high-localization zones.

    Each threshold is the ``1 - top_fraction`` quantile of the corresponding
    valid field. Values equal to the threshold are retained, so tied pixels can
    make the selected fraction slightly larger than requested.
    """

    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must lie in (0, 1]")
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
    quantile = 1.0 - top_fraction
    reference_threshold = float(np.quantile(reference_valid, quantile))
    prediction_threshold = float(np.quantile(prediction_valid, quantile))
    reference_zone = reference_valid >= reference_threshold
    prediction_zone = prediction_valid >= prediction_threshold
    intersection_count = int(np.count_nonzero(reference_zone & prediction_zone))
    reference_count = int(np.count_nonzero(reference_zone))
    prediction_count = int(np.count_nonzero(prediction_zone))
    union_count = reference_count + prediction_count - intersection_count
    return LocalizationOverlapMetrics(
        top_fraction=top_fraction,
        reference_threshold=reference_threshold,
        prediction_threshold=prediction_threshold,
        reference_count=reference_count,
        prediction_count=prediction_count,
        intersection_count=intersection_count,
        intersection_over_union=intersection_count / union_count,
        dice_coefficient=2.0 * intersection_count / (reference_count + prediction_count),
        reference_recall=intersection_count / reference_count,
        prediction_precision=intersection_count / prediction_count,
    )


def signed_difference_field(
    reference: ArrayLike,
    prediction: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> np.ndarray:
    """Return ``prediction - reference``, with invalid or excluded values as NaN."""

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
    difference = np.full(reference_values.shape, np.nan)
    difference[valid] = prediction_values[valid] - reference_values[valid]
    return difference


def evaluate_field_comparison(
    reference: ArrayLike,
    prediction: ArrayLike,
    thresholds: FieldAcceptanceThresholds,
    *,
    top_fraction: float = 0.1,
    mask: ArrayLike | None = None,
) -> FieldComparisonReport:
    """Evaluate a co-registered field against thresholds declared in advance."""

    errors = field_error_metrics(reference, prediction, mask=mask)
    localization = localization_overlap_metrics(
        reference,
        prediction,
        top_fraction=top_fraction,
        mask=mask,
    )
    passed = (
        errors.rmse <= thresholds.maximum_rmse
        and errors.mae <= thresholds.maximum_mae
        and errors.pearson_correlation >= thresholds.minimum_correlation
        and localization.intersection_over_union >= thresholds.minimum_localization_iou
    )
    return FieldComparisonReport(
        errors=errors,
        localization=localization,
        thresholds=thresholds,
        passed=passed,
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
