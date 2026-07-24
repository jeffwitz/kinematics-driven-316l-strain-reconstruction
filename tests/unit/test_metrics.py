import numpy as np
import pytest

from fem_inhouse.partitioning import PartitionLayout
from fem_inhouse.postprocessing import (
    field_error_metrics,
    interface_gradient_ratio,
    localization_overlap_metrics,
)


def test_field_metrics_use_prediction_minus_reference() -> None:
    reference = np.array([[1.0, 2.0], [3.0, 4.0]])
    prediction = np.array([[2.0, 2.0], [1.0, 5.0]])

    metrics = field_error_metrics(reference, prediction)

    assert metrics.count == 4
    assert metrics.rmse == pytest.approx(np.sqrt(1.5))
    assert metrics.mae == 1.0
    assert metrics.signed_mean_error == 0.0
    assert metrics.maximum_absolute_error == 2.0
    assert metrics.relative_l2_error == pytest.approx(np.sqrt(6.0) / np.sqrt(30.0))
    assert -1 <= metrics.pearson_correlation <= 1


def test_field_metrics_apply_mask_and_finite_intersection() -> None:
    reference = np.array([1.0, np.nan, 3.0, 4.0])
    prediction = np.array([1.0, 2.0, np.inf, 4.0])
    metrics = field_error_metrics(reference, prediction, mask=[True, True, True, False])
    assert metrics.count == 1
    assert metrics.rmse == 0.0
    assert metrics.relative_l2_error == 0.0
    assert metrics.pearson_correlation == 1.0


def test_field_metric_contract_failures_and_zero_reference() -> None:
    with pytest.raises(ValueError, match="same shape"):
        field_error_metrics(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="mask"):
        field_error_metrics(np.zeros(2), np.zeros(2), mask=np.ones(3))
    with pytest.raises(ValueError, match="no valid"):
        field_error_metrics([np.nan], [np.nan])

    nonzero_prediction = field_error_metrics(np.zeros(2), np.ones(2))
    assert np.isinf(nonzero_prediction.relative_l2_error)
    assert np.isnan(nonzero_prediction.pearson_correlation)


def test_localization_overlap_quantifies_partial_hotspot_agreement() -> None:
    reference = np.array([4.0, 3.0, 2.0, 1.0])
    prediction = np.array([4.0, 1.0, 3.0, 2.0])

    overlap = localization_overlap_metrics(
        reference,
        prediction,
        top_fraction=0.5,
    )

    assert overlap.reference_count == 2
    assert overlap.prediction_count == 2
    assert overlap.intersection_count == 1
    assert overlap.intersection_over_union == pytest.approx(1.0 / 3.0)
    assert overlap.dice_coefficient == 0.5
    assert overlap.reference_recall == 0.5
    assert overlap.prediction_precision == 0.5


def test_localization_overlap_contracts_masks_and_ties() -> None:
    tied = np.ones((2, 2))
    identical = localization_overlap_metrics(tied, tied, top_fraction=0.25)
    assert identical.reference_count == 4
    assert identical.intersection_over_union == 1.0

    masked = localization_overlap_metrics(
        [4.0, 3.0, np.nan],
        [4.0, 2.0, 1.0],
        top_fraction=0.5,
        mask=[True, False, True],
    )
    assert masked.reference_count == 1
    assert masked.intersection_over_union == 1.0

    with pytest.raises(ValueError, match="top_fraction"):
        localization_overlap_metrics([1.0], [1.0], top_fraction=0.0)
    with pytest.raises(ValueError, match="same shape"):
        localization_overlap_metrics(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="mask"):
        localization_overlap_metrics(np.zeros(2), np.zeros(2), mask=np.ones(3))
    with pytest.raises(ValueError, match="no valid"):
        localization_overlap_metrics([np.nan], [np.nan])


def test_interface_ratio_is_one_for_affine_field_and_detects_seam() -> None:
    layout = PartitionLayout((8, 6), (2, 3))
    x = np.arange(8)[:, None]
    y = np.arange(6)[None, :]
    affine = 2.0 * x + 3.0 * y
    assert interface_gradient_ratio(affine, layout) == pytest.approx(1.0)

    with_seam = affine.copy()
    with_seam[4:, :] += 20.0
    assert interface_gradient_ratio(with_seam, layout) > 1.0


def test_interface_ratio_contract_and_constant_field() -> None:
    layout = PartitionLayout((4, 4), (2, 1))
    assert interface_gradient_ratio(np.ones((4, 4)), layout) == 1.0
    with pytest.raises(ValueError, match="expected element shape"):
        interface_gradient_ratio(np.ones((3, 4)), layout)
    invalid = np.ones((4, 4))
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        interface_gradient_ratio(invalid, layout)
    with pytest.raises(ValueError, match="no internal"):
        interface_gradient_ratio(
            np.ones((4, 4)),
            PartitionLayout((4, 4), (1, 1)),
        )
