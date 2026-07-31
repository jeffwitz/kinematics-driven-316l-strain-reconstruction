from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.validation.fractions_skill_score import (
    DEFAULT_SCALES_PIXELS,
    active_fraction_field,
    fractions_skill_score,
    minimum_skilful_scale,
    skill_curve,
    skill_table,
)

SHAPE = (80, 120)


def _two_bands(shape=SHAPE) -> np.ndarray:
    rows = np.arange(shape[0], dtype=np.float64)[:, None]
    ones = np.ones((1, shape[1]))
    return (
        np.exp(-0.5 * ((rows - 25.0) / 4.0) ** 2) * ones
        + np.exp(-0.5 * ((rows - 55.0) / 4.0) ** 2) * ones
    )


def _active(field: np.ndarray, quantile: float = 0.9) -> np.ndarray:
    return field >= np.quantile(field, quantile)


def test_identity_scores_one_at_every_scale() -> None:
    active = _active(_two_bands())

    for scale in DEFAULT_SCALES_PIXELS:
        assert fractions_skill_score(active, active, scale_pixels=scale) == pytest.approx(1.0)


def test_two_empty_fields_are_undefined_rather_than_perfect() -> None:
    empty = np.zeros(SHAPE, dtype=bool)

    # Reporting 1.0 here would flatter a candidate that predicts nothing.
    assert np.isnan(fractions_skill_score(empty, empty, scale_pixels=8))


def test_a_missing_band_scores_worse_than_a_present_one() -> None:
    reference = _active(_two_bands())
    partial = reference.copy()
    partial[:40] = False  # first band deleted

    at_16 = fractions_skill_score(reference, partial, scale_pixels=16)

    assert at_16 < 0.7


def test_a_spurious_band_is_penalised() -> None:
    reference = _active(_two_bands())
    polluted = reference.copy()
    polluted[70:74, :] = True

    assert fractions_skill_score(reference, polluted, scale_pixels=8) < 1.0


def test_amplitude_change_without_mask_change_does_not_move_the_score() -> None:
    field = _two_bands()
    threshold = float(np.quantile(field, 0.9))
    reference = field >= threshold
    # Scaling the field and the threshold together leaves the active set alone.
    scaled_active = (field * 3.0) >= (threshold * 3.0)

    np.testing.assert_array_equal(reference, scaled_active)
    assert fractions_skill_score(reference, scaled_active, scale_pixels=8) == pytest.approx(1.0)


def test_a_shift_recovers_skill_as_the_window_grows() -> None:
    reference = _active(_two_bands())
    shifted = np.roll(reference, 6, axis=0)

    scores = [
        fractions_skill_score(reference, shifted, scale_pixels=s)
        for s in (1, 4, 16, 48, 96)
    ]

    # The whole point of the score: a displaced band becomes compatible once the
    # neighbourhood exceeds the displacement.
    assert scores[0] < scores[-1]
    assert scores[-1] > 0.9


def test_a_larger_shift_needs_a_larger_window() -> None:
    field = _two_bands()
    threshold = float(np.quantile(field, 0.9))

    near = skill_curve(
        field, np.roll(field, 3, axis=0),
        threshold_value=threshold, threshold_quantile=0.9,
    )
    far = skill_curve(
        field, np.roll(field, 20, axis=0),
        threshold_value=threshold, threshold_quantile=0.9,
    )

    near_scale = minimum_skilful_scale(near, level=0.7)
    far_scale = minimum_skilful_scale(far, level=0.7)

    assert np.isfinite(near_scale)
    assert not np.isfinite(far_scale) or far_scale > near_scale


def test_widening_and_contraction_both_reduce_skill() -> None:
    field = _two_bands()
    threshold = float(np.quantile(field, 0.9))
    reference = field >= threshold
    wide = field >= (threshold * 0.6)
    narrow = field >= (threshold * 1.3)

    assert fractions_skill_score(reference, wide, scale_pixels=8) < 1.0
    assert fractions_skill_score(reference, narrow, scale_pixels=8) < 1.0


def test_the_fraction_field_is_normalised_by_valid_pixels_only() -> None:
    active = np.ones((20, 20), dtype=bool)
    valid = np.zeros((20, 20), dtype=bool)
    valid[5:15, 5:15] = True

    fraction = active_fraction_field(active, scale_pixels=5, valid_mask=valid)

    # Every valid pixel is active, so the fraction is one wherever it is defined,
    # including near the edge of the valid region.
    assert fraction[valid] == pytest.approx(1.0)


def test_minimum_skilful_scale_reports_nan_when_never_reached() -> None:
    field = _two_bands()
    curve = skill_curve(
        field,
        np.zeros_like(field),
        threshold_value=float(np.quantile(field, 0.9)),
        threshold_quantile=0.9,
    )

    assert np.isnan(minimum_skilful_scale(curve, level=0.5))


def test_skill_table_reports_every_threshold_separately() -> None:
    field = _two_bands()
    thresholds = {q: float(np.quantile(field, q)) for q in (0.80, 0.90, 0.95)}

    table = skill_table(field, field, thresholds=thresholds)

    assert [c["threshold_quantile"] for c in table["curves"]] == [0.80, 0.90, 0.95]
    for curve in table["curves"]:
        assert curve["values"] == pytest.approx([1.0] * len(DEFAULT_SCALES_PIXELS))
        assert curve["minimum_skilful_scale"]["0.9"] == 1.0


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        fractions_skill_score(
            np.zeros((4, 4), dtype=bool), np.zeros((4, 5), dtype=bool), scale_pixels=2
        )
