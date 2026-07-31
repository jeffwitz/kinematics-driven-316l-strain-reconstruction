from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.validation.band_profiles import (
    BackgroundEstimate,
    NormalProfile,
    measure_position,
    measure_width,
)
from fem_inhouse.validation.falsification_cases import (
    add_spurious_band,
    change_band_width,
    interrupt_region,
    remove_region,
    scale_amplitude,
    standard_cases,
    translate_field,
)

SIGMA = 4.0


def _two_band_field(shape=(80, 120)) -> np.ndarray:
    rows = np.arange(shape[0], dtype=np.float64)[:, None]
    ones = np.ones((1, shape[1]))
    first = np.exp(-0.5 * ((rows - 25.0) / SIGMA) ** 2) * ones
    second = np.exp(-0.5 * ((rows - 55.0) / SIGMA) ** 2) * ones
    return first + second


def _band_region(shape=(80, 120), centre=25.0, half=12) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[int(centre) - half : int(centre) + half, :] = True
    return mask


def _column_profile(field: np.ndarray, column: int = 60) -> NormalProfile:
    offsets = np.arange(field.shape[0], dtype=np.float64) - 25.0
    return NormalProfile(0, offsets, field[:, column].copy(), True, "")


def test_translation_moves_the_peak_by_the_imposed_amount() -> None:
    field = _two_band_field()
    background = BackgroundEstimate(0.0, 0.0, 0)

    before = measure_position(_column_profile(field), background)
    after = measure_position(_column_profile(translate_field(field, rows=4.0, columns=0.0)),
                             background)

    assert after["peak_offset"] - before["peak_offset"] == pytest.approx(4.0, abs=0.5)


def test_amplitude_scaling_leaves_the_width_untouched() -> None:
    field = _two_band_field()
    background = BackgroundEstimate(0.0, 0.0, 0)

    reference = measure_width(_column_profile(field), background)
    scaled = measure_width(_column_profile(scale_amplitude(field, factor=1.5)), background)

    assert scaled.fwhm_pixels == pytest.approx(reference.fwhm_pixels, rel=1e-6)


def test_widening_increases_width_while_holding_the_peak() -> None:
    field = _two_band_field()
    background = BackgroundEstimate(0.0, 0.0, 0)
    widened = change_band_width(field, factor=1.4)

    reference = measure_width(_column_profile(field), background)
    after = measure_width(_column_profile(widened), background)

    assert after.integral_pixels > reference.integral_pixels
    assert float(np.max(widened)) == pytest.approx(float(np.max(field)), rel=0.05)


def test_contraction_reduces_width() -> None:
    field = _two_band_field()
    background = BackgroundEstimate(0.0, 0.0, 0)

    reference = measure_width(_column_profile(field), background)
    after = measure_width(_column_profile(change_band_width(field, factor=0.7)), background)

    assert after.integral_pixels < reference.integral_pixels


def test_removing_a_band_leaves_the_other_intact() -> None:
    field = _two_band_field()

    reduced = remove_region(field, region=_band_region())

    assert float(np.max(reduced[10:40])) < 0.05
    assert float(np.max(reduced[45:70])) == pytest.approx(1.0, abs=0.05)


def test_interrupting_a_band_blanks_part_of_its_length_only() -> None:
    field = _two_band_field()
    region = np.zeros_like(field, dtype=bool)
    region[13:38, :] = True

    broken = interrupt_region(field, region=region, fraction=0.4)

    # The cut runs along the longer extent, which here is the column direction.
    assert float(np.max(broken[13:38])) == pytest.approx(1.0, abs=0.05)
    assert float(np.min(np.max(broken[13:38], axis=0))) < 0.05


def test_a_spurious_band_adds_signal_where_there_was_none() -> None:
    field = _two_band_field()

    polluted = add_spurious_band(
        field, centre=(70.0, 60.0), orientation_degrees=0.0, amplitude=0.8
    )

    assert float(np.max(field[66:75])) < 0.05
    assert float(np.max(polluted[66:75])) > 0.5


def test_the_standard_set_covers_the_registered_defect_families() -> None:
    cases = standard_cases(_two_band_field(), band_region=_band_region())

    families = {case.defect for case in cases}
    assert families == {"position", "amplitude", "width", "missing_band", "continuity"}
    assert all(case.field.shape == (80, 120) for case in cases)
    assert len({case.name for case in cases}) == len(cases)


def test_position_error_grows_monotonically_with_the_imposed_shift() -> None:
    field = _two_band_field()
    background = BackgroundEstimate(0.0, 0.0, 0)
    reference = measure_position(_column_profile(field), background)["centroid_offset"]

    errors = []
    for shift in (1.0, 4.0, 16.0):
        moved = translate_field(field, rows=shift, columns=0.0)
        centroid = measure_position(_column_profile(moved), background)["centroid_offset"]
        errors.append(abs(centroid - reference))

    # A metric that failed this could not rank position defects at all.
    assert errors == sorted(errors)
    assert errors[-1] > errors[0]


def test_generators_reject_incompatible_regions() -> None:
    with pytest.raises(ValueError, match="match the field shape"):
        remove_region(np.zeros((10, 10)), region=np.zeros((5, 5), dtype=bool))
    with pytest.raises(ValueError, match="finite and positive"):
        scale_amplitude(np.zeros((4, 4)), factor=0.0)
