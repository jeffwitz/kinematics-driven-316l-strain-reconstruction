from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.validation.band_profiles import (
    BackgroundEstimate,
    NormalProfile,
    WidthStatus,
    compare_profiles,
    continuity_metrics,
    estimate_background,
    excess_profile,
    measure_amplitude,
    measure_position,
    measure_width,
    sample_normal_profile,
    summarise,
)

SIGMA = 4.0
#: A Gaussian of standard deviation s has these widths in closed form.
FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0)) * SIGMA
INTEGRAL = np.sqrt(2.0 * np.pi) * SIGMA
SECOND_MOMENT = 2.0 * SIGMA


def _gaussian_profile(*, amplitude=1.0, background=0.0, centre=0.0, half=40.0):
    offsets = np.arange(-half, half + 1.0)
    values = background + amplitude * np.exp(-0.5 * ((offsets - centre) / SIGMA) ** 2)
    return NormalProfile(0, offsets, values, True, "")


def test_gaussian_widths_match_their_closed_forms() -> None:
    profile = _gaussian_profile()
    background = estimate_background(profile, corridor_half_width_pixels=20.0)

    width = measure_width(profile, background)

    assert width.status is WidthStatus.OK
    assert width.fwhm_pixels == pytest.approx(FWHM, rel=0.02)
    assert width.integral_pixels == pytest.approx(INTEGRAL, rel=0.02)
    assert width.second_moment_pixels == pytest.approx(SECOND_MOMENT, rel=0.05)


def test_the_three_width_definitions_disagree_and_are_all_reported() -> None:
    width = measure_width(
        _gaussian_profile(), estimate_background(_gaussian_profile(),
                                                 corridor_half_width_pixels=20.0)
    )

    values = [width.fwhm_pixels, width.integral_pixels, width.second_moment_pixels]
    assert all(np.isfinite(values))
    # None is a substitute for another, even on a clean Gaussian.
    assert len({round(v, 3) for v in values}) == 3


def test_background_is_estimated_from_the_tails_not_assumed_zero() -> None:
    profile = _gaussian_profile(background=0.5)

    background = estimate_background(profile, corridor_half_width_pixels=20.0)

    assert background.level == pytest.approx(0.5, abs=1e-3)
    assert float(np.max(excess_profile(profile, background))) == pytest.approx(1.0, abs=1e-3)


def test_a_nonzero_background_would_inflate_the_width_if_ignored() -> None:
    raised = _gaussian_profile(background=0.5)
    honest = estimate_background(raised, corridor_half_width_pixels=20.0)
    ignored = BackgroundEstimate(level=0.0, spread=0.0, sample_count=0)

    assert measure_width(raised, ignored).integral_pixels > measure_width(
        raised, honest
    ).integral_pixels * 2.0


def test_a_flat_profile_is_reported_as_too_weak() -> None:
    offsets = np.arange(-20.0, 21.0)
    profile = NormalProfile(0, offsets, np.full_like(offsets, 0.3), True, "")

    width = measure_width(profile, estimate_background(profile, corridor_half_width_pixels=8.0))

    assert width.status is WidthStatus.TOO_WEAK
    assert np.isnan(width.fwhm_pixels)


def test_a_monotone_ramp_has_its_peak_at_the_edge() -> None:
    offsets = np.arange(-20.0, 21.0)
    profile = NormalProfile(0, offsets, offsets * 0.01 + 1.0, True, "")

    width = measure_width(profile, BackgroundEstimate(0.0, 0.0, 0))

    assert width.status is WidthStatus.PEAK_AT_EDGE


def test_a_band_wider_than_the_profile_has_no_half_maximum_crossing() -> None:
    offsets = np.arange(-6.0, 7.0)
    values = np.exp(-0.5 * (offsets / 40.0) ** 2)  # far wider than the window
    profile = NormalProfile(0, offsets, values, True, "")

    width = measure_width(profile, BackgroundEstimate(0.0, 0.0, 0))

    assert width.status is WidthStatus.NO_CROSSING
    assert np.isnan(width.fwhm_pixels)
    # The integral definitions still return something usable.
    assert np.isfinite(width.integral_pixels)


def test_a_two_peaked_profile_is_flagged_multimodal() -> None:
    offsets = np.arange(-40.0, 41.0)
    values = np.exp(-0.5 * ((offsets + 15) / 3.0) ** 2) + np.exp(
        -0.5 * ((offsets - 15) / 3.0) ** 2
    )
    profile = NormalProfile(0, offsets, values, True, "")

    width = measure_width(profile, BackgroundEstimate(0.0, 0.0, 0))

    assert width.status is WidthStatus.MULTIMODAL
    assert width.peak_count == 2


def test_an_invalid_profile_yields_an_empty_measurement() -> None:
    profile = NormalProfile(0, np.arange(-3.0, 4.0), np.zeros(0), False, "leaves_support")

    assert measure_width(profile, BackgroundEstimate(0.0, 0.0, 0)).status is WidthStatus.EMPTY


def test_position_recovers_a_known_offset() -> None:
    profile = _gaussian_profile(centre=6.0)
    background = estimate_background(profile, corridor_half_width_pixels=20.0)

    position = measure_position(profile, background)

    assert position["peak_offset"] == pytest.approx(6.0, abs=1.0)
    assert position["centroid_offset"] == pytest.approx(6.0, abs=0.5)
    assert position["detected"] == 1.0


def test_amplitude_scales_with_the_imposed_factor() -> None:
    single = _gaussian_profile(amplitude=1.0)
    double = _gaussian_profile(amplitude=2.0)
    background = BackgroundEstimate(0.0, 0.0, 0)

    a = measure_amplitude(single, background, corridor_half_width_pixels=10.0)
    b = measure_amplitude(double, background, corridor_half_width_pixels=10.0)

    assert b["peak"] == pytest.approx(2.0 * a["peak"], rel=1e-9)
    assert b["mass"] == pytest.approx(2.0 * a["mass"], rel=1e-9)


def test_identical_profiles_compare_as_identical() -> None:
    profile = _gaussian_profile()
    background = BackgroundEstimate(0.0, 0.0, 0)

    result = compare_profiles(profile, profile, background, background)

    assert result["correlation"] == pytest.approx(1.0)
    assert result["l1"] == pytest.approx(0.0)
    assert result["l2"] == pytest.approx(0.0)


def test_a_shifted_profile_is_penalised_in_l2() -> None:
    background = BackgroundEstimate(0.0, 0.0, 0)
    reference = _gaussian_profile()

    near = compare_profiles(reference, _gaussian_profile(centre=2.0), background, background)
    far = compare_profiles(reference, _gaussian_profile(centre=10.0), background, background)

    assert far["l2"] > near["l2"]
    assert far["correlation"] < near["correlation"]


def test_sampling_follows_the_normal_direction() -> None:
    field = np.zeros((60, 60))
    field[30, :] = 1.0  # a one-pixel horizontal ridge

    profile = sample_normal_profile(
        field, origin=(30.0, 30.0), normal=(1.0, 0.0), half_length_pixels=10.0
    )

    assert profile.valid
    assert profile.values[int(np.argmax(profile.values))] == pytest.approx(1.0)
    assert profile.offsets_pixels[int(np.argmax(profile.values))] == pytest.approx(0.0)


def test_a_section_leaving_the_support_is_excluded_with_a_reason() -> None:
    profile = sample_normal_profile(
        np.zeros((20, 20)), origin=(2.0, 2.0), normal=(1.0, 0.0), half_length_pixels=10.0
    )

    assert not profile.valid
    assert profile.exclusion_reason == "leaves_support"


def test_a_section_crossing_an_invalid_pixel_is_excluded_with_a_reason() -> None:
    mask = np.ones((60, 60), dtype=bool)
    mask[34, 30] = False

    profile = sample_normal_profile(
        np.zeros((60, 60)),
        origin=(30.0, 30.0),
        normal=(1.0, 0.0),
        half_length_pixels=10.0,
        valid_mask=mask,
    )

    assert not profile.valid
    assert profile.exclusion_reason == "crosses_invalid"


def test_continuity_measures_the_longest_gap_not_only_the_fraction() -> None:
    scattered = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 2, dtype=bool)
    blocked = np.array([1] * 8 + [0] * 8, dtype=bool)

    a = continuity_metrics(scattered, spacing_pixels=2.0)
    b = continuity_metrics(blocked, spacing_pixels=2.0)

    assert a["detected_fraction"] == pytest.approx(b["detected_fraction"])
    # Same fraction, very different failure: only the gap length separates them.
    assert b["longest_gap_pixels"] > a["longest_gap_pixels"]
    assert a["gap_count"] > b["gap_count"]


def test_a_fully_detected_band_has_no_gap() -> None:
    result = continuity_metrics(np.ones(10, dtype=bool), spacing_pixels=1.5)

    assert result["detected_fraction"] == 1.0
    assert result["longest_gap_pixels"] == 0.0
    assert result["gap_count"] == 0.0


def test_summary_exposes_the_worst_decile_not_only_the_mean() -> None:
    values = np.concatenate([np.ones(90), np.full(10, 100.0)])

    result = summarise(values)

    assert result["median"] == pytest.approx(1.0)
    assert result["worst_decile"] == pytest.approx(100.0)
    assert result["worst_decile"] > result["mean"] * 5


def test_summary_reports_the_missing_fraction() -> None:
    values = np.array([1.0, 2.0, np.nan, np.nan])

    result = summarise(values)

    assert result["valid_fraction"] == pytest.approx(0.5)
    assert result["missing_fraction"] == pytest.approx(0.5)
