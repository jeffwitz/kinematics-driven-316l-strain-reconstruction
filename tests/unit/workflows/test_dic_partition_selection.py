from __future__ import annotations

import json

import numpy as np
import pytest

from fem_inhouse.workflows.dic_partition_selection import (
    _band_morphology,
    _winsorized_kurtosis,
    scan_dic_partition_heterogeneity,
    write_dic_partition_heterogeneity_report,
)


def test_scan_ranks_dic_partitions_and_records_indicators(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    x, y = np.meshgrid(np.arange(8.0), np.arange(8.0), indexing="ij")
    displacement_x = 0.001 * x**2
    displacement_y = 0.001 * y**2
    displacement_x[2:4, 5:7] += 0.2
    np.save(inputs / "displacement_x_mm.npy", displacement_x)
    np.save(inputs / "displacement_y_mm.npy", displacement_y)

    report = scan_dic_partition_heterogeneity(
        input_directory=inputs,
        parts_x=2,
        parts_y=2,
        padding=0,
        spacing_x_mm=1.0,
        spacing_y_mm=1.0,
    )

    assert report["observable"].startswith("EVM_HISTORICAL")
    assert len(report["partitions"]) == 4
    assert report["selection_indicator"] == "dic_band_morphology_score_q85"
    assert "q95_minus_q50_over_iqr" in report["partitions"][0]
    assert "band_aspect_ratio" in report["partitions"][0]
    assert report["partitions"][0]["band_score"] >= report["partitions"][-1]["band_score"]


def test_write_dic_partition_heterogeneity_report(tmp_path) -> None:
    output = tmp_path / "selection.json"
    report = {"partitions": [{"partition_id": 3}], "selection_indicator": "test"}

    write_dic_partition_heterogeneity_report(report, output)

    assert json.loads(output.read_text()) == report


# --------------------------------------------------------------------------
# The band score itself.
#
# This project refuses manual ROI selection, so this scorer *is* the selection.
# Its docstring states a design claim -- that unlike kurtosis it rejects an
# intense but nearly circular hotspot -- and the whole scoring branch was
# unexecuted: the tests above only reach the path where no component qualifies.
# What follows checks the claim rather than restating it.
# --------------------------------------------------------------------------

SHAPE = (120, 120)
SPACING = 0.00184


def _field(mask_builder, *, amplitude: float = 1.0) -> np.ndarray:
    """A quiet background carrying one bright region."""

    field = np.full(SHAPE, 0.01)
    rows, columns = np.indices(SHAPE)
    field[mask_builder(rows, columns)] = amplitude
    return field


def _morphology(field: np.ndarray) -> dict:
    return _band_morphology(field, spacing_x_mm=SPACING, spacing_y_mm=SPACING)


def _stripe(half_width: int = 6):
    return lambda rows, columns: np.abs(columns - SHAPE[1] // 2) <= half_width


def _disc(radius: float):
    return lambda rows, columns: (
        (rows - SHAPE[0] / 2) ** 2 + (columns - SHAPE[1] / 2) ** 2
    ) <= radius**2


def test_a_uniform_field_scores_zero_through_contrast_not_through_geometry() -> None:
    """Worth being precise about, because the two routes to zero differ.

    A constant field still yields a component -- its q85 excursion set is the
    whole domain, with aspect ratio 1 -- so the geometry does not reject it.
    What zeroes the score is `max(contrast_sigma, 0)`: there is no excursion
    above the background because there is no background to exceed.
    """

    result = _morphology(np.full(SHAPE, 0.3))

    assert result["band_score"] == 0.0
    assert result["band_contrast_sigma"] <= 0.0
    assert result["band_aspect_ratio"] == pytest.approx(1.0)
    assert result["band_area_fraction"] > 0.0


def test_an_isolated_hot_pixel_is_smoothed_away_and_scores_zero() -> None:
    """A single outlier must not select a ROI, and it does not -- but the route
    is worth knowing.

    The three-pixel Gaussian erases the pixel, the q85 threshold then falls on
    the background itself, and the excursion set becomes most of the domain with
    a *negative* contrast. The score is clamped to zero by `max(sigma, 0)`, not
    by the area filter. In every field tried here the clamp is what produces a
    zero; the explicit all-zero fallback below it was not reachable, because the
    quantile threshold always leaves a component larger than the minimum area.
    """

    field = np.full((40, 40), 0.01)
    field[0, 0] = 1.0

    result = _morphology(field)

    assert result["band_score"] == 0.0
    assert result["band_contrast_sigma"] < 0.0
    assert result["band_area_fraction"] > 0.5


def test_an_elongated_band_is_detected_with_a_high_aspect_ratio() -> None:
    result = _morphology(_field(_stripe()))

    assert result["band_score"] > 0.0
    assert result["band_aspect_ratio"] > 3.0
    assert result["band_area_fraction"] > 0.0
    assert result["band_major_extent_px"] > result["band_average_width_px"]


def test_an_elongated_band_outscores_a_disc_of_the_same_area() -> None:
    """The documented design claim, at equal area and equal amplitude."""

    half_width = 6
    stripe_area = (2 * half_width + 1) * SHAPE[0]
    radius = float(np.sqrt(stripe_area / np.pi))

    band = _morphology(_field(_stripe(half_width)))
    blob = _morphology(_field(_disc(radius)))

    assert band["band_aspect_ratio"] > blob["band_aspect_ratio"]
    assert band["band_score"] > blob["band_score"]


def test_kurtosis_really_cannot_separate_them() -> None:
    """The premise of the previous test, verified rather than assumed.

    Both fields carry the same intensity histogram, so an intensity statistic
    is blind to the difference. That is precisely why the morphological score
    replaced kurtosis as the selection indicator.
    """

    half_width = 6
    stripe_area = (2 * half_width + 1) * SHAPE[0]
    radius = float(np.sqrt(stripe_area / np.pi))

    band = _winsorized_kurtosis(_field(_stripe(half_width)).ravel(), 1.0, 99.0)
    blob = _winsorized_kurtosis(_field(_disc(radius)).ravel(), 1.0, 99.0)

    assert band == pytest.approx(blob, rel=0.05)


def test_the_score_is_invariant_under_an_affine_rescaling_of_the_field() -> None:
    """A property of the indicator, verified because it is easy to expect the
    opposite.

    `contrast_sigma` divides the component-to-background mean gap by the
    standard deviation of the same field, and both scale together, so the score
    does not change when the whole field is multiplied or shifted. A ROI cannot
    rank higher merely by straining more: the indicator ranks *shape*. That is
    the intent -- it is the morphology score, not an intensity score -- but it
    also means it cannot be used to compare how strongly two ROIs localise.
    """

    field = _field(_stripe())
    reference = _morphology(field)
    rescaled = _morphology(7.5 * field + 0.3)

    assert reference["band_contrast_sigma"] > 0.0
    assert rescaled["band_score"] == pytest.approx(reference["band_score"], rel=1e-12)
    assert rescaled["band_contrast_sigma"] == pytest.approx(
        reference["band_contrast_sigma"], rel=1e-12
    )


def test_the_quantile_threshold_pins_the_detected_area_whatever_the_band_width() -> None:
    """A consequence of using a quantile, and a real limit on what is compared.

    The excursion set is the top 15 percent of the smoothed field by
    construction, so a stripe four times wider yields the *same* detected area
    fraction and the *same* aspect ratio. The indicator therefore compares the
    shape of each ROI's top-15-percent set, not the size of its band. Two ROIs
    whose bands differ only in width are, to this score, distinguished solely
    through contrast.
    """

    results = {half: _morphology(_field(_stripe(half))) for half in (3, 6, 12, 20)}
    areas = {round(r["band_area_fraction"], 6) for r in results.values()}
    aspects = {round(r["band_aspect_ratio"], 6) for r in results.values()}

    assert len(areas) == 1
    assert len(aspects) == 1
    assert areas.pop() == pytest.approx(0.153, abs=0.01)


def test_the_score_is_not_monotone_in_the_true_band_width() -> None:
    """Which matters when reading a ranking: a lower score does not mean a
    narrower band. The maximum sits at an intermediate width."""

    scores = [_morphology(_field(_stripe(half)))["band_score"] for half in (3, 6, 12, 20)]

    assert scores[0] < scores[2]
    assert scores[3] < scores[2]


def test_touching_the_border_is_counted() -> None:
    """A band running out of the ROI is partly unobserved, so it is discounted.

    The vertical stripe reaches the top and bottom edges: two contacts, which
    apply the `0.75` factor. A stripe kept clear of every edge reaches none.
    """

    def interior(rows, columns):
        return (np.abs(columns - SHAPE[1] // 2) <= 6) & (rows > 20) & (rows < SHAPE[0] - 21)

    assert _morphology(_field(_stripe()))["band_boundary_contacts"] == 2
    assert _morphology(_field(interior))["band_boundary_contacts"] == 0


def test_the_physical_extent_scales_with_the_pixel_spacing() -> None:
    """The millimetre figures are computed separately, on scaled coordinates."""

    field = _field(_stripe())

    single = _band_morphology(field, spacing_x_mm=SPACING, spacing_y_mm=SPACING)
    double = _band_morphology(field, spacing_x_mm=2 * SPACING, spacing_y_mm=2 * SPACING)

    assert single["band_major_extent_px"] == pytest.approx(double["band_major_extent_px"])
    assert double["band_major_extent_mm"] == pytest.approx(
        2.0 * single["band_major_extent_mm"], rel=1e-9
    )
    assert double["band_average_width_mm"] == pytest.approx(
        2.0 * single["band_average_width_mm"], rel=1e-9
    )


def test_the_best_component_is_reported_when_several_qualify() -> None:
    """Two disjoint components: the winner is the best one, not the first."""

    def two(rows, columns):
        long_stripe = (np.abs(columns - 30) <= 5) & (rows > 5) & (rows < SHAPE[0] - 6)
        short_stripe = (np.abs(columns - 90) <= 5) & (rows > 45) & (rows < 75)
        return long_stripe | short_stripe

    result = _morphology(_field(two))
    only_long = _morphology(
        _field(lambda r, c: (np.abs(c - 30) <= 5) & (r > 5) & (r < SHAPE[0] - 6))
    )

    assert result["band_aspect_ratio"] > 3.0
    assert result["band_score"] == pytest.approx(only_long["band_score"], rel=0.4)


def test_winsorising_removes_the_influence_of_a_single_outlier() -> None:
    """The point of it: one bad pixel must not select a ROI."""

    values = np.random.default_rng(2).standard_normal(2_000)
    contaminated = values.copy()
    contaminated[0] = 500.0

    assert _winsorized_kurtosis(contaminated, 0.0, 100.0) > 100.0
    assert _winsorized_kurtosis(contaminated, 1.0, 99.0) == pytest.approx(
        _winsorized_kurtosis(values, 1.0, 99.0), abs=0.2
    )
