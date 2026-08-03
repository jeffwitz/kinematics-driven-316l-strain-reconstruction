"""The ROI qualification gate, which decides whether a matrix campaign is worth
running at all.

This module had no test. It is the filter the P43 review lacked, it rejects or
authorises ten hours of computation, and its verdict is quoted in
`validation/roi_qualification_results.md`. These tests exist to say what each of
its seven checks actually measures, on fields small enough to reason about.

Every field here is a synthetic Gaussian ridge. The DIC and the local field are
compared through the same section operator a real campaign would use, so the
numbers below are produced by the shipped code path, not by a stub.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.workflows.compare_observed_evm_candidates import MTF50_PIXELS
from fem_inhouse.workflows.qualify_roi import (
    MINIMUM_CONSISTENT_SIGN_FRACTION,
    MINIMUM_DIC_WIDTH_MTF50,
    MINIMUM_VALID_SECTION_FRACTION,
    MINIMUM_WIDTH_DEFICIT,
    _ratio_interval,
    format_report,
    qualify_roi,
)

SHAPE = (160, 160)
CENTRE = 80.0


def _ridge(sigma: float, *, amplitude: float = 1.0, background: float = 0.02) -> np.ndarray:
    """A straight band along x, Gaussian across y, on a nonzero background."""

    across = np.arange(SHAPE[1], dtype=float)
    profile = amplitude * np.exp(-0.5 * ((across - CENTRE) / sigma) ** 2) + background
    return np.tile(profile, (SHAPE[0], 1))


@pytest.fixture(scope="module")
def narrower_local() -> dict:
    """The configuration the gate is looking for: local bands too narrow."""

    return qualify_roi(dic_evm=_ridge(9.0), local_evm=_ridge(5.0), partition_id=7)


@pytest.fixture(scope="module")
def wider_local() -> dict:
    """The configuration P43 actually showed: local bands as wide or wider."""

    return qualify_roi(dic_evm=_ridge(6.0), local_evm=_ridge(9.0), partition_id=43)


def test_the_two_fields_must_share_a_support() -> None:
    with pytest.raises(ValueError, match="same support"):
        qualify_roi(
            dic_evm=np.zeros((10, 10)),
            local_evm=np.zeros((10, 12)),
            partition_id=0,
        )


def test_a_constant_field_is_refused_rather_than_thresholded() -> None:
    """No Otsu threshold exists, and inventing one would fabricate a band."""

    flat = np.full(SHAPE, 0.5)

    with pytest.raises(ValueError, match="constant field has no Otsu threshold"):
        qualify_roi(dic_evm=flat, local_evm=flat, partition_id=0)


def test_a_narrower_local_band_satisfies_the_directional_conditions(
    narrower_local: dict,
) -> None:
    """The three checks that carry the physical argument of the gate."""

    band = narrower_local["bands"]["band1"]

    assert band["width_ratio"]["median"] > 1.0 / (1.0 - MINIMUM_WIDTH_DEFICIT)
    assert band["narrower_fraction"] >= MINIMUM_CONSISTENT_SIGN_FRACTION
    assert narrower_local["checks"]["local_is_measurably_narrower"]["passed"] is True
    assert narrower_local["checks"]["ratio_interval_excludes_one"]["passed"] is True
    assert narrower_local["checks"]["deficit_sign_is_consistent"]["passed"] is True


def test_a_wider_local_band_fails_all_three(wider_local: dict) -> None:
    """The observed situation. The mechanism has no room to act."""

    band = wider_local["bands"]["band1"]

    assert band["width_ratio"]["median"] < 1.0
    assert band["narrower_fraction"] == 0.0
    for name in (
        "local_is_measurably_narrower",
        "ratio_interval_excludes_one",
        "deficit_sign_is_consistent",
    ):
        assert wider_local["checks"][name]["passed"] is False
        assert name in wider_local["failed"]
    assert wider_local["qualified"] is False


def test_the_resolution_check_cannot_be_satisfied_by_any_field(
    narrower_local: dict,
) -> None:
    """A structural property of the gate, not a property of the data.

    `dic_band_is_resolved` demands a DIC width of at least
    `1.5 * 49 = 73.5 px`. The integral-width estimator saturates near `35 px`,
    a ceiling set by the `40 px` section half-length and the background taken
    beyond `+/- 12 px` -- measured and reported in
    `validation/roi_qualification_results.md`. The bound therefore sits above
    the largest value the estimator can return, so this particular check fails
    for every field, including one built to be as wide as the window allows.

    That does not weaken the archived conclusion, which rests on the
    directional checks above: the widest ratio any real ROI reached was `1.06`
    against a bound of `1.33`. It does mean this one condition carries no
    information, and a reader should not count it among the reasons a ROI was
    rejected.
    """

    estimator_ceiling_pixels = 35.0
    required = MINIMUM_DIC_WIDTH_MTF50 * MTF50_PIXELS

    assert required > estimator_ceiling_pixels
    for sigma in (9.0, 20.0, 40.0):
        wide = qualify_roi(dic_evm=_ridge(sigma), local_evm=_ridge(sigma / 2), partition_id=0)
        width = wide["checks"]["dic_band_is_resolved"]["narrowest_dic_width_px"]
        assert width < estimator_ceiling_pixels
        assert wide["checks"]["dic_band_is_resolved"]["passed"] is False
    assert narrower_local["qualified"] is False


def test_every_section_of_a_straight_band_is_usable(narrower_local: dict) -> None:
    band = narrower_local["bands"]["band1"]

    assert band["sections"] > 0
    assert band["valid_fraction"] == 1.0
    assert (
        narrower_local["checks"]["enough_usable_sections"]["worst_band_valid_fraction"]
        >= MINIMUM_VALID_SECTION_FRACTION
    )


def test_sections_that_leave_the_roi_are_dropped_and_reported() -> None:
    """A band running into a corner: half its sections cannot be sampled.

    This is the ordinary campaign case, not a pathology -- a band that touches
    the ROI edge has normals reaching outside it, and a profile sampled partly
    off the field would report a width that is an artefact of the crop. Those
    sections are excluded, and `enough_usable_sections` is what makes the loss
    visible instead of silently averaging over whatever survived.
    """

    size = 200
    across = np.arange(size, dtype=float)[None, :]
    along = np.arange(size, dtype=float)[:, None]
    distance = (across - (along - 60.0)) / np.sqrt(2.0)
    dic = np.exp(-0.5 * (distance / 6.0) ** 2) + 0.02
    local = np.exp(-0.5 * (distance / 4.0) ** 2) + 0.02

    result = qualify_roi(dic_evm=dic, local_evm=local, partition_id=3)
    band = result["bands"]["band1"]

    assert 0.0 < band["valid_fraction"] < MINIMUM_VALID_SECTION_FRACTION
    assert result["checks"]["enough_usable_sections"]["passed"] is False
    assert "enough_usable_sections" in result["failed"]
    # The surviving sections still produce a usable width, so the failure is
    # about coverage of the band, not about the measurement being broken.
    assert np.isfinite(band["dic_width_median"])


def test_two_concentric_bands_have_a_negligible_centreline_offset(
    narrower_local: dict,
) -> None:
    """Both ridges sit on the same line, so the offset check must pass."""

    band = narrower_local["bands"]["band1"]

    assert band["centreline_offset_median"] < 0.1 * band["dic_width_median"]
    assert narrower_local["checks"]["centreline_is_close"]["passed"] is True


def test_the_local_field_is_thresholded_with_the_dic_threshold(
    narrower_local: dict,
) -> None:
    """A design choice worth stating: one threshold, taken from the measurement.

    Thresholding each field on its own Otsu value would compare two differently
    defined objects and make the object-count check meaningless.
    """

    assert narrower_local["checks"]["same_object_count"]["dic"] == 1
    assert narrower_local["checks"]["same_object_count"]["local"] == 1
    assert narrower_local["checks"]["same_object_count"]["passed"] is True
    assert narrower_local["otsu_threshold"] > 0.0


def test_the_verdict_is_exactly_the_set_of_failed_checks(
    narrower_local: dict, wider_local: dict
) -> None:
    for result in (narrower_local, wider_local):
        expected = sorted(
            name for name, check in result["checks"].items() if not check["passed"]
        )
        assert result["failed"] == expected
        assert result["qualified"] is (not expected)


class TestRatioInterval:
    """The bootstrap behind `width_ratio`, which the verdict reads directly."""

    def test_too_few_paired_sections_report_nothing_rather_than_a_number(self) -> None:
        """Below eight pairs the interval is not reported. A median of three
        sections would look like a measurement and would not be one."""

        result = _ratio_interval(
            np.arange(1.0, 6.0), np.arange(1.0, 6.0), np.ones(5, dtype=bool)
        )

        assert result["count"] == 5
        assert np.isnan(result["median"])
        assert np.isnan(result["q05"])
        assert np.isnan(result["q95"])

    def test_unusable_and_non_finite_sections_are_excluded_from_the_count(self) -> None:
        dic = np.array([2.0, 2.0, np.nan, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        local = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        usable = np.ones(10, dtype=bool)
        usable[0] = False

        result = _ratio_interval(dic, local, usable)

        assert result["count"] == 8
        assert result["median"] == pytest.approx(2.0)

    def test_a_zero_local_width_is_dropped_rather_than_dividing_by_zero(self) -> None:
        dic = np.full(10, 2.0)
        local = np.full(10, 1.0)
        local[:2] = 0.0

        assert _ratio_interval(dic, local, np.ones(10, dtype=bool))["count"] == 8

    def test_the_interval_brackets_the_median_and_is_deterministic(self) -> None:
        generator = np.random.default_rng(0)
        local = np.full(40, 1.0)
        dic = 1.5 + 0.1 * generator.standard_normal(40)

        first = _ratio_interval(dic, local, np.ones(40, dtype=bool))
        second = _ratio_interval(dic, local, np.ones(40, dtype=bool))

        assert first == second, "the bootstrap seed is registered, so it must repeat"
        assert first["q05"] <= first["median"] <= first["q95"]


class TestReport:
    def test_a_rejected_partition_names_itself_and_its_failures(
        self, wider_local: dict
    ) -> None:
        report = format_report(wider_local)

        assert report.startswith("partition 043: REJECTED")
        assert "fails: " in report
        for name in wider_local["failed"]:
            assert name in report

    def test_every_check_appears_with_its_evidence(self, wider_local: dict) -> None:
        """A verdict line without its numbers cannot be audited."""

        report = format_report(wider_local)

        for name in wider_local["checks"]:
            assert name in report
        assert "narrowest_dic_width_px" in report
        assert report.count("pass ") + report.count("FAIL ") == len(wider_local["checks"])

    def test_a_qualified_partition_says_so_without_a_failure_line(self) -> None:
        report = format_report(
            {
                "partition_id": 12,
                "qualified": True,
                "failed": [],
                "checks": {"only_check": {"passed": True, "value": 1.0}},
            }
        )

        assert report.startswith("partition 012: QUALIFIED")
        assert "fails:" not in report
