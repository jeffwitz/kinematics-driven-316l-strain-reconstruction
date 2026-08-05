"""Tests for amplitude-independent paired crystal-slip metrics."""

from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.validation.crystal_slip_metrics import (
    SlipMetricConfig,
    compare_slip_fields,
    validate_slip_system_order,
)


def _fields() -> tuple[np.ndarray, np.ndarray]:
    meric = np.zeros((12, 4, 4))
    srix = np.zeros((12, 4, 4))
    meric[0, 1:3, 1:3] = 2.0
    srix[0, 1:3, 1:3] = 1.0
    return meric, srix


def test_identical_fields_have_zero_distance_and_unit_overlap() -> None:
    meric, _ = _fields()
    result = compare_slip_fields(meric, meric)
    assert result["system_distribution"]["s95_jaccard"] == 1.0
    assert result["system_distribution"]["total_variation_distance"] == 0.0
    assert result["system_distribution"]["cosine_similarity"] == pytest.approx(1.0)
    assert result["systems"][0]["spatial"]["normalized"]["l1"] == 0.0
    assert result["systems"][0]["spatial"]["normalized"]["cosine"] == pytest.approx(1.0)


def test_disjoint_systems_have_zero_jaccard_and_unit_variation_distance() -> None:
    meric, srix = _fields()
    srix[0] = 0.0
    srix[1, 1:3, 1:3] = 1.0
    result = compare_slip_fields(meric, srix)
    assert result["system_distribution"]["s95_jaccard"] == 0.0
    assert result["system_distribution"]["total_variation_distance"] == pytest.approx(1.0)


def test_common_shape_with_amplitude_scaling_has_zero_normalized_distance() -> None:
    meric, srix = _fields()
    srix *= 7.0
    result = compare_slip_fields(meric, srix)
    spatial = result["systems"][0]["spatial"]
    assert spatial["absolute"]["integral_ratio_meric_over_srix"] == pytest.approx(2.0 / 7.0)
    assert spatial["normalized"]["l1"] == pytest.approx(0.0)
    assert spatial["normalized"]["cosine"] == pytest.approx(1.0)


def test_zero_fields_are_undefined_not_division_by_zero() -> None:
    result = compare_slip_fields(np.zeros((12, 4, 4)), np.zeros((12, 4, 4)))
    assert result["global_amplitude"]["ratios_meric_over_srix"]["maximum"] is None
    assert result["systems"][0]["amplitude_ratio_meric_over_srix"] is None
    assert result["systems"][0]["spatial"]["status"] == "not_significant"


def test_system_order_mismatch_is_rejected() -> None:
    expected = ("01", "02", "03")
    with pytest.raises(ValueError, match="slip-system order"):
        validate_slip_system_order(("02", "01", "03"), expected)


def test_fraction_vector_is_normalized() -> None:
    meric, srix = _fields()
    result = compare_slip_fields(meric, srix, config=SlipMetricConfig())
    assert sum(item["meric_fraction"] for item in result["systems"]) == pytest.approx(1.0)
    assert sum(item["srix_fraction"] for item in result["systems"]) == pytest.approx(1.0)
