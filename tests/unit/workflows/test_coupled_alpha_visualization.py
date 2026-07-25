from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.workflows.coupled_alpha_visualization import (
    common_color_limits,
    symmetric_color_limit,
)


def test_common_color_limits_use_one_scale_for_all_fields() -> None:
    first = np.array([[0.0, 1.0], [2.0, 3.0]])
    second = np.array([[0.5, 4.0], [1.0, 2.0]])

    limits = common_color_limits((first, second))

    assert limits[:4] == (0.0, 4.0, 0.0, 4.0)
    assert limits[4] is None


def test_common_color_limits_record_robust_percentile() -> None:
    field = np.arange(100, dtype=float).reshape(10, 10)

    limits = common_color_limits((field,), percentile=95.0)

    assert limits[0] == 0.0
    assert limits[1] == 99.0
    assert limits[2] == 0.0
    assert limits[3] == pytest.approx(94.05)
    assert limits[4] == 95.0


def test_symmetric_color_limit_is_centered_on_zero() -> None:
    first = np.array([[-2.0, 1.0]])
    second = np.array([[3.0, -0.5]])

    limits = symmetric_color_limit((first, second))

    assert limits == (-3.0, 3.0, 3.0, None)


def test_symmetric_color_limit_supports_percentile() -> None:
    field = np.arange(-10.0, 11.0).reshape(3, 7)

    limits = symmetric_color_limit((field,), percentile=50.0)

    assert limits[0] == pytest.approx(-5.0)
    assert limits[1] == pytest.approx(5.0)
    assert limits[2] == 10.0
    assert limits[3] == 50.0


@pytest.mark.parametrize("function", (common_color_limits, symmetric_color_limit))
def test_limits_reject_empty_input(function) -> None:
    with pytest.raises(ValueError, match="at least one"):
        function(())


@pytest.mark.parametrize("function", (common_color_limits, symmetric_color_limit))
def test_limits_reject_non_two_dimensional_fields(function) -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        function((np.zeros(3),))


@pytest.mark.parametrize("function", (common_color_limits, symmetric_color_limit))
def test_limits_reject_non_finite_fields(function) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        function((np.array([[np.nan]]),))


def test_common_color_limits_reject_invalid_percentile() -> None:
    with pytest.raises(ValueError, match="percentile"):
        common_color_limits((np.ones((2, 2)),), percentile=0.0)
