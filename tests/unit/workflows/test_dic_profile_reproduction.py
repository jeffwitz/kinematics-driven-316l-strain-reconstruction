from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.workflows.dic_profile_reproduction import (
    DISCRIMINATION_FACTOR,
    P43_SOLVE_BOUNDS,
    reproduction_metrics,
)


def test_reproduction_metrics_vanish_on_an_identical_field() -> None:
    field = np.random.default_rng(1).normal(size=(9, 7, 2))

    metrics = reproduction_metrics(field, field)

    assert metrics.component_rms_mm == 0.0
    assert metrics.maximum_absolute_mm == 0.0
    assert metrics.relative_vector_norm == 0.0


def test_reproduction_metrics_scale_with_a_known_offset() -> None:
    prepared = np.full((4, 5, 2), 3.0)
    recomputed = prepared + 0.6

    metrics = reproduction_metrics(recomputed, prepared)

    assert metrics.component_rms_mm == pytest.approx(0.6)
    assert metrics.maximum_absolute_mm == pytest.approx(0.6)
    assert metrics.relative_vector_norm == pytest.approx(0.2)


def test_reproduction_metrics_reject_mismatched_supports() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        reproduction_metrics(np.zeros((4, 5, 2)), np.zeros((4, 6, 2)))


def test_reproduction_metrics_reject_a_zero_reference() -> None:
    with pytest.raises(ValueError, match="identically zero"):
        reproduction_metrics(np.ones((3, 3, 2)), np.zeros((3, 3, 2)))


def test_reproduction_metrics_reject_nonfinite_values() -> None:
    prepared = np.ones((3, 3, 2))
    recomputed = prepared.copy()
    recomputed[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        reproduction_metrics(recomputed, prepared)


def test_p43_window_matches_the_archived_partition_shape() -> None:
    x0, x1, y0, y1 = P43_SOLVE_BOUNDS
    window = (slice(x0, x1 + 1), slice(y0, y1 + 1))
    support = np.zeros((3600, 3100, 2))[window]

    # The archived P43 nodal support is 661 by 611.
    assert support.shape[:2] == (661, 611)


def test_discrimination_factor_is_declared_above_unity() -> None:
    # A factor at or below one would make any ordering count as a decision.
    assert DISCRIMINATION_FACTOR > 1.0
