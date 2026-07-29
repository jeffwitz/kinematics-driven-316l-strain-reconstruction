from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.workflows.dic_multistep import anchor_displacement_history


def test_anchor_history_preserves_deviation_and_reaches_endpoint() -> None:
    raw = np.zeros((5, 3, 2, 2))
    fractions = np.linspace(0.0, 1.0, 5)
    raw[..., 0] = fractions[:, None, None] * 2.0
    raw[..., 1] = fractions[:, None, None] * 3.0
    raw[2, ..., 0] += 0.4
    prepared = np.full((3, 2, 2), (2.2, 2.7))
    anchored = anchor_displacement_history(raw, prepared)

    np.testing.assert_array_equal(anchored[0], 0.0)
    np.testing.assert_allclose(anchored[-1], prepared)
    raw_deviation = raw[2] - 0.5 * raw[-1]
    anchored_deviation = anchored[2] - 0.5 * anchored[-1]
    np.testing.assert_allclose(anchored_deviation, raw_deviation)


def test_anchor_history_rejects_nonzero_reference() -> None:
    raw = np.ones((3, 2, 2, 2))
    with pytest.raises(ValueError, match="start from zero"):
        anchor_displacement_history(raw, raw[-1])


def test_float32_history_endpoint_can_be_restored_exactly() -> None:
    endpoint = np.array([[[0.123456789, -0.234567891]]], dtype=np.float64)
    stored = np.asarray(endpoint, dtype=np.float32).astype(np.float64)
    assert np.max(np.abs(stored - endpoint)) > 0.0
    stored[...] = endpoint
    np.testing.assert_array_equal(stored, endpoint)
