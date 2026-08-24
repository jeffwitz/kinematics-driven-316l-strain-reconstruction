"""Tests for the twin-only latent-mode reconstruction helper."""

from __future__ import annotations

import numpy as np

from scripts.qualify_srix_regm_latent_modes import _pod_missing_modes, _rank_history


def test_rank_zero_is_observed_and_full_rank_recovers_raw() -> None:
    observed = np.zeros((3, 2, 2, 2), dtype=float)
    raw = observed.copy()
    raw[1, 0, 0, 0] = 2.0
    raw[2, 1, 1, 1] = -3.0
    left, singular, right = _pod_missing_modes(raw, observed)
    assert np.array_equal(_rank_history(observed, left, singular, right, 0), observed)
    np.testing.assert_allclose(_rank_history(observed, left, singular, right, len(singular)), raw)
