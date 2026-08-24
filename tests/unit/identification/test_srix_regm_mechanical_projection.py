from __future__ import annotations

import numpy as np

from scripts.qualify_srix_regm_mechanical_projection import _apply_corrections


def test_apply_corrections_preserves_input_and_updates_only_replayed_states() -> None:
    history = np.zeros((3, 2, 2, 2), dtype=np.float64)
    history[0, 0, 0] = (4.0, 5.0)
    corrections = [np.ones((2, 2, 2)), 2.0 * np.ones((2, 2, 2))]
    result = _apply_corrections(history, corrections, 0.25)
    np.testing.assert_array_equal(history[0], result[0])
    np.testing.assert_allclose(result[1], 0.25)
    np.testing.assert_allclose(result[2], 0.5)
    np.testing.assert_allclose(history[1:], 0.0)
