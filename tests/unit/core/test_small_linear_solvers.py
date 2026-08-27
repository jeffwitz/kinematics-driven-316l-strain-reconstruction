"""Checks for optional fixed-size local linear-system accelerators."""

import numpy as np
import pytest


def test_numba_lu12_matches_numpy_lapack() -> None:
    pytest.importorskip("numba")
    from fem_inhouse.core.small_linear_solvers import solve12_batch_numba

    rng = np.random.default_rng(12)
    matrix = rng.normal(0.0, 0.2, (8, 12, 12)) + 3.0 * np.eye(12)[None]
    rhs = rng.normal(size=(8, 12))
    expected = np.linalg.solve(matrix, rhs[..., None])[..., 0]
    actual, success = solve12_batch_numba(matrix, rhs)
    assert np.all(success)
    assert np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)

