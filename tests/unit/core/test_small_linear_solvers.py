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


def test_numba_lu12_multi_rhs_matches_numpy_lapack() -> None:
    pytest.importorskip("numba")
    from fem_inhouse.core.small_linear_solvers import solve12_batch_rhs_numba

    rng = np.random.default_rng(13)
    matrix = rng.normal(0.0, 0.2, (8, 12, 12)) + 3.0 * np.eye(12)[None]
    rhs = rng.normal(size=(8, 12, 6))
    expected = np.stack([np.linalg.solve(matrix[i], rhs[i]) for i in range(8)])
    actual, success = solve12_batch_rhs_numba(matrix, rhs)
    assert np.all(success)
    assert np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_numba_lu3_multi_rhs_matches_numpy_lapack() -> None:
    pytest.importorskip("numba")
    from fem_inhouse.core.small_linear_solvers import solve3_batch_rhs_numba

    rng = np.random.default_rng(14)
    matrix = rng.normal(0.0, 0.2, (8, 3, 3)) + 3.0 * np.eye(3)[None]
    rhs = rng.normal(size=(8, 3, 3))
    expected = np.stack([np.linalg.solve(matrix[i], rhs[i]) for i in range(8)])
    actual, success = solve3_batch_rhs_numba(matrix, rhs)
    assert np.all(success)
    assert np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_numba_fused_srix_jacobian_matches_explicit_system() -> None:
    pytest.importorskip("numba")
    from fem_inhouse.core.small_linear_solvers import solve12_jacobian_batch_numba

    rng = np.random.default_rng(15)
    count = 6
    slope = rng.uniform(0.01, 0.1, count)
    active = rng.integers(0, 2, (count, 12)).astype(float)
    sgn = np.where(rng.random((count, 12)) > 0.5, 1.0, -1.0)
    exp_bp = rng.uniform(0.2, 1.0, (count, 12))
    sign_dg = np.where(rng.random((count, 12)) > 0.5, 1.0, -1.0)
    dda = rng.normal(size=(count, 12))
    residual = rng.normal(size=(count, 12))
    plastic_modulus = rng.normal(size=(12, 12))
    interaction = rng.normal(size=(12, 12))
    args = (slope, active, sgn, exp_bp, sign_dg, dda, residual, plastic_modulus, interaction)
    actual, success = solve12_jacobian_batch_numba(*args, 2.0, 0.7, 1.5)
    expected = []
    for point in range(count):
        matrix = np.eye(12)
        matrix += (active[point] * slope[point])[:, None] * plastic_modulus
        matrix += (
            active[point] * slope[point] * sgn[point]
        )[:, None] * 2.0 * 0.7 * interaction * exp_bp[point][None, :] * sign_dg[point][None, :]
        matrix[np.arange(12), np.arange(12)] += active[point] * slope[point] * 1.5 * dda[point]
        expected.append(np.linalg.solve(matrix, -residual[point]))
    assert np.all(success)
    assert np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)
