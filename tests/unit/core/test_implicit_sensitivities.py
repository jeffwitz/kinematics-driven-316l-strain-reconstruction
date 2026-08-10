from __future__ import annotations

import numpy as np

from fem_inhouse.core.implicit_sensitivities import solve_implicit_sensitivities


def test_implicit_sensitivity_solves_a_scalar_local_system() -> None:
    # F(z, q) = z + 2 q0 - 3 q1 and y(z, q) = 4 z + q0.
    sensitivity = solve_implicit_sensitivities(
        [[[2.0]]],
        [[[4.0, -6.0]]],
        [[[4.0]]],
        [[[1.0, 0.0]]],
    )
    np.testing.assert_allclose(sensitivity, [[[-7.0, 12.0]]])


def test_implicit_sensitivity_is_dimension_free_for_crystal_like_system() -> None:
    rng = np.random.default_rng(12)
    jacobian = np.eye(18) + 0.02 * rng.standard_normal((4, 18, 18))
    residual_q = rng.standard_normal((4, 18, 3))
    observable_z = rng.standard_normal((4, 13, 18))
    observable_q = rng.standard_normal((4, 13, 3))

    result = solve_implicit_sensitivities(
        jacobian, residual_q, observable_z, observable_q
    )
    expected = observable_q + np.einsum(
        "...rn,...nm->...rm",
        observable_z,
        np.linalg.solve(jacobian, -residual_q),
    )
    np.testing.assert_allclose(result, expected)
    assert result.shape == (4, 13, 3)
