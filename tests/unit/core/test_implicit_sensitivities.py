from __future__ import annotations

import numpy as np

from fem_inhouse.core.constitutive_sensitivities import finite_difference_sensitivities
from fem_inhouse.core.implicit_sensitivities import (
    ImplicitSensitivityBlocks,
    solve_implicit_sensitivities,
)


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


def test_exported_blocks_are_a_law_independent_adapter_contract() -> None:
    blocks = ImplicitSensitivityBlocks(
        local_jacobian=np.array([[[2.0]]]),
        residual_parameter_derivatives=np.array([[[4.0, -6.0]]]),
        observable_state_derivatives=np.array([[[4.0]]]),
        observable_parameter_derivatives=np.array([[[1.0, 0.0]]]),
    )
    np.testing.assert_allclose(blocks.solve(), [[[-7.0, 12.0]]])


def test_finite_difference_fallback_is_dimension_and_law_independent() -> None:
    def response(strain: np.ndarray, parameter: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        parameter_values = parameter if parameter.ndim == 2 else parameter[:, None]
        stress = strain @ np.diag([2.0, 3.0, 5.0]) + parameter_values[:, :1]
        observable = strain @ np.array([7.0, -2.0, 11.0]) + 4.0 * parameter_values[:, 0]
        return stress, observable

    result = finite_difference_sensitivities(
        response,
        np.ones((4, 3)),
        np.full(4, 0.3),
        strain_step=1.0e-7,
        parameter_step=1.0e-7,
    )
    np.testing.assert_allclose(result.observable_strain, np.tile([7.0, -2.0, 11.0], (4, 1)))
    np.testing.assert_allclose(result.observable_parameter, 4.0)
    np.testing.assert_allclose(result.stress_parameter[..., 0], 1.0)
