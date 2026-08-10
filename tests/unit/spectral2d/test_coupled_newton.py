from __future__ import annotations

import numpy as np

from fem_inhouse.spectral2d.coupled_blocks import CoupledBlockActions
from fem_inhouse.spectral2d.coupled_newton import (
    CoupledLinearisation,
    CoupledNewtonConfig,
    solve_coupled_newton,
)


def test_experimental_coupled_newton_solves_a_nonlinear_block_problem() -> None:
    target = np.array([1.0, -0.5, 0.25])

    def evaluate(state: tuple[np.ndarray, np.ndarray]) -> CoupledLinearisation:
        u, chi = state
        ru = np.array([u[0] ** 2 + 0.2 * chi[0] - 1.0, u[1] + 0.1 * chi[0] + 0.5])
        g = np.array([chi[0] + 0.3 * u[0] - 0.3 * target[2]])
        return CoupledLinearisation(
            mechanical_residual=ru,
            nonlocal_residual=g,
            actions=CoupledBlockActions(
                mechanical_size=2,
                nonlocal_size=1,
                ruu=lambda value: np.array([[2.0 * u[0], 0.0], [0.0, 1.0]]) @ value,
                ruchi=lambda value: np.array([0.2 * value[0], 0.1 * value[0]]),
                g_u=lambda value: np.array([0.3 * value[0]]),
                g_chi=lambda value: np.array([value[0]]),
                mechanical_inverse=lambda value: value / 2.0,
                nonlocal_inverse=lambda value: value,
            ),
        )

    result = solve_coupled_newton(
        [1.2, -0.2],
        [0.1],
        evaluate,
        config=CoupledNewtonConfig(maximum_iterations=12),
    )
    assert result.converged
    np.testing.assert_allclose(result.final_residual_norm, 0.0, atol=1.0e-8)
    np.testing.assert_allclose(result.nonlocal_field, [0.3 * (target[2] - result.mechanical[0])])


def test_coupled_newton_does_not_linearise_after_convergence() -> None:
    linearisation_calls = 0
    residual_calls = 0

    def evaluate(state: tuple[np.ndarray, np.ndarray]) -> CoupledLinearisation:
        nonlocal linearisation_calls
        linearisation_calls += 1
        u, chi = state
        ru = np.array([u[0] - 1.0])
        g = np.array([chi[0] - 0.25])
        return CoupledLinearisation(
            mechanical_residual=ru,
            nonlocal_residual=g,
            actions=CoupledBlockActions(
                mechanical_size=1,
                nonlocal_size=1,
                ruu=lambda value: value,
                ruchi=lambda value: np.zeros(1),
                g_u=lambda value: np.zeros(1),
                g_chi=lambda value: value,
                mechanical_inverse=lambda value: value,
                nonlocal_inverse=lambda value: value,
            ),
        )

    def evaluate_residual(
        state: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal residual_calls
        residual_calls += 1
        return state[0] - 1.0, state[1] - 0.25

    result = solve_coupled_newton(
        [0.0],
        [0.0],
        evaluate,
        evaluate_residual=evaluate_residual,
        config=CoupledNewtonConfig(maximum_iterations=4),
    )

    assert result.converged
    assert linearisation_calls == 1
    assert residual_calls == 2
