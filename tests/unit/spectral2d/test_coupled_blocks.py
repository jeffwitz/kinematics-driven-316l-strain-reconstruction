from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import gmres

from fem_inhouse.spectral2d.coupled_blocks import CoupledBlockActions


def test_coupled_block_operator_and_diagonal_preconditioner_match_dense_system() -> None:
    ruu_matrix = np.array([[4.0, 1.0], [0.5, 3.0]])
    ruchi_matrix = np.array([[0.4], [-0.2]])
    gu_matrix = np.array([[0.7, -0.1]])
    gchi_matrix = np.array([[2.5]])
    actions = CoupledBlockActions(
        mechanical_size=2,
        nonlocal_size=1,
        ruu=lambda value: ruu_matrix @ value,
        ruchi=lambda value: ruchi_matrix @ value,
        g_u=lambda value: gu_matrix @ value,
        g_chi=lambda value: gchi_matrix @ value,
        mechanical_inverse=lambda value: value / 3.0,
        nonlocal_inverse=lambda value: value / 2.5,
    )
    dense = np.block([[ruu_matrix, ruchi_matrix], [gu_matrix, gchi_matrix]])
    rhs = np.array([1.0, -2.0, 0.5])
    solution, info = gmres(
        actions.operator(),
        rhs,
        M=actions.preconditioner(),
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_array_equal(info, 0)
    np.testing.assert_allclose(
        solution,
        np.linalg.solve(dense, rhs),
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_coupled_block_actions_reject_wrong_block_outputs() -> None:
    actions = CoupledBlockActions(
        mechanical_size=1,
        nonlocal_size=1,
        ruu=lambda value: np.zeros(2),
        ruchi=lambda value: np.zeros(1),
        g_u=lambda value: np.zeros(1),
        g_chi=lambda value: np.zeros(1),
        mechanical_inverse=lambda value: value,
        nonlocal_inverse=lambda value: value,
    )
    with np.testing.assert_raises(ValueError):
        actions.operator().matvec(np.ones(2))
