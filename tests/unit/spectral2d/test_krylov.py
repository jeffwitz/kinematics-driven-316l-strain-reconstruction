from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

from fem_inhouse.spectral2d.krylov import KrylovRecycleState, solve_nonsymmetric_krylov


@pytest.mark.parametrize("method", ["gmres", "lgmres", "gcrotmk"])
def test_nonsymmetric_krylov_methods_solve_and_report(method: str) -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(18, 18)) + 6.0 * np.eye(18)
    rhs = rng.normal(size=18)
    operator = LinearOperator(matrix.shape, matvec=lambda value: matrix @ value, dtype=float)
    identity = LinearOperator(matrix.shape, matvec=lambda value: value, dtype=float)
    recycle = KrylovRecycleState()

    solution, info, iterations = solve_nonsymmetric_krylov(
        operator,
        rhs,
        preconditioner=identity,
        method=method,  # type: ignore[arg-type]
        rtol=1.0e-10,
        maximum_iterations=100,
        restart=10,
        recycle=recycle,
    )

    assert info == 0
    assert iterations > 0
    assert np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs) < 2.0e-10


def test_recycle_state_is_explicitly_reset() -> None:
    state = KrylovRecycleState(
        lgmres_outer_v=[(np.ones(2), np.ones(2))],
        gcrotmk_cu=[(np.ones(2), np.ones(2))],
    )
    state.reset()
    assert state.lgmres_outer_v == []
    assert state.gcrotmk_cu == []
