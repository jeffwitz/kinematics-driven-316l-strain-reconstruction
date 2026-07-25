import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fem_inhouse.core.linear_solver import ExplicitPardisoSolver


def test_explicit_pardiso_reuses_analysis_for_changing_values() -> None:
    pytest.importorskip("pypardiso")
    matrix = csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    right_hand_side = np.array([1.0, 2.0])

    with ExplicitPardisoSolver() as solver:
        first = solver.factorize_and_solve(matrix, right_hand_side)
        matrix.data[:] = np.array([5.0, 1.0, 1.0, 4.0])
        second = solver.factorize_and_solve(matrix, right_hand_side)
        statistics = solver.statistics

    np.testing.assert_allclose(
        first,
        np.linalg.solve([[4.0, 1.0], [1.0, 3.0]], right_hand_side),
    )
    np.testing.assert_allclose(
        second,
        np.linalg.solve([[5.0, 1.0], [1.0, 4.0]], right_hand_side),
    )
    assert statistics.analysis_calls == 1
    assert statistics.factorization_calls == 2
    assert statistics.solve_calls == 2
    assert statistics.total_seconds > 0.0


def test_explicit_pardiso_rejects_pattern_change_after_analysis() -> None:
    pytest.importorskip("pypardiso")
    first = csr_matrix(np.array([[4.0, 0.0], [0.0, 3.0]]))
    changed_pattern = csr_matrix(np.array([[4.0, 1.0], [0.0, 3.0]]))

    with ExplicitPardisoSolver() as solver:
        solver.factorize_and_solve(first, np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="sparsity changed"):
            solver.factorize_and_solve(
                changed_pattern,
                np.array([1.0, 2.0]),
            )


def test_explicit_pardiso_rejects_use_after_close() -> None:
    pytest.importorskip("pypardiso")
    matrix = csr_matrix(np.eye(2))
    solver = ExplicitPardisoSolver()
    solver.close()
    with pytest.raises(RuntimeError, match="closed"):
        solver.factorize_and_solve(matrix, np.ones(2))
