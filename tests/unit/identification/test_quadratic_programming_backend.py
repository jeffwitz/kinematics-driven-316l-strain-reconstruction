from __future__ import annotations

import numpy as np
import scipy.sparse as sparse


def test_osqp_solves_an_inequality_constrained_least_squares() -> None:
    """The dependency exists for one job; check it does that job.

    The positive-dissipation constraint is `min 1/2 a^T P a + q^T a` subject to
    `G a >= 0` with a very sparse `G` -- each row touches only two consecutive
    states, at most 2r of 320 coefficients. SciPy has no QP, and the two
    hand-rolled attempts failed for reasons that had nothing to do with the
    mechanics: a squared penalty on `min(D, 0)` has a trivial minimiser at zero,
    and an add-only active set over-determines the system once the cuts
    outnumber the unknowns.

    The problem here is small enough to have a known answer: the unconstrained
    minimiser violates the constraint, so the solution must sit exactly on it.
    """

    import osqp

    # min ||x - b||^2 with b infeasible, subject to x_0 >= 0 and x_1 >= 0.
    target = np.array([-1.0, 2.0])
    quadratic = sparse.csc_matrix(2.0 * np.eye(2))
    linear = -2.0 * target
    constraint = sparse.csc_matrix(np.eye(2))

    problem = osqp.OSQP()
    problem.setup(
        P=quadratic,
        q=linear,
        A=constraint,
        l=np.zeros(2),
        u=np.full(2, np.inf),
        verbose=False,
        eps_abs=1e-10,
        eps_rel=1e-10,
    )
    result = problem.solve()

    assert result.info.status_val in (1, 2), result.info.status
    # The first coordinate is pushed onto its bound, the second is untouched.
    np.testing.assert_allclose(result.x, [0.0, 2.0], atol=1e-6)


def test_osqp_handles_a_sparse_constraint_block_of_the_expected_shape() -> None:
    """The real problem's shape: many sparse rows, few unknowns, warm-startable.

    Each dissipation constraint couples two consecutive states, so the matrix is
    block-bidiagonal and extremely sparse. This checks that a problem of that
    shape is set up and solved, and that the solution is feasible -- the failure
    mode worth guarding is a solver that returns success on an infeasible point.
    """

    import osqp

    states, rank = 20, 16
    unknowns = states * rank
    generator = np.random.default_rng(11)

    quadratic = sparse.csc_matrix(np.eye(unknowns))
    linear = generator.normal(size=unknowns)

    rows, columns, values = [], [], []
    for row in range(2000):
        step = int(generator.integers(1, states))
        block = generator.normal(size=rank)
        for index in range(rank):
            rows.extend([row, row])
            columns.extend([step * rank + index, (step - 1) * rank + index])
            values.extend([block[index], -block[index]])
    constraint = sparse.csc_matrix(
        (values, (rows, columns)), shape=(2000, unknowns)
    )

    problem = osqp.OSQP()
    problem.setup(
        P=quadratic,
        q=linear,
        A=constraint,
        l=np.zeros(2000),
        u=np.full(2000, np.inf),
        verbose=False,
        eps_abs=1e-8,
        eps_rel=1e-8,
    )
    result = problem.solve()

    assert result.info.status_val in (1, 2), result.info.status
    assert float((constraint @ result.x).min()) > -1e-5
