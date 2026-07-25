"""Linear-system adapters used by the nonlinear finite-element solver."""

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LinearSolverStatistics:
    """Detailed timings and call counts for one linear-solver lifecycle."""

    analysis_seconds: float = 0.0
    factorization_seconds: float = 0.0
    solve_seconds: float = 0.0
    analysis_calls: int = 0
    factorization_calls: int = 0
    solve_calls: int = 0

    @property
    def total_seconds(self) -> float:
        """Return the time spent in all solver phases."""

        return self.analysis_seconds + self.factorization_seconds + self.solve_seconds


def _pattern_equal(
    matrix: csr_matrix,
    shape: tuple[int, int] | None,
    indptr: NDArray[np.int32] | None,
    indices: NDArray[np.int32] | None,
) -> bool:
    return (
        shape == matrix.shape
        and indptr is not None
        and indices is not None
        and np.array_equal(indptr, matrix.indptr)
        and np.array_equal(indices, matrix.indices)
    )


class ExplicitPardisoSolver:
    """Reuse one symbolic analysis while explicitly running phases 11/22/33.

    PyPardiso 0.4.7 exposes phase selection but its public ``spsolve`` helper
    combines analysis and numerical factorization whenever matrix values
    change. This adapter deliberately keeps ``mtype=11`` and calls the
    underlying phase interface so a fixed CSR pattern is analysed once.
    """

    backend_name = "pypardiso explicit phases 11/22/33 (mtype=11)"

    def __init__(self) -> None:
        import pypardiso

        self._solver: Any = pypardiso.PyPardisoSolver(mtype=11)
        self._shape: tuple[int, int] | None = None
        self._indptr: NDArray[np.int32] | None = None
        self._indices: NDArray[np.int32] | None = None
        self._analysis_seconds = 0.0
        self._factorization_seconds = 0.0
        self._solve_seconds = 0.0
        self._analysis_calls = 0
        self._factorization_calls = 0
        self._solve_calls = 0
        self._closed = False

    @property
    def statistics(self) -> LinearSolverStatistics:
        """Return an immutable snapshot of phase timings and counts."""

        return LinearSolverStatistics(
            analysis_seconds=self._analysis_seconds,
            factorization_seconds=self._factorization_seconds,
            solve_seconds=self._solve_seconds,
            analysis_calls=self._analysis_calls,
            factorization_calls=self._factorization_calls,
            solve_calls=self._solve_calls,
        )

    def _call_phase(
        self,
        phase: int,
        matrix: csr_matrix,
        right_hand_side: FloatArray,
    ) -> FloatArray:
        self._solver.set_phase(phase)
        return np.asarray(
            self._solver._call_pardiso(matrix, right_hand_side),
            dtype=np.float64,
        )

    def _analyse(self, matrix: csr_matrix) -> None:
        if self._shape is not None:
            if not _pattern_equal(matrix, self._shape, self._indptr, self._indices):
                raise ValueError(
                    "PARDISO matrix sparsity changed after symbolic analysis"
                )
            return
        started = time.perf_counter()
        self._call_phase(11, matrix, np.zeros(matrix.shape[0], dtype=np.float64))
        self._analysis_seconds += time.perf_counter() - started
        self._analysis_calls += 1
        self._shape = matrix.shape
        self._indptr = np.asarray(matrix.indptr, dtype=np.int32).copy()
        self._indices = np.asarray(matrix.indices, dtype=np.int32).copy()

    def factorize_and_solve(
        self,
        matrix: csr_matrix,
        right_hand_side: ArrayLike,
    ) -> FloatArray:
        """Run phase 11 once, then phases 22 and 33 for current values."""

        if self._closed:
            raise RuntimeError("PARDISO solver is closed")
        if not isinstance(matrix, csr_matrix):
            raise TypeError("matrix must be a scipy.sparse.csr_matrix")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")
        rhs = np.asarray(right_hand_side, dtype=np.float64)
        if rhs.shape not in {(matrix.shape[0],), (matrix.shape[0], 1)}:
            raise ValueError("right_hand_side has an incompatible shape")
        if not np.isfinite(matrix.data).all() or not np.isfinite(rhs).all():
            raise ValueError("matrix and right_hand_side must be finite")

        self._analyse(matrix)

        started = time.perf_counter()
        self._call_phase(22, matrix, np.zeros(matrix.shape[0], dtype=np.float64))
        self._factorization_seconds += time.perf_counter() - started
        self._factorization_calls += 1

        started = time.perf_counter()
        solution = self._call_phase(33, matrix, rhs)
        self._solve_seconds += time.perf_counter() - started
        self._solve_calls += 1
        return solution.squeeze()

    def close(self) -> None:
        """Release PARDISO internal memory once."""

        if self._closed:
            return
        self._closed = True
        self._solver.free_memory(everything=True)

    def __enter__(self) -> ExplicitPardisoSolver:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


class ScipySparseSolver:
    """Compatibility fallback when PyPardiso is unavailable."""

    backend_name = "scipy SuperLU (single-threaded)"

    def __init__(self) -> None:
        self._solve_seconds = 0.0
        self._solve_calls = 0

    @property
    def statistics(self) -> LinearSolverStatistics:
        return LinearSolverStatistics(
            solve_seconds=self._solve_seconds,
            solve_calls=self._solve_calls,
        )

    def factorize_and_solve(
        self,
        matrix: csr_matrix,
        right_hand_side: ArrayLike,
    ) -> FloatArray:
        started = time.perf_counter()
        solution = spsolve(matrix, np.asarray(right_hand_side, dtype=np.float64))
        self._solve_seconds += time.perf_counter() - started
        self._solve_calls += 1
        return np.asarray(solution, dtype=np.float64)

    def close(self) -> None:
        return None


def create_linear_solver() -> ExplicitPardisoSolver | ScipySparseSolver:
    """Create the explicit PARDISO adapter or the compatibility fallback."""

    try:
        import pypardiso  # noqa: F401
    except Exception:
        return ScipySparseSolver()
    return ExplicitPardisoSolver()
