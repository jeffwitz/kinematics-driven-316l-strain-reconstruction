"""Optional compiled solvers for the fixed-size local systems.

The kernels in this module are accelerators only.  NumPy/LAPACK remains the
reference implementation; callers must compare results before enabling them
in a constitutive path.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

try:  # pragma: no cover - availability depends on the optional performance extra
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None
    prange = range


if njit is not None:

    @njit(cache=True, fastmath=False, boundscheck=False)
    def _solve_small_lu_single(
        matrix: FloatArray, rhs: FloatArray
    ) -> tuple[FloatArray, bool]:
        """Solve one small dense system with one right-hand side."""
        n = matrix.shape[0]
        lu = matrix.copy()
        value = rhs.copy()
        for column in range(n - 1):
            pivot = column
            pivot_abs = abs(lu[column, column])
            for row in range(column + 1, n):
                candidate = abs(lu[row, column])
                if candidate > pivot_abs:
                    pivot = row
                    pivot_abs = candidate
            if pivot_abs <= 1.0e-14:
                return np.zeros(n, dtype=matrix.dtype), False
            if pivot != column:
                for entry in range(column, n):
                    lu[column, entry], lu[pivot, entry] = lu[pivot, entry], lu[column, entry]
                value[column], value[pivot] = value[pivot], value[column]
            for row in range(column + 1, n):
                factor = lu[row, column] / lu[column, column]
                lu[row, column] = factor
                for entry in range(column + 1, n):
                    lu[row, entry] -= factor * lu[column, entry]
                value[row] -= factor * value[column]
        if abs(lu[n - 1, n - 1]) <= 1.0e-14:
            return np.zeros(n, dtype=matrix.dtype), False
        solution = np.empty(n, dtype=matrix.dtype)
        for row in range(n - 1, -1, -1):
            total = value[row]
            for entry in range(row + 1, n):
                total -= lu[row, entry] * solution[entry]
            solution[row] = total / lu[row, row]
        return solution, True

    @njit(cache=True, fastmath=False, boundscheck=False)
    def _solve_small_lu_multi(
        matrix: FloatArray, rhs: FloatArray
    ) -> tuple[FloatArray, bool]:
        """Solve one small dense system for one or more right-hand sides."""
        n = matrix.shape[0]
        rhs_count = rhs.shape[1]
        lu = matrix.copy()
        value = rhs.copy()
        for column in range(n - 1):
            pivot = column
            pivot_abs = abs(lu[column, column])
            for row in range(column + 1, n):
                candidate = abs(lu[row, column])
                if candidate > pivot_abs:
                    pivot = row
                    pivot_abs = candidate
            if pivot_abs <= 1.0e-14:
                return np.zeros((n, rhs_count), dtype=matrix.dtype), False
            if pivot != column:
                for entry in range(column, n):
                    lu[column, entry], lu[pivot, entry] = lu[pivot, entry], lu[column, entry]
                for right in range(rhs_count):
                    value[column, right], value[pivot, right] = (
                        value[pivot, right],
                        value[column, right],
                    )
            for row in range(column + 1, n):
                factor = lu[row, column] / lu[column, column]
                lu[row, column] = factor
                for entry in range(column + 1, n):
                    lu[row, entry] -= factor * lu[column, entry]
                for right in range(rhs_count):
                    value[row, right] -= factor * value[column, right]
        if abs(lu[n - 1, n - 1]) <= 1.0e-14:
            return np.zeros((n, rhs_count), dtype=matrix.dtype), False
        solution = np.empty((n, rhs_count), dtype=matrix.dtype)
        for row in range(n - 1, -1, -1):
            for right in range(rhs_count):
                total = value[row, right]
                for entry in range(row + 1, n):
                    total -= lu[row, entry] * solution[entry, right]
                solution[row, right] = total / lu[row, row]
        return solution, True


    @njit(parallel=True, cache=True, fastmath=False, boundscheck=False)
    def solve12_batch_numba(
        matrix: FloatArray, rhs: FloatArray
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        """Solve a batch of 12x12 systems with one compiled solve per point."""
        count = matrix.shape[0]
        result = np.empty((count, 12), dtype=matrix.dtype)
        success = np.empty(count, dtype=np.bool_)
        for point in prange(count):
            result[point], success[point] = _solve_small_lu_single(matrix[point], rhs[point])
        return result, success


    @njit(parallel=True, cache=True, fastmath=False, boundscheck=False)
    def solve12_batch_rhs_numba(
        matrix: FloatArray, rhs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Solve batched 12x12 systems with multiple RHS and one LU factorization."""
        count = matrix.shape[0]
        rhs_count = rhs.shape[2]
        result = np.empty((count, 12, rhs_count), dtype=matrix.dtype)
        success = np.empty(count, dtype=np.bool_)
        for point in prange(count):
            result[point], success[point] = _solve_small_lu_multi(matrix[point], rhs[point])
        return result, success


    @njit(parallel=False, cache=True, fastmath=False, boundscheck=False)
    def solve3_batch_numba(
        matrix: FloatArray, rhs: FloatArray
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        """Solve a batch of 3x3 systems with one compiled solve per point."""
        count = matrix.shape[0]
        result = np.empty((count, 3), dtype=matrix.dtype)
        success = np.empty(count, dtype=np.bool_)
        for point in range(count):
            result[point], success[point] = _solve_small_lu_single(matrix[point], rhs[point])
        return result, success


    @njit(parallel=True, cache=True, fastmath=False, boundscheck=False)
    def solve3_batch_rhs_numba(
        matrix: FloatArray, rhs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Solve batched 3x3 systems with multiple RHS and one LU factorization."""
        count = matrix.shape[0]
        rhs_count = rhs.shape[2]
        result = np.empty((count, 3, rhs_count), dtype=matrix.dtype)
        success = np.empty(count, dtype=np.bool_)
        for point in prange(count):
            result[point], success[point] = _solve_small_lu_multi(matrix[point], rhs[point])
        return result, success

else:

    def solve12_batch_numba(
        matrix: FloatArray, rhs: FloatArray
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        """Raise a clear error when the optional Numba extra is unavailable."""
        raise ImportError("solve12_batch_numba requires the optional numba dependency")

    def solve12_batch_rhs_numba(matrix: FloatArray, rhs: NDArray[np.float64]):
        raise ImportError("solve12_batch_rhs_numba requires the optional numba dependency")

    def solve3_batch_numba(matrix: FloatArray, rhs: FloatArray):
        raise ImportError("solve3_batch_numba requires the optional numba dependency")

    def solve3_batch_rhs_numba(matrix: FloatArray, rhs: NDArray[np.float64]):
        raise ImportError("solve3_batch_rhs_numba requires the optional numba dependency")
