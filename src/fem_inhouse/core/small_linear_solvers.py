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
    def _solve_small_lu(matrix: FloatArray, rhs: FloatArray) -> tuple[FloatArray, bool]:
        """Solve one small dense system by in-place partial-pivot LU."""
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


    @njit(parallel=True, cache=True, fastmath=False, boundscheck=False)
    def solve12_batch_numba(
        matrix: FloatArray, rhs: FloatArray
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        """Solve a batch of 12x12 systems with one compiled solve per point."""
        count = matrix.shape[0]
        result = np.empty((count, 12), dtype=matrix.dtype)
        success = np.empty(count, dtype=np.bool_)
        for point in prange(count):
            result[point], success[point] = _solve_small_lu(matrix[point], rhs[point])
        return result, success

else:

    def solve12_batch_numba(
        matrix: FloatArray, rhs: FloatArray
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        """Raise a clear error when the optional Numba extra is unavailable."""
        raise ImportError("solve12_batch_numba requires the optional numba dependency")
