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
    def solve12_jacobian_batch_numba(
        slope: FloatArray,
        active: FloatArray,
        sgn: FloatArray,
        exp_bp: FloatArray,
        sign_dg: FloatArray,
        dda: FloatArray,
        residual: FloatArray,
        plastic_modulus: FloatArray,
        interaction: FloatArray,
        q_mpa: float,
        b: float,
        c_mpa: float,
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        """Build and solve the reduced SRIX Newton system per point.

        This fuses the hot-path 12x12 Jacobian construction with the
        single-RHS LU solve, avoiding a temporary ``(N, 12, 12)`` array.
        ``active`` is a float mask to match the constitutive expressions.
        """
        count = residual.shape[0]
        result = np.empty((count, 12), dtype=residual.dtype)
        success = np.empty(count, dtype=np.bool_)
        for point in prange(count):
            jac = np.eye(12, dtype=residual.dtype)
            for row in range(12):
                for column in range(12):
                    jac[row, column] += (
                        active[point, row] * slope[point] * plastic_modulus[row, column]
                    )
                    jac[row, column] += (
                        active[point, row]
                        * slope[point]
                        * sgn[point, row]
                        * q_mpa
                        * b
                        * interaction[row, column]
                        * exp_bp[point, column]
                        * sign_dg[point, column]
                    )
                jac[row, row] += (
                    active[point, row] * slope[point] * c_mpa * dda[point, row]
                )
            result[point], success[point] = _solve_small_lu_single(
                jac, -residual[point]
            )
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


    @njit(parallel=True, cache=True, fastmath=False, boundscheck=False)
    def solve_coupled_block_numba(
        slope: FloatArray,
        active: FloatArray,
        sgn: FloatArray,
        exp_bp: FloatArray,
        sign_dg: FloatArray,
        dda: FloatArray,
        residual: FloatArray,
        stress_b: FloatArray,
        de: FloatArray,
        deq: FloatArray,
        overstress: FloatArray,
        mce: FloatArray,
        transform_b: FloatArray,
        plastic_modulus: FloatArray,
        interaction: FloatArray,
        dmat: FloatArray,
        cbase: FloatArray,
        q_mpa: float,
        b: float,
        c_mpa: float,
        overstress_modulus_mpa: float,
    ) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
        """Fuse coupled A/B construction, solve, Schur and correction.

        The point-local work arrays never leave this kernel.  This is an
        accelerator only; the NumPy/LAPACK path remains the reference.
        """
        count = residual.shape[0]
        delta_g = np.empty((count, 12), dtype=residual.dtype)
        delta_b = np.empty((count, 3), dtype=residual.dtype)
        success = np.empty(count, dtype=np.bool_)
        for point in prange(count):
            a = np.eye(12, dtype=residual.dtype)
            block_b = np.empty((12, 3), dtype=residual.dtype)
            ndeq = np.zeros(6, dtype=residual.dtype)
            if deq[point] > 1.0e-14:
                factor = 2.0 / (3.0 * deq[point])
                for component in range(6):
                    ndeq[component] = factor * de[point, component]
            jfd = np.empty((12, 6), dtype=residual.dtype)
            for row in range(12):
                row_factor = active[point, row] * slope[point]
                for column in range(12):
                    a[row, column] += row_factor * plastic_modulus[row, column]
                    a[row, column] += (
                        row_factor
                        * sgn[point, row]
                        * q_mpa
                        * b
                        * interaction[row, column]
                        * exp_bp[point, column]
                        * sign_dg[point, column]
                    )
                a[row, row] += row_factor * c_mpa * dda[point, row]
                for component in range(6):
                    jfd[row, component] = (
                        -row_factor * mce[row, component]
                        - overstress[point, row]
                        * sgn[point, row]
                        / overstress_modulus_mpa
                        * ndeq[component]
                    )
                for component in range(3):
                    value = 0.0
                    for material_component in range(6):
                        value += jfd[row, material_component] * transform_b[
                            point, material_component, component
                        ]
                    block_b[row, component] = value
            rhs = np.empty((12, 4), dtype=residual.dtype)
            for row in range(12):
                rhs[row, 0] = residual[point, row]
                for component in range(3):
                    rhs[row, component + 1] = block_b[row, component]
            a_solution, ok_a = _solve_small_lu_multi(a, rhs)
            if not ok_a:
                success[point] = False
                delta_g[point] = 0.0
                delta_b[point] = 0.0
                continue
            schur = np.empty((3, 3), dtype=residual.dtype)
            rhs_schur = np.empty(3, dtype=residual.dtype)
            inv_a_r = np.empty(12, dtype=residual.dtype)
            for row in range(12):
                inv_a_r[row] = a_solution[row, 0]
            inv_a_b = a_solution[:, 1:]
            for row in range(3):
                rhs_schur[row] = -stress_b[point, row]
                for column in range(12):
                    rhs_schur[row] += cbase[point, row, column] * inv_a_r[column]
                for column in range(3):
                    value = dmat[point, row, column]
                    for inner in range(12):
                        value -= cbase[point, row, inner] * inv_a_b[inner, column]
                    schur[row, column] = value
            db, ok_b = _solve_small_lu_single(schur, rhs_schur)
            if not ok_b:
                success[point] = False
                delta_g[point] = 0.0
                delta_b[point] = 0.0
                continue
            for row in range(3):
                delta_b[point, row] = db[row]
            for row in range(12):
                value = -inv_a_r[row]
                for column in range(3):
                    value -= inv_a_b[row, column] * db[column]
                delta_g[point, row] = value
            success[point] = True
        return delta_g, delta_b, success


    @njit(parallel=False, cache=True, fastmath=False, boundscheck=False)
    def solve_coupled_tangent_numba(
        slope: FloatArray,
        active: FloatArray,
        sgn: FloatArray,
        exp_bp: FloatArray,
        sign_dg: FloatArray,
        dda: FloatArray,
        de: FloatArray,
        deq: FloatArray,
        overstress: FloatArray,
        mce: FloatArray,
        transform_a: FloatArray,
        transform_b: FloatArray,
        plastic_modulus: FloatArray,
        interaction: FloatArray,
        dmat: FloatArray,
        c: FloatArray,
        g: FloatArray,
        h: FloatArray,
        i: FloatArray,
        j: FloatArray,
        q_mpa: float,
        b: float,
        c_mpa: float,
        overstress_modulus_mpa: float,
    ) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
        """Fuse the direct coupled plane-stress tangent point by point."""
        count = slope.shape[0]
        cps = np.empty((count, 3, 3), dtype=slope.dtype)
        cbb = np.empty((count, 3, 3), dtype=slope.dtype)
        cba = np.empty((count, 3, 3), dtype=slope.dtype)
        success = np.empty(count, dtype=np.bool_)
        for point in range(count):
            a = np.eye(12, dtype=slope.dtype)
            e = np.empty((12, 3), dtype=slope.dtype)
            block_b = np.empty((12, 3), dtype=slope.dtype)
            ndeq = np.zeros(6, dtype=slope.dtype)
            if deq[point] > 1.0e-14:
                factor = 2.0 / (3.0 * deq[point])
                for component in range(6):
                    ndeq[component] = factor * de[point, component]
            for row in range(12):
                row_factor = active[point, row] * slope[point]
                for column in range(12):
                    a[row, column] += row_factor * plastic_modulus[row, column]
                    a[row, column] += (
                        row_factor
                        * sgn[point, row]
                        * q_mpa
                        * b
                        * interaction[row, column]
                        * exp_bp[point, column]
                        * sign_dg[point, column]
                    )
                a[row, row] += row_factor * c_mpa * dda[point, row]
                for component in range(6):
                    value_a = 0.0
                    value_b = 0.0
                    for material_component in range(6):
                        local_jfd = (
                            -row_factor * mce[row, material_component]
                            - overstress[point, row]
                            * sgn[point, row]
                            / overstress_modulus_mpa
                            * ndeq[material_component]
                        )
                        value_a += local_jfd * transform_a[
                            point, material_component, component
                        ]
                        value_b += local_jfd * transform_b[
                            point, material_component, component
                        ]
                    e[row, component] = value_a
                    block_b[row, component] = value_b
            rhs = np.empty((12, 6), dtype=slope.dtype)
            for row in range(12):
                for component in range(3):
                    rhs[row, component] = e[row, component]
                    rhs[row, component + 3] = block_b[row, component]
            a_solution, ok_a = _solve_small_lu_multi(a, rhs)
            if not ok_a:
                success[point] = False
                cps[point] = 0.0
                cbb[point] = 0.0
                cba[point] = 0.0
                continue
            schur = np.empty((3, 3), dtype=slope.dtype)
            cba_point = np.empty((3, 3), dtype=slope.dtype)
            for row in range(3):
                for column in range(3):
                    value = g[point, row, column]
                    for inner in range(12):
                        value -= c[point, row, inner] * a_solution[inner, column]
                    cba_point[row, column] = value
                    schur[row, column] = dmat[point, row, column]
                    for inner in range(12):
                        schur[row, column] -= c[point, row, inner] * a_solution[inner, column + 3]
            de_b, ok_b = _solve_small_lu_multi(schur, -cba_point)
            if not ok_b:
                success[point] = False
                cps[point] = 0.0
                cbb[point] = 0.0
                cba[point] = 0.0
                continue
            dg_de = np.empty((12, 3), dtype=slope.dtype)
            for row in range(12):
                for column in range(3):
                    value = -a_solution[row, column]
                    for inner in range(3):
                        value -= a_solution[row, inner + 3] * de_b[inner, column]
                    dg_de[row, column] = value
            for row in range(3):
                for column in range(3):
                    value = h[point, row, column]
                    for inner in range(3):
                        value += i[point, row, inner] * de_b[inner, column]
                    for inner in range(12):
                        value += j[point, row, inner] * dg_de[inner, column]
                    cps[point, row, column] = value
                    cbb[point, row, column] = schur[row, column]
                    cba[point, row, column] = cba_point[row, column]
            success[point] = True
        return cps, cbb, cba, success

else:

    def solve12_batch_numba(
        matrix: FloatArray, rhs: FloatArray
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        """Raise a clear error when the optional Numba extra is unavailable."""
        raise ImportError("solve12_batch_numba requires the optional numba dependency")

    def solve12_batch_rhs_numba(matrix: FloatArray, rhs: NDArray[np.float64]):
        raise ImportError("solve12_batch_rhs_numba requires the optional numba dependency")

    def solve12_jacobian_batch_numba(*args, **kwargs):
        raise ImportError("solve12_jacobian_batch_numba requires the optional numba dependency")

    def solve3_batch_numba(matrix: FloatArray, rhs: FloatArray):
        raise ImportError("solve3_batch_numba requires the optional numba dependency")

    def solve3_batch_rhs_numba(matrix: FloatArray, rhs: NDArray[np.float64]):
        raise ImportError("solve3_batch_rhs_numba requires the optional numba dependency")

    def solve_coupled_block_numba(*args, **kwargs):
        raise ImportError("solve_coupled_block_numba requires the optional numba dependency")

    def solve_coupled_tangent_numba(*args, **kwargs):
        raise ImportError("solve_coupled_tangent_numba requires the optional numba dependency")
