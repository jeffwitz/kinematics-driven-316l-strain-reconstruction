"""Applied displacement extensions for DIC boundary data."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from fem_inhouse.spectral2d.grid import StructuredGrid2D

FloatArray = NDArray[np.float64]


def _validate_boundary_field(
    boundary_displacement: ArrayLike, grid: StructuredGrid2D
) -> FloatArray:
    values = np.asarray(boundary_displacement, dtype=np.float64)
    expected = (*grid.node_shape, 2)
    if values.shape != expected:
        raise ValueError(f"expected boundary displacement shape {expected}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("boundary displacement must be finite")
    return values


@runtime_checkable
class AppliedDisplacementExtension2D(Protocol):
    """Extend contour displacement data to the complete nodal grid."""

    def extend(self, boundary_displacement: ArrayLike, grid: StructuredGrid2D) -> FloatArray: ...


class HarmonicDirichletExtension2D:
    """Discrete harmonic extension with the supplied values on every border."""

    def extend(self, boundary_displacement: ArrayLike, grid: StructuredGrid2D) -> FloatArray:
        values = _validate_boundary_field(boundary_displacement, grid)
        if grid.nx < 2 or grid.ny < 2:
            return values.copy()

        nx_i, ny_i = grid.interior_shape
        n_unknowns = nx_i * ny_i
        matrix = _interior_laplacian(grid)
        extended = values.copy()
        for component in range(2):
            rhs = np.zeros(n_unknowns, dtype=np.float64)
            for i in range(1, grid.nx):
                for j in range(1, grid.ny):
                    row = (i - 1) * ny_i + (j - 1)
                    rhs[row] = _boundary_laplacian_rhs(values[..., component], grid, i, j)
            solution = np.asarray(spsolve(matrix, rhs), dtype=np.float64)
            extended[1:-1, 1:-1, component] = solution.reshape(nx_i, ny_i)
        return extended


class TransfiniteBoundaryInterpolation2D:
    """Bilinear transfinite interpolation used as a deterministic test baseline."""

    def extend(self, boundary_displacement: ArrayLike, grid: StructuredGrid2D) -> FloatArray:
        values = _validate_boundary_field(boundary_displacement, grid)
        result = values.copy()
        x = np.linspace(0.0, 1.0, grid.nx + 1)
        y = np.linspace(0.0, 1.0, grid.ny + 1)
        for i in range(1, grid.nx):
            for j in range(1, grid.ny):
                result[i, j] = (
                    (1.0 - x[i]) * values[0, j]
                    + x[i] * values[-1, j]
                    + (1.0 - y[j]) * values[i, 0]
                    + y[j] * values[i, -1]
                    - (
                        (1.0 - x[i]) * (1.0 - y[j]) * values[0, 0]
                        + x[i] * (1.0 - y[j]) * values[-1, 0]
                        + (1.0 - x[i]) * y[j] * values[0, -1]
                        + x[i] * y[j] * values[-1, -1]
                    )
                )
        return result


def _interior_laplacian(grid: StructuredGrid2D) -> csr_matrix:
    nx_i, ny_i = grid.interior_shape
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    inv_hx2 = 1.0 / grid.spacing_x**2
    inv_hy2 = 1.0 / grid.spacing_y**2
    for i in range(nx_i):
        for j in range(ny_i):
            row = i * ny_i + j
            rows.append(row)
            columns.append(row)
            data.append(2.0 * inv_hx2 + 2.0 * inv_hy2)
            for di, dj, weight in (
                (-1, 0, -inv_hx2),
                (1, 0, -inv_hx2),
                (0, -1, -inv_hy2),
                (0, 1, -inv_hy2),
            ):
                ii, jj = i + di, j + dj
                if 0 <= ii < nx_i and 0 <= jj < ny_i:
                    rows.append(row)
                    columns.append(ii * ny_i + jj)
                    data.append(weight)
    return csr_matrix((data, (rows, columns)), shape=(nx_i * ny_i, nx_i * ny_i))


def _boundary_laplacian_rhs(field: FloatArray, grid: StructuredGrid2D, i: int, j: int) -> float:
    rhs = 0.0
    inv_hx2 = 1.0 / grid.spacing_x**2
    inv_hy2 = 1.0 / grid.spacing_y**2
    if i == 1:
        rhs += inv_hx2 * field[0, j]
    if i == grid.nx - 1:
        rhs += inv_hx2 * field[-1, j]
    if j == 1:
        rhs += inv_hy2 * field[i, 0]
    if j == grid.ny - 1:
        rhs += inv_hy2 * field[i, -1]
    return rhs
