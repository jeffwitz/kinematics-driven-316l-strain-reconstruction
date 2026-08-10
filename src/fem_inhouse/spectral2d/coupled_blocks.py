"""Matrix-free block operators for an experimental ``(u, chi)`` solve.

The production solver is still partitioned.  This module only supplies the
linear-algebra seam needed by a future monolithic Newton driver: four
matrix-free Jacobian blocks and a block-diagonal preconditioner.  The
mechanical and non-local diagonal inverses are intentionally injected by the
caller, so they can be the DST-I ``B0`` and DCT-II Helmholtz actions without
coupling this module to either transform implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import dctn, idctn
from scipy.sparse.linalg import LinearOperator

from fem_inhouse.spectral2d.transforms import TransformPlan2D

FloatArray = NDArray[np.float64]
VectorAction = Callable[[FloatArray], FloatArray]
CoupledAction = Callable[[FloatArray, FloatArray], tuple[FloatArray, FloatArray]]


def make_dst_b0_inverse(transform_plan: TransformPlan2D, green_operator: object) -> VectorAction:
    """Create a flat-vector action for the DST-I/$B_0^{-1}$ inverse."""

    interior_shape = tuple(transform_plan.diagnostics.interior_shape)
    expected_size = int(np.prod(interior_shape)) * 2

    def apply(value: FloatArray) -> FloatArray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (expected_size,):
            raise ValueError(
                f"mechanical vector has shape {vector.shape}, expected {(expected_size,)}"
            )
        physical = vector.reshape((*interior_shape, 2))
        transformed = np.asarray(transform_plan.forward_displacement(physical))
        reference = np.asarray(green_operator.apply(transformed))
        result = np.asarray(transform_plan.inverse_displacement(reference))
        return result.reshape(-1)

    return apply


def make_dct_helmholtz_inverse(
    shape: tuple[int, int],
    *,
    length_scale: float,
    spacing_x: float,
    spacing_y: float,
) -> VectorAction:
    """Create a flat-vector action for $(I-l^2 L_N)^{-1}$ by DCT-II."""

    if min(shape) < 1 or length_scale < 0.0 or spacing_x <= 0.0 or spacing_y <= 0.0:
        raise ValueError("invalid Helmholtz inverse geometry or length scale")
    nx, ny = shape
    wave_x = np.arange(nx, dtype=np.float64)
    wave_y = np.arange(ny, dtype=np.float64)
    eigenvalues_x = (2.0 - 2.0 * np.cos(np.pi * wave_x / nx)) / spacing_x**2
    eigenvalues_y = (2.0 - 2.0 * np.cos(np.pi * wave_y / ny)) / spacing_y**2
    denominator = 1.0 + length_scale**2 * (
        eigenvalues_x[:, np.newaxis] + eigenvalues_y[np.newaxis, :]
    )

    def apply(value: FloatArray) -> FloatArray:
        vector = np.asarray(value, dtype=np.float64)
        expected_size = nx * ny
        if vector.shape != (expected_size,):
            raise ValueError(
                f"non-local vector has shape {vector.shape}, expected {(expected_size,)}"
            )
        field = vector.reshape(shape)
        transformed = dctn(field, type=2, norm="ortho")
        filtered = idctn(transformed / denominator, type=2, norm="ortho")
        return np.asarray(filtered, dtype=np.float64).reshape(-1)

    return apply


def make_dct_helmholtz_operator(
    shape: tuple[int, int],
    *,
    length_scale: float,
    spacing_x: float,
    spacing_y: float,
) -> VectorAction:
    """Create the Neumann Helmholtz action ``H = I-l^2 Delta``.

    The coupled formulation uses ``H chi - p = 0`` rather than applying
    ``H^-1`` to the residual.  The inverse remains the preconditioner; this
    direct action avoids a DCT/IDCT pair in every lower-block matvec.
    """

    if min(shape) < 1 or length_scale < 0.0 or spacing_x <= 0.0 or spacing_y <= 0.0:
        raise ValueError("invalid Helmholtz operator geometry or length scale")
    nx, ny = shape
    def apply(value: FloatArray) -> FloatArray:
        vector = np.asarray(value, dtype=np.float64)
        expected_size = nx * ny
        if vector.shape != (expected_size,):
            raise ValueError(
                f"non-local vector has shape {vector.shape}, expected {(expected_size,)}"
            )
        field = vector.reshape(shape)
        laplacian = np.zeros_like(field)
        if nx == 1:
            laplacian[:, :] = 0.0
        else:
            laplacian[0, :] += (field[0, :] - field[1, :]) / spacing_x**2
            laplacian[-1, :] += (field[-1, :] - field[-2, :]) / spacing_x**2
            if nx > 2:
                laplacian[1:-1, :] += (
                    2.0 * field[1:-1, :] - field[:-2, :] - field[2:, :]
                ) / spacing_x**2
        if ny == 1:
            laplacian[:, :] += 0.0
        else:
            laplacian[:, 0] += (field[:, 0] - field[:, 1]) / spacing_y**2
            laplacian[:, -1] += (field[:, -1] - field[:, -2]) / spacing_y**2
            if ny > 2:
                laplacian[:, 1:-1] += (
                    2.0 * field[:, 1:-1] - field[:, :-2] - field[:, 2:]
                ) / spacing_y**2
        return (field + length_scale**2 * laplacian).reshape(-1)

    return apply


@dataclass(frozen=True, slots=True)
class CoupledBlockActions:
    """Matrix-free actions for the four blocks of a coupled linearisation."""

    mechanical_size: int
    nonlocal_size: int
    ruu: VectorAction
    ruchi: VectorAction
    g_u: VectorAction
    g_chi: VectorAction
    mechanical_inverse: VectorAction
    nonlocal_inverse: VectorAction
    combined: CoupledAction | None = None

    def __post_init__(self) -> None:
        if self.mechanical_size < 1 or self.nonlocal_size < 1:
            raise ValueError("block sizes must be positive")

    @property
    def size(self) -> int:
        return self.mechanical_size + self.nonlocal_size

    def _checked(self, value: ArrayLike, size: int, name: str) -> FloatArray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (size,):
            raise ValueError(f"{name} returned {result.shape}, expected {(size,)}")
        if not np.isfinite(result).all():
            raise ValueError(f"{name} returned non-finite values")
        return result

    def operator(self) -> LinearOperator:
        """Return the full matrix-free block Jacobian operator."""

        nu = self.mechanical_size
        nc = self.nonlocal_size

        def matvec(value: ArrayLike) -> FloatArray:
            vector = np.asarray(value, dtype=np.float64)
            if vector.shape != (nu + nc,):
                raise ValueError(f"vector has shape {vector.shape}, expected {(nu + nc,)}")
            du = vector[:nu]
            dchi = vector[nu:]
            if self.combined is not None:
                upper, lower = self.combined(du, dchi)
                return np.concatenate(
                    (
                        self._checked(upper, nu, "combined R_u action"),
                        self._checked(lower, nc, "combined non-local action"),
                    )
                )
            upper = self._checked(self.ruu(du), nu, "R_u action")
            upper += self._checked(self.ruchi(dchi), nu, "R_chi action")
            lower = self._checked(self.g_u(du), nc, "G_u action")
            lower += self._checked(self.g_chi(dchi), nc, "G_chi action")
            return np.concatenate((upper, lower))

        return LinearOperator((nu + nc, nu + nc), matvec=matvec, dtype=np.float64)

    def preconditioner(self) -> LinearOperator:
        """Return the block-diagonal inverse preconditioner."""

        nu = self.mechanical_size
        nc = self.nonlocal_size

        def matvec(value: ArrayLike) -> FloatArray:
            vector = np.asarray(value, dtype=np.float64)
            if vector.shape != (nu + nc,):
                raise ValueError(f"vector has shape {vector.shape}, expected {(nu + nc,)}")
            du = self._checked(self.mechanical_inverse(vector[:nu]), nu, "B0 inverse")
            dchi = self._checked(self.nonlocal_inverse(vector[nu:]), nc, "Helmholtz inverse")
            return np.concatenate((du, dchi))

        return LinearOperator((nu + nc, nu + nc), matvec=matvec, dtype=np.float64)
