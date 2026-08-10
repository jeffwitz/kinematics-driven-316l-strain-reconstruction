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
from scipy.sparse.linalg import LinearOperator

FloatArray = NDArray[np.float64]
VectorAction = Callable[[FloatArray], FloatArray]


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
