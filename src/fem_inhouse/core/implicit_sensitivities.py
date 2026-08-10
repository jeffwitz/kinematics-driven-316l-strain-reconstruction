"""Implicit-function sensitivities for local constitutive systems.

For a converged local system ``F(z, q) = 0`` and an observable ``y(z, q)``,
the sensitivities with respect to parameters ``q`` are obtained from

``F_z dz_dq = -F_q``

and

``dy_dq = y_q + y_z dz_dq``.

This module deliberately knows nothing about J2 plasticity, crystal slips, or
the meaning of the local unknowns.  A constitutive adapter supplies the local
Jacobian and the partial derivatives of its residuals and observables.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def solve_implicit_sensitivities(
    local_jacobian: ArrayLike,
    residual_parameter_derivatives: ArrayLike,
    observable_state_derivatives: ArrayLike,
    observable_parameter_derivatives: ArrayLike | None = None,
) -> FloatArray:
    """Return ``dy/dq`` without explicitly forming a local inverse.

    Parameters use the following trailing dimensions and may have arbitrary
    leading batch dimensions:

    ``local_jacobian``
        ``(..., n, n)`` matrix ``F_z``.
    ``residual_parameter_derivatives``
        ``(..., n, m)`` matrix ``F_q``.
    ``observable_state_derivatives``
        ``(..., r, n)`` matrix ``y_z``.
    ``observable_parameter_derivatives``
        Optional ``(..., r, m)`` matrix ``y_q``; zero if omitted.

    The result has shape ``(..., r, m)``.  NumPy's batched solve is used so
    one factorisation handles all parameter right-hand sides at a point.
    """

    jacobian = np.asarray(local_jacobian, dtype=np.float64)
    residual_q = np.asarray(residual_parameter_derivatives, dtype=np.float64)
    observable_z = np.asarray(observable_state_derivatives, dtype=np.float64)
    if jacobian.ndim < 2 or jacobian.shape[-1] != jacobian.shape[-2]:
        raise ValueError("local_jacobian must have trailing shape (n, n)")
    n = jacobian.shape[-1]
    if residual_q.ndim < 2 or residual_q.shape[-2] != n:
        raise ValueError("residual_parameter_derivatives must have trailing shape (n, m)")
    if observable_z.ndim < 2 or observable_z.shape[-1] != n:
        raise ValueError("observable_state_derivatives must have trailing shape (r, n)")
    if (
        jacobian.shape[:-2] != residual_q.shape[:-2]
        or jacobian.shape[:-2] != observable_z.shape[:-2]
    ):
        raise ValueError("batch dimensions must match")
    if not np.isfinite(jacobian).all() or not np.isfinite(residual_q).all():
        raise ValueError("local Jacobian and residual derivatives must be finite")

    state_sensitivity = np.linalg.solve(jacobian, -residual_q)
    result = np.einsum("...rn,...nm->...rm", observable_z, state_sensitivity)
    if observable_parameter_derivatives is not None:
        observable_q = np.asarray(observable_parameter_derivatives, dtype=np.float64)
        if observable_q.shape != result.shape:
            raise ValueError("observable_parameter_derivatives has an incompatible shape")
        if not np.isfinite(observable_q).all():
            raise ValueError("observable_parameter_derivatives must be finite")
        result = result + observable_q
    return result
