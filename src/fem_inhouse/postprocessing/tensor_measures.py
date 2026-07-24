"""Invariant measures evaluated from complete symmetric 3D tensors."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _finite_tensor(values: ArrayLike, *, name: str) -> FloatArray:
    tensor = np.asarray(values, dtype=float)
    if tensor.ndim < 2 or tensor.shape[-2:] != (3, 3):
        raise ValueError(f"{name} must have trailing dimensions (3, 3)")
    if not np.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(tensor, tensor.swapaxes(-1, -2), rtol=0.0, atol=1e-14):
        raise ValueError(f"{name} must be symmetric")
    return tensor


def reconstructed_equivalent_strain(total_strain_tensor: ArrayLike) -> FloatArray:
    """Return ``sqrt(2/3 * dev(e):dev(e))`` from a full total-strain tensor."""

    strain = _finite_tensor(total_strain_tensor, name="total_strain_tensor")
    trace = np.trace(strain, axis1=-2, axis2=-1)
    deviator = strain - trace[..., None, None] * np.eye(3) / 3.0
    double_contraction = np.einsum("...ij,...ij->...", deviator, deviator)
    return np.sqrt(np.maximum((2.0 / 3.0) * double_contraction, 0.0))


def instantaneous_equivalent_plastic_strain(
    plastic_strain_tensor: ArrayLike,
) -> FloatArray:
    """Return the tensor norm of plastic strain, which is not accumulated PEEQ."""

    plastic = _finite_tensor(plastic_strain_tensor, name="plastic_strain_tensor")
    double_contraction = np.einsum("...ij,...ij->...", plastic, plastic)
    return np.sqrt(np.maximum((2.0 / 3.0) * double_contraction, 0.0))


def von_mises_from_stress_tensor(stress_tensor_mpa: ArrayLike) -> FloatArray:
    """Return ``sqrt(3/2 * dev(sigma):dev(sigma))`` from full Cauchy stress."""

    stress = _finite_tensor(stress_tensor_mpa, name="stress_tensor_mpa")
    trace = np.trace(stress, axis1=-2, axis2=-1)
    deviator = stress - trace[..., None, None] * np.eye(3) / 3.0
    double_contraction = np.einsum("...ij,...ij->...", deviator, deviator)
    return np.sqrt(np.maximum(1.5 * double_contraction, 0.0))
