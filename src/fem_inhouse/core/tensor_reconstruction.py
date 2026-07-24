"""Vectorized reconstruction of full 3D tensors from plane-stress states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
TensorQuantity = Literal["strain", "stress"]
_SQRT_TWO = np.sqrt(2.0)


@dataclass(frozen=True, slots=True)
class FullTensorState:
    """Full symmetric tensors associated with one converged plane-stress state."""

    stress_tensor_mpa: FloatArray
    total_strain_tensor: FloatArray
    elastic_strain_tensor: FloatArray
    plastic_strain_tensor: FloatArray
    plane_stress_residual_mpa: FloatArray
    plane_stress_residual_vector_mpa: FloatArray


def _finite_components(values: ArrayLike, *, name: str, size: int) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim < 1 or array.shape[-1] != size:
        raise ValueError(f"{name} must have a trailing dimension of {size}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_tensor(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim < 2 or array.shape[-2:] != (3, 3):
        raise ValueError(f"{name} must have trailing dimensions (3, 3)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(array, array.swapaxes(-1, -2), rtol=0.0, atol=1e-14):
        raise ValueError(f"{name} must be symmetric")
    return array


def _broadcast_scalar(
    values: ArrayLike,
    *,
    leading_shape: tuple[int, ...],
    name: str,
) -> FloatArray:
    array = np.asarray(values, dtype=float)
    try:
        broadcast = np.broadcast_to(array, leading_shape)
    except ValueError as error:
        raise ValueError(f"{name} cannot be broadcast to shape {leading_shape}") from error
    if not np.isfinite(broadcast).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(broadcast, dtype=float)


def _validate_poisson_ratio(poisson_ratio: float) -> float:
    value = float(poisson_ratio)
    if not np.isfinite(value) or not -1.0 < value < 0.5:
        raise ValueError("poisson_ratio must be finite and satisfy -1 < nu < 0.5")
    return value


def engineering_strain_2d_to_tensor(
    strain_2d: ArrayLike,
    strain_33: ArrayLike,
) -> FloatArray:
    """Convert ``[..., e11, e22, gamma12]`` to symmetric 3D strain tensors."""

    strain = _finite_components(strain_2d, name="strain_2d", size=3)
    axial = _broadcast_scalar(
        strain_33,
        leading_shape=strain.shape[:-1],
        name="strain_33",
    )
    tensor = np.zeros((*strain.shape[:-1], 3, 3), dtype=float)
    tensor[..., 0, 0] = strain[..., 0]
    tensor[..., 1, 1] = strain[..., 1]
    tensor[..., 2, 2] = axial
    tensor[..., 0, 1] = 0.5 * strain[..., 2]
    tensor[..., 1, 0] = tensor[..., 0, 1]
    return tensor


def engineering_stress_2d_to_tensor(
    stress_2d_mpa: ArrayLike,
    stress_33_mpa: ArrayLike = 0.0,
) -> FloatArray:
    """Convert ``[..., s11, s22, s12]`` to symmetric 3D stress tensors."""

    stress = _finite_components(stress_2d_mpa, name="stress_2d_mpa", size=3)
    axial = _broadcast_scalar(
        stress_33_mpa,
        leading_shape=stress.shape[:-1],
        name="stress_33_mpa",
    )
    tensor = np.zeros((*stress.shape[:-1], 3, 3), dtype=float)
    tensor[..., 0, 0] = stress[..., 0]
    tensor[..., 1, 1] = stress[..., 1]
    tensor[..., 2, 2] = axial
    tensor[..., 0, 1] = stress[..., 2]
    tensor[..., 1, 0] = tensor[..., 0, 1]
    return tensor


def kelvin_plane_stress_to_tensor(
    values: ArrayLike,
    *,
    quantity: TensorQuantity,
) -> FloatArray:
    """Convert ``[..., 11, 22, 33, sqrt(2)*12]`` Kelvin values to tensors."""

    if quantity not in {"strain", "stress"}:
        raise ValueError("quantity must be 'strain' or 'stress'")
    kelvin = _finite_components(values, name=f"Kelvin {quantity}", size=4)
    tensor = np.zeros((*kelvin.shape[:-1], 3, 3), dtype=float)
    tensor[..., 0, 0] = kelvin[..., 0]
    tensor[..., 1, 1] = kelvin[..., 1]
    tensor[..., 2, 2] = kelvin[..., 2]
    tensor[..., 0, 1] = kelvin[..., 3] / _SQRT_TWO
    tensor[..., 1, 0] = tensor[..., 0, 1]
    return tensor


def tensor_to_kelvin_plane_stress(
    tensor: ArrayLike,
    *,
    quantity: TensorQuantity,
) -> FloatArray:
    """Convert a symmetric membrane tensor to four-component Kelvin notation."""

    if quantity not in {"strain", "stress"}:
        raise ValueError("quantity must be 'strain' or 'stress'")
    values = _finite_tensor(tensor, name=f"{quantity}_tensor")
    if not np.allclose(values[..., (0, 1), 2], 0.0, rtol=0.0, atol=1e-14):
        raise ValueError(f"{quantity}_tensor must have zero transverse shear")
    kelvin = np.empty((*values.shape[:-2], 4), dtype=float)
    kelvin[..., 0] = values[..., 0, 0]
    kelvin[..., 1] = values[..., 1, 1]
    kelvin[..., 2] = values[..., 2, 2]
    kelvin[..., 3] = _SQRT_TWO * values[..., 0, 1]
    return kelvin


def kelvin_3d_to_tensor(
    values: ArrayLike,
    *,
    quantity: TensorQuantity,
) -> FloatArray:
    """Convert MGIS ``[11, 22, 33, 12, 13, 23]`` Kelvin values to tensors."""

    if quantity not in {"strain", "stress"}:
        raise ValueError("quantity must be 'strain' or 'stress'")
    kelvin = _finite_components(values, name=f"3D Kelvin {quantity}", size=6)
    tensor = np.zeros((*kelvin.shape[:-1], 3, 3), dtype=float)
    tensor[..., 0, 0] = kelvin[..., 0]
    tensor[..., 1, 1] = kelvin[..., 1]
    tensor[..., 2, 2] = kelvin[..., 2]
    tensor[..., 0, 1] = kelvin[..., 3] / _SQRT_TWO
    tensor[..., 0, 2] = kelvin[..., 4] / _SQRT_TWO
    tensor[..., 1, 2] = kelvin[..., 5] / _SQRT_TWO
    tensor[..., 1, 0] = tensor[..., 0, 1]
    tensor[..., 2, 0] = tensor[..., 0, 2]
    tensor[..., 2, 1] = tensor[..., 1, 2]
    return tensor


def tensor_to_kelvin_3d(
    tensor: ArrayLike,
    *,
    quantity: TensorQuantity,
) -> FloatArray:
    """Convert a symmetric 3D tensor to MGIS Kelvin notation."""

    if quantity not in {"strain", "stress"}:
        raise ValueError("quantity must be 'strain' or 'stress'")
    values = _finite_tensor(tensor, name=f"{quantity}_tensor")
    kelvin = np.empty((*values.shape[:-2], 6), dtype=float)
    kelvin[..., 0] = values[..., 0, 0]
    kelvin[..., 1] = values[..., 1, 1]
    kelvin[..., 2] = values[..., 2, 2]
    kelvin[..., 3] = _SQRT_TWO * values[..., 0, 1]
    kelvin[..., 4] = _SQRT_TWO * values[..., 0, 2]
    kelvin[..., 5] = _SQRT_TWO * values[..., 1, 2]
    return kelvin


def tensor_to_engineering_strain_2d(tensor: ArrayLike) -> FloatArray:
    """Extract ``[..., e11, e22, gamma12]`` from symmetric strain tensors."""

    values = _finite_tensor(tensor, name="strain_tensor")
    return np.stack(
        (values[..., 0, 0], values[..., 1, 1], 2.0 * values[..., 0, 1]),
        axis=-1,
    )


def tensor_to_engineering_stress_2d(tensor_mpa: ArrayLike) -> FloatArray:
    """Extract ``[..., s11, s22, s12]`` from symmetric stress tensors."""

    values = _finite_tensor(tensor_mpa, name="stress_tensor_mpa")
    return np.stack(
        (values[..., 0, 0], values[..., 1, 1], values[..., 0, 1]),
        axis=-1,
    )


def reconstruct_python_plane_stress_state(
    total_strain_2d: ArrayLike,
    plastic_strain_2d: ArrayLike,
    stress_2d_mpa: ArrayLike,
    poisson_ratio: float,
) -> FullTensorState:
    """Reconstruct a full state from converged 2D J2 plane-stress fields."""

    total = _finite_components(total_strain_2d, name="total_strain_2d", size=3)
    plastic = _finite_components(plastic_strain_2d, name="plastic_strain_2d", size=3)
    stress = _finite_components(stress_2d_mpa, name="stress_2d_mpa", size=3)
    if plastic.shape != total.shape or stress.shape != total.shape:
        raise ValueError("total strain, plastic strain, and stress must have identical shapes")
    poisson = _validate_poisson_ratio(poisson_ratio)

    elastic = total - plastic
    plastic_33 = -(plastic[..., 0] + plastic[..., 1])
    elastic_33 = -poisson / (1.0 - poisson) * (elastic[..., 0] + elastic[..., 1])
    total_33 = elastic_33 + plastic_33
    residual = np.zeros(total.shape[:-1], dtype=float)
    residual_vector = np.zeros((*total.shape[:-1], 3), dtype=float)

    return FullTensorState(
        stress_tensor_mpa=engineering_stress_2d_to_tensor(stress, residual),
        total_strain_tensor=engineering_strain_2d_to_tensor(total, total_33),
        elastic_strain_tensor=engineering_strain_2d_to_tensor(elastic, elastic_33),
        plastic_strain_tensor=engineering_strain_2d_to_tensor(plastic, plastic_33),
        plane_stress_residual_mpa=residual,
        plane_stress_residual_vector_mpa=residual_vector,
    )


def reconstruct_native_plane_stress_state(
    total_strain_kelvin: ArrayLike,
    elastic_strain_kelvin: ArrayLike,
    stress_kelvin_mpa: ArrayLike,
) -> FullTensorState:
    """Build a full state from native four-component MFront/MGIS quantities."""

    total_kelvin = _finite_components(
        total_strain_kelvin,
        name="total_strain_kelvin",
        size=4,
    )
    elastic_kelvin = _finite_components(
        elastic_strain_kelvin,
        name="elastic_strain_kelvin",
        size=4,
    )
    stress_kelvin = _finite_components(
        stress_kelvin_mpa,
        name="stress_kelvin_mpa",
        size=4,
    )
    if elastic_kelvin.shape != total_kelvin.shape or stress_kelvin.shape != total_kelvin.shape:
        raise ValueError(
            "native total strain, elastic strain, and stress must have identical shapes"
        )
    plastic_kelvin = total_kelvin - elastic_kelvin
    stress_tensor = kelvin_plane_stress_to_tensor(stress_kelvin, quantity="stress")
    residual_vector = np.zeros((*stress_tensor.shape[:-2], 3), dtype=float)
    residual_vector[..., 0] = stress_tensor[..., 2, 2]
    return FullTensorState(
        stress_tensor_mpa=stress_tensor,
        total_strain_tensor=kelvin_plane_stress_to_tensor(total_kelvin, quantity="strain"),
        elastic_strain_tensor=kelvin_plane_stress_to_tensor(
            elastic_kelvin,
            quantity="strain",
        ),
        plastic_strain_tensor=kelvin_plane_stress_to_tensor(
            plastic_kelvin,
            quantity="strain",
        ),
        plane_stress_residual_mpa=stress_tensor[..., 2, 2].copy(),
        plane_stress_residual_vector_mpa=residual_vector,
    )


def elastic_axial_strain_from_stress(
    stress_2d_mpa: ArrayLike,
    *,
    young_modulus_mpa: float,
    poisson_ratio: float,
) -> FloatArray:
    """Evaluate ``ee33 = -nu/E * (s11+s22)`` as a consistency check."""

    stress = _finite_components(stress_2d_mpa, name="stress_2d_mpa", size=3)
    young = float(young_modulus_mpa)
    if not np.isfinite(young) or young <= 0:
        raise ValueError("young_modulus_mpa must be finite and positive")
    poisson = _validate_poisson_ratio(poisson_ratio)
    return -poisson / young * (stress[..., 0] + stress[..., 1])
