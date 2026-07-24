"""Bilinear plane-stress CPS4 element operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.core.mesh import StructuredMesh

GAUSS_COORDINATE = 1.0 / np.sqrt(3.0)
GAUSS_POINTS = np.array(
    [
        [-GAUSS_COORDINATE, -GAUSS_COORDINATE],
        [GAUSS_COORDINATE, -GAUSS_COORDINATE],
        [GAUSS_COORDINATE, GAUSS_COORDINATE],
        [-GAUSS_COORDINATE, GAUSS_COORDINATE],
    ]
)
GAUSS_WEIGHTS = np.ones(4)
GAUSS_POINT_COUNT = 4


@dataclass(frozen=True, slots=True)
class ElementOperators:
    """Constant operators shared by every element of a regular mesh."""

    elastic_stiffness: NDArray
    strain_displacement: NDArray
    jacobian_determinants: NDArray


def shape_function_derivatives(xi: float, eta: float) -> NDArray:
    """Derivatives of the four shape functions in natural coordinates."""

    return 0.25 * np.array(
        [
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
        ]
    )


def strain_displacement_matrix(
    coordinates: NDArray,
    xi: float,
    eta: float,
) -> tuple[NDArray, float]:
    """Return engineering-shear B and the positive Jacobian determinant."""

    natural_derivatives = shape_function_derivatives(xi, eta)
    jacobian = natural_derivatives @ coordinates
    determinant = float(np.linalg.det(jacobian))
    if determinant <= 0:
        raise ValueError("CPS4 element has a non-positive Jacobian")
    physical_derivatives = np.linalg.solve(jacobian, natural_derivatives)
    matrix = np.zeros((3, 8))
    for node in range(4):
        matrix[0, 2 * node] = physical_derivatives[0, node]
        matrix[1, 2 * node + 1] = physical_derivatives[1, node]
        matrix[2, 2 * node] = physical_derivatives[1, node]
        matrix[2, 2 * node + 1] = physical_derivatives[0, node]
    return matrix, determinant


def plane_stress_elasticity(young_modulus_mpa: float, poisson_ratio: float) -> NDArray:
    """Engineering-shear isotropic plane-stress elasticity matrix."""

    if young_modulus_mpa <= 0:
        raise ValueError("young_modulus_mpa must be positive")
    if not -1 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must satisfy -1 < nu < 0.5")
    factor = young_modulus_mpa / (1 - poisson_ratio**2)
    return factor * np.array(
        [
            [1, poisson_ratio, 0],
            [poisson_ratio, 1, 0],
            [0, 0, (1 - poisson_ratio) / 2],
        ]
    )


def precompute_element(
    mesh: StructuredMesh,
    elasticity: NDArray,
) -> ElementOperators:
    """Compute the constant CPS4 operators once for a regular mesh."""

    coordinates = mesh.coords[mesh.conn[0]]
    stiffness = np.zeros((8, 8))
    matrices = np.zeros((GAUSS_POINT_COUNT, 3, 8))
    determinants = np.zeros(GAUSS_POINT_COUNT)
    for index, (point, weight) in enumerate(zip(GAUSS_POINTS, GAUSS_WEIGHTS, strict=True)):
        matrix, determinant = strain_displacement_matrix(
            coordinates,
            point[0],
            point[1],
        )
        stiffness += weight * (matrix.T @ elasticity @ matrix) * determinant
        matrices[index] = matrix
        determinants[index] = determinant
    return ElementOperators(stiffness, matrices, determinants)
