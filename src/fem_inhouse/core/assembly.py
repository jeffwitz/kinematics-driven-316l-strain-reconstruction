"""Sparse assembly and internal-force operations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from fem_inhouse.core.element import GAUSS_WEIGHTS
from fem_inhouse.core.mesh import StructuredMesh

AssemblyIndices = tuple[NDArray, NDArray]


def assembly_indices(location_matrix: NDArray) -> AssemblyIndices:
    """Precompute raveled COO row and column indices."""

    return (
        location_matrix[:, :, None].repeat(8, axis=2).ravel(),
        location_matrix[:, None, :].repeat(8, axis=1).ravel(),
    )


def assemble_stiffness(
    mesh: StructuredMesh,
    element_stiffness: NDArray,
    location_matrix: NDArray,
    indices: AssemblyIndices | None = None,
) -> csr_matrix:
    """Assemble one shared or one-per-element 8-by-8 stiffness matrix."""

    rows, columns = assembly_indices(location_matrix) if indices is None else indices
    if element_stiffness.shape == (8, 8):
        data = np.broadcast_to(element_stiffness, (mesh.n_elems, 8, 8))
    elif element_stiffness.shape == (mesh.n_elems, 8, 8):
        data = element_stiffness
    else:
        raise ValueError("element_stiffness must have shape (8, 8) or (n_elements, 8, 8)")
    return coo_matrix(
        (data.ravel(), (rows, columns)),
        shape=(mesh.n_dof, mesh.n_dof),
    ).tocsr()


def element_tangent_stiffness(
    elastic_element_stiffness: NDArray,
    elasticity: NDArray,
    plastic_tangents: NDArray,
    plastic_flat_indices: NDArray,
    strain_displacement: NDArray,
    jacobian_determinants: NDArray,
    *,
    element_count: int,
    chunk_size: int = 8_192,
) -> NDArray:
    """Add plastic Gauss-point corrections without a dense all-point tensor."""

    gauss_count = len(GAUSS_WEIGHTS)
    indices = np.asarray(plastic_flat_indices, dtype=np.int64)
    if elastic_element_stiffness.shape != (8, 8):
        raise ValueError("elastic_element_stiffness must have shape (8, 8)")
    if elasticity.shape != (3, 3):
        raise ValueError("elasticity must have shape (3, 3)")
    if plastic_tangents.shape != (len(indices), 3, 3):
        raise ValueError("plastic_tangents must have shape (n_plastic, 3, 3)")
    if strain_displacement.shape != (gauss_count, 3, 8):
        raise ValueError("strain_displacement has an invalid shape")
    if jacobian_determinants.shape != (gauss_count,):
        raise ValueError("jacobian_determinants has an invalid shape")
    if element_count < 1:
        raise ValueError("element_count must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if np.any(indices < 0) or np.any(indices >= element_count * gauss_count):
        raise ValueError("plastic_flat_indices contain an out-of-range index")

    result = np.broadcast_to(
        elastic_element_stiffness,
        (element_count, 8, 8),
    ).copy()
    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        chunk_indices = indices[start:stop]
        element_indices = chunk_indices // gauss_count
        gauss_indices = chunk_indices % gauss_count
        matrices_b = strain_displacement[gauss_indices]
        tangent_difference = plastic_tangents[start:stop] - elasticity
        integration_weights = GAUSS_WEIGHTS[gauss_indices] * jacobian_determinants[gauss_indices]
        correction = np.einsum(
            "n,nai,nab,nbj->nij",
            integration_weights,
            matrices_b,
            tangent_difference,
            matrices_b,
            optimize=True,
        )
        np.add.at(result, element_indices, correction)
    return result


def internal_force(
    mesh: StructuredMesh,
    stress: NDArray,
    strain_displacement: NDArray,
    jacobian_determinants: NDArray,
    location_matrix: NDArray,
) -> NDArray:
    """Assemble nodal internal forces from Gauss-point stresses."""

    element_force = np.einsum(
        "g,g,gak,ega->ek",
        GAUSS_WEIGHTS,
        jacobian_determinants,
        strain_displacement,
        stress,
    )
    force = np.zeros(mesh.n_dof)
    np.add.at(force, location_matrix, element_force)
    return force
