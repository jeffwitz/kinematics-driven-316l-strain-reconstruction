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
