"""Sparse assembly and internal-force operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from fem_inhouse.core.mesh import StructuredMesh

AssemblyIndices = tuple[NDArray, NDArray]
CSRStorage = Literal["full", "upper"]


@dataclass(slots=True)
class FixedCSRAssembler:
    """Assemble changing element values into one immutable CSR structure."""

    matrix: csr_matrix
    contribution_positions: NDArray[np.int32]
    element_count: int
    storage: CSRStorage

    @classmethod
    def from_location_matrix(
        cls,
        location_matrix: NDArray,
        selected_dofs: NDArray,
        *,
        chunk_size: int = 8_192,
        storage: CSRStorage = "full",
    ) -> FixedCSRAssembler:
        """Build a reduced CSR pattern and element-to-data mapping once."""

        location = np.asarray(location_matrix, dtype=np.int64)
        dofs = np.asarray(selected_dofs, dtype=np.int64)
        if location.ndim != 2 or location.shape[1] != 8:
            raise ValueError("location_matrix must have shape (n_elements, 8)")
        if dofs.ndim != 1 or dofs.size == 0:
            raise ValueError("selected_dofs must be a non-empty one-dimensional array")
        if np.any(dofs < 0) or len(np.unique(dofs)) != len(dofs):
            raise ValueError("selected_dofs must contain unique nonnegative indices")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if storage not in {"full", "upper"}:
            raise ValueError("storage must be 'full' or 'upper'")
        maximum_dof = int(max(np.max(location), np.max(dofs)))
        global_to_reduced = np.full(maximum_dof + 1, -1, dtype=np.int64)
        global_to_reduced[dofs] = np.arange(len(dofs), dtype=np.int64)
        reduced_location = global_to_reduced[location]
        reduced_size = len(dofs)
        sentinel_key = reduced_size * reduced_size

        keys = np.empty((len(location), 8, 8), dtype=np.int64)
        np.multiply(
            reduced_location[:, :, None],
            reduced_size,
            out=keys,
        )
        np.add(keys, reduced_location[:, None, :], out=keys)
        invalid = (reduced_location[:, :, None] < 0) | (
            reduced_location[:, None, :] < 0
        )
        if storage == "upper":
            invalid |= reduced_location[:, :, None] > reduced_location[:, None, :]
        keys[invalid] = sentinel_key
        unique_keys = np.unique(keys)
        if unique_keys[-1] == sentinel_key:
            unique_keys = unique_keys[:-1]
        del keys, invalid
        if unique_keys.size == 0:
            raise ValueError("selected_dofs produce an empty stiffness pattern")
        if unique_keys.size >= np.iinfo(np.int32).max:
            raise ValueError("fixed CSR pattern exceeds int32 mapping capacity")

        rows = unique_keys // reduced_size
        indices = np.asarray(unique_keys % reduced_size, dtype=np.int32)
        row_counts = np.bincount(rows, minlength=reduced_size)
        indptr = np.empty(reduced_size + 1, dtype=np.int32)
        indptr[0] = 0
        np.cumsum(row_counts, dtype=np.int64, out=indptr[1:])
        matrix = csr_matrix(
            (
                np.zeros(unique_keys.size, dtype=np.float64),
                indices,
                indptr,
            ),
            shape=(reduced_size, reduced_size),
        )
        matrix.has_sorted_indices = True

        positions = np.empty((len(location), 8, 8), dtype=np.int32)
        sentinel_position = int(unique_keys.size)
        for start in range(0, len(location), chunk_size):
            stop = min(start + chunk_size, len(location))
            local = reduced_location[start:stop]
            chunk_keys = np.empty((stop - start, 8, 8), dtype=np.int64)
            np.multiply(local[:, :, None], reduced_size, out=chunk_keys)
            np.add(chunk_keys, local[:, None, :], out=chunk_keys)
            chunk_invalid = (local[:, :, None] < 0) | (local[:, None, :] < 0)
            if storage == "upper":
                chunk_invalid |= local[:, :, None] > local[:, None, :]
            chunk_keys[chunk_invalid] = sentinel_key
            chunk_positions = np.searchsorted(unique_keys, chunk_keys)
            chunk_positions[chunk_invalid] = sentinel_position
            positions[start:stop] = chunk_positions.astype(np.int32, copy=False)

        return cls(
            matrix=matrix,
            contribution_positions=positions,
            element_count=len(location),
            storage=storage,
        )

    def assemble(self, element_stiffness: NDArray) -> csr_matrix:
        """Update only ``matrix.data`` and return the same CSR object."""

        values = np.asarray(element_stiffness, dtype=np.float64)
        if values.shape == (8, 8):
            flat_values = np.broadcast_to(
                values,
                (self.element_count, 8, 8),
            ).reshape(-1)
        elif values.shape == (self.element_count, 8, 8):
            flat_values = values.reshape(-1)
        else:
            raise ValueError(
                "element_stiffness must have shape (8, 8) "
                "or (n_elements, 8, 8)"
            )
        sums = np.bincount(
            self.contribution_positions.reshape(-1),
            weights=flat_values,
            minlength=self.matrix.nnz + 1,
        )
        np.copyto(self.matrix.data, sums[: self.matrix.nnz])
        return self.matrix


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
    gauss_weights: NDArray,
    *,
    element_count: int,
    chunk_size: int = 8_192,
) -> NDArray:
    """Add plastic Gauss-point corrections without a dense all-point tensor.

    The quadrature arrives as an argument rather than a module constant so the
    reduced formulation, which has one point instead of four, uses the same
    code path.
    """

    gauss_count = len(gauss_weights)
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
        integration_weights = gauss_weights[gauss_indices] * jacobian_determinants[gauss_indices]
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
    gauss_weights: NDArray,
) -> NDArray:
    """Assemble nodal internal forces from Gauss-point stresses."""

    element_force = np.einsum(
        "g,g,gak,ega->ek",
        gauss_weights,
        jacobian_determinants,
        strain_displacement,
        stress,
    )
    force = np.zeros(mesh.n_dof)
    np.add.at(force, location_matrix, element_force)
    return force


def hourglass_force(
    mesh: StructuredMesh,
    hourglass_stiffness: NDArray,
    displacement: NDArray,
    location_matrix: NDArray,
) -> NDArray:
    """Nodal force of the hourglass stabilisation, `sum_e K_hg u_e`.

    It must be added to the internal force everywhere the internal force is
    used -- residual, line-search trials, reactions -- or the Newton iteration
    would be solving a different problem from the one its tangent describes.
    The stabilisation is already inside the element tangent, which starts from
    the elastic stiffness.
    """

    element_displacement = displacement[location_matrix]
    element_force = element_displacement @ hourglass_stiffness
    force = np.zeros(mesh.n_dof)
    np.add.at(force, location_matrix, element_force)
    return force


def hourglass_energy_by_element(
    hourglass_stiffness: NDArray,
    displacement: NDArray,
    location_matrix: NDArray,
) -> NDArray:
    """`0.5 u_e^T K_hg u_e` for every element.

    This is a NUMERICAL energy, not a physical one: it is the work done against
    a stabilisation that exists only because one integration point cannot see
    the hourglass modes. It is reported separately from the constitutive energy
    for exactly that reason.
    """

    element_displacement = displacement[location_matrix]
    return 0.5 * np.einsum(
        "ei,ij,ej->e", element_displacement, hourglass_stiffness, element_displacement
    )
