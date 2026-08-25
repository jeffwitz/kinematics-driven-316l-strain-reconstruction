"""SVD-derived parameter subspaces for locally identifiable coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SVDParameterBasis:
    """Orthonormal right-singular-vector basis of a parameter Jacobian."""

    singular_values: FloatArray
    normalized_singular_values: FloatArray
    right_singular_vectors: FloatArray
    retained_basis: FloatArray
    discarded_basis: FloatArray
    effective_rank: int


def svd_parameter_basis(
    jacobian: ArrayLike,
    *,
    fixed_rank: int | None = None,
    relative_threshold: float = 1.0e-6,
) -> SVDParameterBasis:
    """Return the right-singular-vector parameter basis of ``jacobian``.

    ``right_singular_vectors`` stores vectors as columns, matching the usual
    ``eta = eta_ref + V z`` convention. A fixed rank is useful for a
    preregistered experiment; otherwise the relative threshold determines the
    reported effective rank.
    """

    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("jacobian must be a finite two-dimensional array")
    if relative_threshold <= 0.0 or not np.isfinite(relative_threshold):
        raise ValueError("relative_threshold must be finite and positive")
    _, singular, right_transposed = np.linalg.svd(matrix, full_matrices=False)
    scale = max(float(singular[0]), np.finfo(float).tiny)
    normalized = singular / scale
    detected = int(np.count_nonzero(normalized > relative_threshold))
    if fixed_rank is None:
        rank = detected
    else:
        if not 0 < fixed_rank <= right_transposed.shape[0]:
            raise ValueError("fixed_rank must be between one and the parameter count")
        rank = int(fixed_rank)
    vectors = right_transposed.T.copy()
    return SVDParameterBasis(
        singular_values=singular.copy(),
        normalized_singular_values=normalized.copy(),
        right_singular_vectors=vectors,
        retained_basis=vectors[:, :rank].copy(),
        discarded_basis=vectors[:, rank:].copy(),
        effective_rank=rank,
    )


def eta_from_reduced_coordinates(
    eta_reference: ArrayLike,
    retained_basis: ArrayLike,
    z: ArrayLike,
) -> FloatArray:
    """Reconstruct log-parameters from reduced coordinates."""

    reference = np.asarray(eta_reference, dtype=np.float64)
    basis = np.asarray(retained_basis, dtype=np.float64)
    coordinates = np.asarray(z, dtype=np.float64)
    if reference.ndim != 1 or basis.ndim != 2 or coordinates.ndim != 1:
        raise ValueError("reference, basis and coordinates must be vectors/matrix/vector")
    if basis.shape[0] != reference.size or basis.shape[1] != coordinates.size:
        raise ValueError("basis dimensions do not match reference and coordinates")
    return reference + basis @ coordinates


def reduced_coordinates_from_eta(
    eta: ArrayLike,
    eta_reference: ArrayLike,
    retained_basis: ArrayLike,
) -> FloatArray:
    """Project log-parameters onto an orthonormal retained SVD basis."""

    values = np.asarray(eta, dtype=np.float64)
    reference = np.asarray(eta_reference, dtype=np.float64)
    basis = np.asarray(retained_basis, dtype=np.float64)
    if values.ndim != 1 or reference.shape != values.shape or basis.ndim != 2:
        raise ValueError("eta, reference and basis have incompatible dimensions")
    if basis.shape[0] != values.size:
        raise ValueError("basis row count must equal parameter count")
    return basis.T @ (values - reference)


def project_eta_to_basis(
    eta: ArrayLike,
    eta_reference: ArrayLike,
    retained_basis: ArrayLike,
) -> FloatArray:
    """Return the orthogonal projection of ``eta`` into the retained subspace."""

    coordinates = reduced_coordinates_from_eta(eta, eta_reference, retained_basis)
    return eta_from_reduced_coordinates(eta_reference, retained_basis, coordinates)

