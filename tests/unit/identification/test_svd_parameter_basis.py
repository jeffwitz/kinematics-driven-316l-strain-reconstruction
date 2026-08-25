import numpy as np
import pytest

from fem_inhouse.identification.svd_parameter_basis import (
    eta_from_reduced_coordinates,
    project_eta_to_basis,
    reduced_coordinates_from_eta,
    svd_parameter_basis,
)


def test_full_basis_round_trip_is_exact() -> None:
    matrix = np.diag([4.0, 3.0, 2.0, 1.0])
    basis = svd_parameter_basis(matrix, fixed_rank=4)
    reference = np.array([1.0, -2.0, 0.5, 3.0])
    eta = np.array([1.2, -1.8, 0.4, 2.7])
    z = reduced_coordinates_from_eta(eta, reference, basis.retained_basis)
    reconstructed = eta_from_reduced_coordinates(reference, basis.retained_basis, z)
    np.testing.assert_allclose(reconstructed, eta, atol=1.0e-14)
    np.testing.assert_allclose(
        basis.right_singular_vectors.T @ basis.right_singular_vectors,
        np.eye(4),
        atol=1.0e-14,
    )


def test_rank_three_projection_discards_only_fourth_direction() -> None:
    basis = svd_parameter_basis(np.diag([4.0, 3.0, 2.0, 1.0]), fixed_rank=3)
    reference = np.zeros(4)
    eta = np.ones(4)
    projected = project_eta_to_basis(eta, reference, basis.retained_basis)
    assert np.isclose(np.linalg.norm(projected), np.sqrt(3.0))
    np.testing.assert_allclose(
        basis.retained_basis.T @ (projected - reference),
        basis.retained_basis.T @ (eta - reference),
    )
    np.testing.assert_allclose(basis.discarded_basis.T @ (projected - reference), 0.0)


def test_invalid_fixed_rank_is_rejected() -> None:
    with pytest.raises(ValueError):
        svd_parameter_basis(np.eye(4), fixed_rank=0)
    with pytest.raises(ValueError):
        svd_parameter_basis(np.eye(4), fixed_rank=5)

