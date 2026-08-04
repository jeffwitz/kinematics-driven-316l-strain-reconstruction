import numpy as np
import pytest

from fem_inhouse.spectral2d import (
    B0Green2D,
    ReferenceOperatorSymbols,
    TwoMuGreen2D,
    project_isotropic_plane_stress_tangent,
)


def test_b0_modal_factors_match_the_diagonal_reference_operator() -> None:
    symbols = ReferenceOperatorSymbols(
        laplacian=np.array([[2.0, 5.0], [7.0, 11.0]]),
        directional_x=np.array([[0.5, 1.0], [1.5, 2.0]]),
        directional_y=np.array([[1.5, 1.0], [0.5, 2.0]]),
    )
    green = B0Green2D(symbols, lambda_0=4.0, mu_0=3.0)
    displacement = np.ones((2, 2, 2))
    reference_force = green.reference_force(displacement)
    expected_x = -(6.0 * symbols.laplacian + 4.0 * symbols.directional_x)
    expected_y = -(6.0 * symbols.laplacian + 4.0 * symbols.directional_y)
    np.testing.assert_allclose(reference_force[..., 0], expected_x)
    np.testing.assert_allclose(reference_force[..., 1], expected_y)


def test_b0_green_matches_modal_inverse_and_nulls_zero_mode() -> None:
    symbols = ReferenceOperatorSymbols(
        laplacian=np.array([[0.0, 5.0], [7.0, 12.0]]),
        directional_x=np.array([[0.0, 2.0], [3.0, 8.0]]),
        directional_y=np.array([[0.0, 3.0], [4.0, 4.0]]),
    )
    green = B0Green2D(symbols, lambda_0=2.0, mu_0=3.0)
    polarization = np.ones((2, 2, 2))
    result = green.apply(polarization)
    denominator_x = 6.0 * symbols.laplacian + 2.0 * symbols.directional_x
    denominator_y = 6.0 * symbols.laplacian + 2.0 * symbols.directional_y

    np.testing.assert_array_equal(result[0, 0], 0.0)
    np.testing.assert_allclose(result[1:, :, 0], -1.0 / denominator_x[1:, :])
    np.testing.assert_allclose(result[1:, :, 1], -1.0 / denominator_y[1:, :])
    assert green.diagnostics.null_modes == 1


def test_two_mu_green_uses_same_laplacian_for_both_components() -> None:
    symbols = ReferenceOperatorSymbols(
        laplacian=np.array([[2.0, 4.0]]),
        directional_x=np.array([[1.0, 3.0]]),
        directional_y=np.array([[1.0, 3.0]]),
    )
    green = TwoMuGreen2D(symbols, mu_0=2.0)
    result = green.apply(np.ones((1, 2, 2)))
    expected = np.broadcast_to(-1.0 / (4.0 * symbols.laplacian)[..., None], result.shape)
    np.testing.assert_allclose(result, expected)


def test_isotropic_projection_recovers_plane_stress_parameters() -> None:
    tangent = np.array([[8.0, 2.0, 0.0], [2.0, 8.0, 0.0], [0.0, 0.0, 3.0]])
    lambda_0, mu_0, error = project_isotropic_plane_stress_tangent(tangent)
    assert lambda_0 == pytest.approx(2.0)
    assert mu_0 == pytest.approx(3.0)
    assert error == pytest.approx(0.0)


def test_anisotropic_projection_uses_kelvin_energy_metric() -> None:
    tangent = np.diag([10.0, 10.0, 1.0])
    lambda_0, mu_0, _ = project_isotropic_plane_stress_tangent(tangent)
    assert lambda_0 == pytest.approx(2.0)
    assert mu_0 == pytest.approx(3.0)


def test_green_rejects_invalid_reference_parameters() -> None:
    symbols = ReferenceOperatorSymbols(
        laplacian=np.ones((2, 2)),
        directional_x=np.ones((2, 2)),
        directional_y=np.ones((2, 2)),
    )
    with pytest.raises(ValueError, match=r"lambda\+mu"):
        B0Green2D(symbols, lambda_0=-2.0, mu_0=1.0)


def test_green_inverts_its_reference_force() -> None:
    symbols = ReferenceOperatorSymbols(
        laplacian=np.array([[2.0, 4.0]]),
        directional_x=np.array([[1.0, 3.0]]),
        directional_y=np.array([[1.0, 3.0]]),
    )
    green = B0Green2D(symbols, lambda_0=2.0, mu_0=3.0)
    displacement = np.array([[[1.0, -2.0], [0.5, 3.0]]])
    np.testing.assert_allclose(green.apply(green.reference_force(displacement)), displacement)
