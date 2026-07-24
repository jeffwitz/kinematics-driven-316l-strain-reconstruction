from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.core.tensor_reconstruction import (
    elastic_axial_strain_from_stress,
    engineering_strain_2d_to_tensor,
    engineering_stress_2d_to_tensor,
    kelvin_plane_stress_to_tensor,
    reconstruct_native_plane_stress_state,
    reconstruct_python_plane_stress_state,
    tensor_to_engineering_strain_2d,
    tensor_to_engineering_stress_2d,
    tensor_to_kelvin_plane_stress,
)
from fem_inhouse.postprocessing.tensor_measures import (
    instantaneous_equivalent_plastic_strain,
    reconstructed_equivalent_strain,
    von_mises_from_stress_tensor,
)


def test_engineering_tensor_round_trip_preserves_shear_and_inputs() -> None:
    strain = np.array([[0.01, -0.003, 0.008], [0.0, 0.002, -0.004]])
    stress = np.array([[120.0, -20.0, 35.0], [10.0, 15.0, -8.0]])
    strain_before = strain.copy()
    stress_before = stress.copy()

    strain_tensor = engineering_strain_2d_to_tensor(strain, [-0.004, -0.001])
    stress_tensor = engineering_stress_2d_to_tensor(stress, [1e-12, -2e-12])

    np.testing.assert_array_equal(strain, strain_before)
    np.testing.assert_array_equal(stress, stress_before)
    np.testing.assert_array_equal(strain_tensor, strain_tensor.swapaxes(-1, -2))
    np.testing.assert_array_equal(stress_tensor, stress_tensor.swapaxes(-1, -2))
    np.testing.assert_allclose(tensor_to_engineering_strain_2d(strain_tensor), strain)
    np.testing.assert_allclose(tensor_to_engineering_stress_2d(stress_tensor), stress)
    np.testing.assert_allclose(strain_tensor[..., 0, 1], strain[..., 2] / 2.0)
    np.testing.assert_allclose(stress_tensor[..., 0, 1], stress[..., 2])


@pytest.mark.parametrize("quantity", ["strain", "stress"])
def test_kelvin_tensor_round_trip_preserves_double_contraction(quantity: str) -> None:
    first = np.array([[0.3, -0.2, 0.0], [-0.2, 0.4, 0.0], [0.0, 0.0, -0.1]])
    second = np.array([[1.2, 0.3, 0.0], [0.3, -0.7, 0.0], [0.0, 0.0, 0.6]])
    first_kelvin = tensor_to_kelvin_plane_stress(first, quantity=quantity)  # type: ignore[arg-type]
    second_kelvin = tensor_to_kelvin_plane_stress(second, quantity=quantity)  # type: ignore[arg-type]

    np.testing.assert_allclose(
        np.dot(first_kelvin, second_kelvin),
        np.einsum("ij,ij->", first, second),
    )
    np.testing.assert_allclose(
        kelvin_plane_stress_to_tensor(first_kelvin, quantity=quantity),  # type: ignore[arg-type]
        first,
    )
    assert first_kelvin[3] == pytest.approx(np.sqrt(2.0) * first[0, 1])


def test_python_reconstruction_is_additive_isochoric_and_plane_stress() -> None:
    total = np.array([[[0.012, -0.001, 0.006], [0.004, 0.003, -0.002]]])
    plastic = np.array([[[0.008, -0.002, 0.004], [0.001, 0.001, -0.001]]])
    stress = np.array([[[410.0, 125.0, 42.0], [280.0, 260.0, -15.0]]])
    state = reconstruct_python_plane_stress_state(total, plastic, stress, 0.3)

    np.testing.assert_allclose(
        state.total_strain_tensor,
        state.elastic_strain_tensor + state.plastic_strain_tensor,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.trace(state.plastic_strain_tensor, axis1=-2, axis2=-1),
        0.0,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(state.plane_stress_residual_mpa, np.zeros((1, 2)))
    np.testing.assert_array_equal(
        state.plane_stress_residual_mpa,
        state.stress_tensor_mpa[..., 2, 2],
    )


def test_elastic_uniaxial_and_biaxial_axial_strain_expressions_agree() -> None:
    young = 205_000.0
    poisson = 0.3
    uniaxial_stress = np.array([[400.0, 0.0, 0.0]])
    uniaxial_total = np.array([[400.0 / young, -poisson * 400.0 / young, 0.0]])
    uniaxial = reconstruct_python_plane_stress_state(
        uniaxial_total,
        np.zeros_like(uniaxial_total),
        uniaxial_stress,
        poisson,
    )
    assert uniaxial.total_strain_tensor[0, 1, 1] == pytest.approx(
        uniaxial.total_strain_tensor[0, 2, 2]
    )

    biaxial_stress = np.array([[320.0, 180.0, 0.0]])
    compliance = (
        np.array(
            [[1.0, -poisson], [-poisson, 1.0]],
        )
        / young
    )
    biaxial_total = np.array([[*(compliance @ biaxial_stress[0, :2]), 0.0]])
    biaxial = reconstruct_python_plane_stress_state(
        biaxial_total,
        np.zeros_like(biaxial_total),
        biaxial_stress,
        poisson,
    )
    np.testing.assert_allclose(
        biaxial.elastic_strain_tensor[..., 2, 2],
        elastic_axial_strain_from_stress(
            biaxial_stress,
            young_modulus_mpa=young,
            poisson_ratio=poisson,
        ),
        rtol=0.0,
        atol=1e-15,
    )


def test_native_reconstruction_keeps_axial_values_and_residual() -> None:
    total_kelvin = np.array([[0.01, -0.001, -0.0078, 0.002 / np.sqrt(2.0)]])
    elastic_kelvin = np.array([[0.0018, 0.0002, -0.00086, 0.0003 / np.sqrt(2.0)]])
    stress_kelvin = np.array([[420.0, 160.0, -2.5e-12, np.sqrt(2.0) * 23.0]])
    state = reconstruct_native_plane_stress_state(
        total_kelvin,
        elastic_kelvin,
        stress_kelvin,
    )

    assert state.total_strain_tensor[0, 2, 2] == -0.0078
    assert state.elastic_strain_tensor[0, 2, 2] == -0.00086
    assert state.plastic_strain_tensor[0, 2, 2] == pytest.approx(-0.00694)
    assert state.plane_stress_residual_mpa[0] == -2.5e-12


def test_tensor_measures_match_known_plane_stress_values() -> None:
    stress = engineering_stress_2d_to_tensor([100.0, 0.0, 0.0])
    total = engineering_strain_2d_to_tensor([0.01, -0.002, 0.004], -0.003)
    plastic = engineering_strain_2d_to_tensor([0.006, -0.001, 0.003], -0.005)

    assert von_mises_from_stress_tensor(stress) == pytest.approx(100.0)
    assert reconstructed_equivalent_strain(total) > 0
    expected = np.sqrt((2.0 / 3.0) * np.einsum("ij,ij->", plastic, plastic))
    assert instantaneous_equivalent_plastic_strain(plastic) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (engineering_strain_2d_to_tensor, (np.zeros(4), 0.0)),
        (engineering_stress_2d_to_tensor, (np.zeros(2), 0.0)),
        (kelvin_plane_stress_to_tensor, (np.zeros(3),)),
        (tensor_to_engineering_strain_2d, (np.zeros((2, 2)),)),
    ],
)
def test_reconstruction_rejects_invalid_shapes(function: object, args: tuple[object, ...]) -> None:
    kwargs = {"quantity": "strain"} if function is kelvin_plane_stress_to_tensor else {}
    with pytest.raises(ValueError):
        function(*args, **kwargs)  # type: ignore[operator]


@pytest.mark.parametrize("poisson", [-1.0, 0.5, np.nan])
def test_reconstruction_rejects_invalid_poisson_ratio(poisson: float) -> None:
    with pytest.raises(ValueError, match="poisson_ratio"):
        reconstruct_python_plane_stress_state(
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            poisson,
        )
