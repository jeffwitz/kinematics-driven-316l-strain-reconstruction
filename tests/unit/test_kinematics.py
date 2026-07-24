import numpy as np
import pytest

from fem_inhouse.postprocessing.kinematics import (
    cell_average,
    plane_stress_equivalent_strain,
    strain_from_displacement,
)


def test_affine_displacement_recovers_exact_strain_components() -> None:
    dx, dy = 0.2, 0.3
    x = np.arange(6)[:, None] * dx
    y = np.arange(5)[None, :] * dy
    u_x = 0.012 * x - 0.004 * y + np.zeros((6, 5))
    u_y = 0.007 * x + 0.020 * y + np.zeros((6, 5))

    strain = strain_from_displacement(u_x, u_y, spacing_x=dx, spacing_y=dy)

    assert strain.epsilon_xx == pytest.approx(np.full((6, 5), 0.012))
    assert strain.epsilon_yy == pytest.approx(np.full((6, 5), 0.020))
    assert strain.gamma_xy == pytest.approx(np.full((6, 5), 0.003))
    assert strain.epsilon_xy == pytest.approx(np.full((6, 5), 0.0015))


def test_cell_average_preserves_affine_centre_value() -> None:
    values = np.array([[0.0, 2.0, 4.0], [2.0, 4.0, 6.0], [4.0, 6.0, 8.0]])
    np.testing.assert_allclose(cell_average(values), [[2.0, 4.0], [4.0, 6.0]])


def test_tensorial_and_engineering_shear_give_same_invariant() -> None:
    tensorial = plane_stress_equivalent_strain(
        0.01,
        -0.002,
        0.003,
        poisson_ratio=0.3,
        shear_convention="tensorial",
    )
    engineering = plane_stress_equivalent_strain(
        0.01,
        -0.002,
        0.006,
        poisson_ratio=0.3,
        shear_convention="engineering",
    )
    assert tensorial == pytest.approx(engineering)


def test_plane_stress_invariant_matches_direct_deviatoric_calculation() -> None:
    exx, eyy, exy, nu = 0.01, 0.004, 0.002, 0.3
    ezz = -nu / (1.0 - nu) * (exx + eyy)
    strain_tensor = np.array([[exx, exy, 0.0], [exy, eyy, 0.0], [0.0, 0.0, ezz]])
    deviator = strain_tensor - np.trace(strain_tensor) / 3.0 * np.eye(3)
    expected = np.sqrt(2.0 / 3.0 * np.sum(deviator * deviator))

    actual = plane_stress_equivalent_strain(
        exx,
        eyy,
        exy,
        poisson_ratio=nu,
        shear_convention="tensorial",
    )
    assert actual == pytest.approx(expected)


def test_displacement_validation_rejects_shape_and_nan_errors() -> None:
    with pytest.raises(ValueError, match="same shape"):
        strain_from_displacement(
            np.zeros((3, 3)),
            np.zeros((4, 3)),
            spacing_x=1.0,
            spacing_y=1.0,
        )
    bad = np.zeros((3, 3))
    bad[1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        strain_from_displacement(bad, np.zeros((3, 3)), spacing_x=1.0, spacing_y=1.0)


def test_kinematic_input_contract_rejects_invalid_dimensions_and_spacing() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        strain_from_displacement(
            np.zeros(3),
            np.zeros(3),
            spacing_x=1.0,
            spacing_y=1.0,
        )
    with pytest.raises(ValueError, match="at least two"):
        strain_from_displacement(
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            spacing_x=1.0,
            spacing_y=1.0,
        )
    with pytest.raises(ValueError, match="positive"):
        strain_from_displacement(
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            spacing_x=0.0,
            spacing_y=1.0,
        )
    with pytest.raises(ValueError, match="nodal grid"):
        cell_average(np.zeros((1, 3)))


def test_plane_stress_invariant_rejects_invalid_options() -> None:
    with pytest.raises(ValueError, match="poisson_ratio"):
        plane_stress_equivalent_strain(0.0, 0.0, 0.0, poisson_ratio=0.5)
    with pytest.raises(ValueError, match="shear_convention"):
        plane_stress_equivalent_strain(
            0.0,
            0.0,
            0.0,
            poisson_ratio=0.3,
            shear_convention="invalid",
        )
