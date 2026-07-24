import numpy as np
import pytest

from fem_inhouse.postprocessing.stress_curves import (
    direct_fe_equivalent_stress_curve,
    reconstructed_equivalent_stress,
    reconstructed_stress_curve_from_strain,
    von_mises_stress,
)


def test_plane_stress_von_mises_known_states() -> None:
    assert von_mises_stress(100.0, 0.0, 0.0) == pytest.approx(100.0)
    assert von_mises_stress(100.0, 100.0, 0.0) == pytest.approx(100.0)
    assert von_mises_stress(0.0, 0.0, 100.0) == pytest.approx(np.sqrt(3.0) * 100.0)


def test_direct_curve_averages_components_before_von_mises() -> None:
    sxx = np.array([[[100.0, 200.0], [300.0, 400.0]]])
    syy = np.array([[[20.0, 40.0], [60.0, 80.0]]])
    sxy = np.array([[[10.0, 20.0], [30.0, 40.0]]])

    direct = direct_fe_equivalent_stress_curve(sxx, syy, sxy)
    expected = von_mises_stress(250.0, 50.0, 25.0)
    assert direct == pytest.approx(np.array([expected]))


def test_strain_reconstruction_uses_spatially_averaged_equivalent_strain() -> None:
    evm = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.001, 0.002], [0.003, 0.004]],
        ]
    )
    curve = reconstructed_stress_curve_from_strain(
        evm,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.3,
        yield_stress_mpa=124.0,
        hardening_coefficient_mpa=380.0,
        hardening_exponent=0.245,
    )
    expected = reconstructed_equivalent_stress(
        np.array([0.0, 0.0025]),
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.3,
        yield_stress_mpa=124.0,
        hardening_coefficient_mpa=380.0,
        hardening_exponent=0.245,
    )
    assert curve == pytest.approx(expected)


def test_direct_and_reconstructed_curves_remain_distinct() -> None:
    direct = direct_fe_equivalent_stress_curve(
        np.full((1, 2, 2), 80.0),
        np.full((1, 2, 2), 20.0),
        np.zeros((1, 2, 2)),
    )
    reconstructed = reconstructed_stress_curve_from_strain(
        np.full((1, 2, 2), 0.005),
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.3,
        yield_stress_mpa=124.0,
        hardening_coefficient_mpa=380.0,
        hardening_exponent=0.245,
    )
    assert direct[0] != pytest.approx(reconstructed[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"young_modulus_mpa": 0},
        {"poisson_ratio": 0.5},
        {"yield_stress_mpa": 0},
        {"hardening_coefficient_mpa": -1},
        {"hardening_exponent": 0},
    ],
)
def test_invalid_reconstruction_parameters_are_rejected(kwargs) -> None:
    valid = {
        "young_modulus_mpa": 205_000.0,
        "poisson_ratio": 0.3,
        "yield_stress_mpa": 124.0,
        "hardening_coefficient_mpa": 380.0,
        "hardening_exponent": 0.245,
    }
    valid.update(kwargs)
    with pytest.raises(ValueError):
        reconstructed_equivalent_stress(0.001, **valid)


def test_negative_equivalent_strain_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        reconstructed_equivalent_stress(
            -1e-6,
            young_modulus_mpa=205_000.0,
            poisson_ratio=0.3,
            yield_stress_mpa=124.0,
            hardening_coefficient_mpa=380.0,
            hardening_exponent=0.245,
        )
