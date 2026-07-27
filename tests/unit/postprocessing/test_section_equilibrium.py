from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.postprocessing.section_equilibrium import integrated_section_equilibrium


def test_uniform_axial_stress_has_constant_section_force() -> None:
    stress = np.zeros((7, 9, 3), dtype=np.float64)
    stress[..., 1] = 125.0

    result = integrated_section_equilibrium(
        stress,
        spacing_x_mm=0.2,
        spacing_y_mm=0.3,
        thickness_mm=2.0,
    )

    np.testing.assert_allclose(result.section_force_n, 350.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.interval_balance_residual_n, 0.0, rtol=0.0, atol=0.0)
    assert result.section_force_relative_dispersion == 0.0
    assert result.balance_residual_relative_l2 == 0.0
    assert result.balance_residual_relative_to_mean_force == 0.0


def test_lateral_shear_flux_balances_varying_section_force() -> None:
    nx, ny = 8, 11
    hx, hy, thickness = 0.1, 0.25, 1.7
    slope_n_per_mm = 4.2
    stress = np.zeros((nx, ny, 3), dtype=np.float64)
    section_force = 20.0 + slope_n_per_mm * hy * np.arange(ny)
    stress[..., 1] = section_force[np.newaxis, :] / (thickness * hx * nx)
    stress[-1, :, 2] = -slope_n_per_mm / thickness

    result = integrated_section_equilibrium(
        stress,
        spacing_x_mm=hx,
        spacing_y_mm=hy,
        thickness_mm=thickness,
    )

    np.testing.assert_allclose(
        result.interval_balance_residual_n,
        0.0,
        rtol=0.0,
        atol=2.0e-14,
    )
    assert result.section_force_relative_dispersion > 0.0
    assert result.balance_residual_relative_l2 < 1.0e-14
    assert result.boundary_flux_closure_gain == pytest.approx(1.0, abs=1.0e-14)


def test_result_scales_with_thickness_but_relative_metrics_do_not() -> None:
    rng = np.random.default_rng(4)
    stress = rng.normal(size=(5, 6, 3))
    thin = integrated_section_equilibrium(
        stress,
        spacing_x_mm=0.2,
        spacing_y_mm=0.3,
        thickness_mm=1.0,
    )
    thick = integrated_section_equilibrium(
        stress,
        spacing_x_mm=0.2,
        spacing_y_mm=0.3,
        thickness_mm=2.0,
    )

    np.testing.assert_allclose(thick.section_force_n, 2.0 * thin.section_force_n)
    np.testing.assert_allclose(
        thick.interval_balance_residual_n,
        2.0 * thin.interval_balance_residual_n,
    )
    assert thick.section_force_relative_dispersion == pytest.approx(
        thin.section_force_relative_dispersion
    )
    assert thick.balance_residual_relative_l2 == pytest.approx(
        thin.balance_residual_relative_l2
    )
    assert thick.balance_residual_relative_to_mean_force == pytest.approx(
        thin.balance_residual_relative_to_mean_force
    )
    assert thick.boundary_flux_closure_gain == pytest.approx(thin.boundary_flux_closure_gain)


@pytest.mark.parametrize(
    ("stress", "message"),
    [
        (np.zeros((3, 4)), "shape"),
        (np.zeros((1, 4, 3)), "at least two"),
        (np.full((3, 4, 3), np.nan), "non-finite"),
    ],
)
def test_invalid_stress_is_rejected(stress: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        integrated_section_equilibrium(
            stress,
            spacing_x_mm=0.1,
            spacing_y_mm=0.1,
            thickness_mm=1.0,
        )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("spacing_x_mm", 0.0),
        ("spacing_y_mm", -1.0),
        ("thickness_mm", np.inf),
    ],
)
def test_invalid_geometry_is_rejected(keyword: str, value: float) -> None:
    arguments = {
        "spacing_x_mm": 0.1,
        "spacing_y_mm": 0.1,
        "thickness_mm": 1.0,
    }
    arguments[keyword] = value
    with pytest.raises(ValueError, match=keyword):
        integrated_section_equilibrium(np.zeros((3, 4, 3)), **arguments)
