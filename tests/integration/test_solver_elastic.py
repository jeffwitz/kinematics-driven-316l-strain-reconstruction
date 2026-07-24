import logging

import numpy as np

from fem_inhouse.config import CaseStudyConfig, MeshConfig, SolverConfig
from fem_inhouse.solver import run_case_study


def test_affine_elastic_field_is_reproduced_with_typed_api(caplog) -> None:
    mesh = MeshConfig(nx=4, ny=4)
    config = CaseStudyConfig(
        mesh,
        solver=SolverConfig(
            increments=2,
            residual_tolerance=1e-9,
            hardening_mode="ludwik",
            constitutive_backend="python",
        ),
    )
    x = np.linspace(0.0, mesh.physical_size_mm[0], mesh.nx + 1)
    y = np.linspace(0.0, mesh.physical_size_mm[1], mesh.ny + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    strain_xx = 1e-4

    with caplog.at_level(logging.INFO, logger="fem_inhouse.core.nonlinear"):
        result = run_case_study(
            config,
            displacement_x_mm=strain_xx * xx,
            displacement_y_mm=np.zeros_like(yy),
            yield_stress_mpa=np.full((mesh.nx, mesh.ny), 250.0),
            hardening_coefficient_mpa=np.full((mesh.nx, mesh.ny), 500.0),
            verbose=True,
        )

    expected_s11 = (
        config.material.young_modulus_mpa * strain_xx / (1.0 - config.material.poisson_ratio**2)
    )
    expected_s22 = config.material.poisson_ratio * expected_s11
    np.testing.assert_allclose(result.displacement_mm[..., 0], strain_xx * xx, atol=1e-15)
    np.testing.assert_allclose(result.displacement_mm[..., 1], 0.0, atol=1e-15)
    np.testing.assert_allclose(result.total_strain[..., 0], strain_xx, rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(result.stress_mpa[..., 0], expected_s11, rtol=1e-11)
    np.testing.assert_allclose(result.stress_mpa[..., 1], expected_s22, rtol=1e-11)
    np.testing.assert_allclose(result.equivalent_plastic_strain, 0.0)
    expected_horizontal_reaction = expected_s11 * mesh.physical_size_mm[1]
    expected_vertical_reaction = expected_s22 * mesh.physical_size_mm[0]
    np.testing.assert_allclose(
        result.reaction_force[0, :, 0].sum(),
        -expected_horizontal_reaction,
        rtol=1e-11,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.reaction_force[-1, :, 0].sum(),
        expected_horizontal_reaction,
        rtol=1e-11,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.reaction_force[:, 0, 1].sum(),
        -expected_vertical_reaction,
        rtol=1e-11,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.reaction_force[:, -1, 1].sum(),
        expected_vertical_reaction,
        rtol=1e-11,
        atol=1e-14,
    )
    events = [getattr(record, "event", None) for record in caplog.records]
    assert events[0] == "nonlinear_solve_started"
    assert "newton_iteration" in events
    assert events[-1] == "nonlinear_solve_completed"


def test_affine_shear_patch_reproduces_engineering_shear_convention() -> None:
    mesh = MeshConfig(nx=4, ny=3)
    config = CaseStudyConfig(
        mesh,
        solver=SolverConfig(
            increments=2,
            residual_tolerance=1e-9,
            hardening_mode="ludwik",
            constitutive_backend="python",
        ),
    )
    x = np.linspace(0.0, mesh.physical_size_mm[0], mesh.nx + 1)
    y = np.linspace(0.0, mesh.physical_size_mm[1], mesh.ny + 1)
    _xx, yy = np.meshgrid(x, y, indexing="ij")
    engineering_shear = 2e-4

    result = run_case_study(
        config,
        displacement_x_mm=engineering_shear * yy,
        displacement_y_mm=np.zeros_like(yy),
        yield_stress_mpa=np.full((mesh.nx, mesh.ny), 250.0),
        hardening_coefficient_mpa=np.full((mesh.nx, mesh.ny), 500.0),
    )

    shear_modulus = config.material.young_modulus_mpa / (
        2.0 * (1.0 + config.material.poisson_ratio)
    )
    np.testing.assert_allclose(
        result.displacement_mm[..., 0],
        engineering_shear * yy,
        atol=1e-15,
    )
    np.testing.assert_allclose(result.displacement_mm[..., 1], 0.0, atol=1e-15)
    np.testing.assert_allclose(result.total_strain[..., :2], 0.0, atol=1e-14)
    np.testing.assert_allclose(
        result.total_strain[..., 2],
        engineering_shear,
        rtol=1e-11,
    )
    np.testing.assert_allclose(result.stress_mpa[..., :2], 0.0, atol=1e-10)
    np.testing.assert_allclose(
        result.stress_mpa[..., 2],
        shear_modulus * engineering_shear,
        rtol=1e-11,
    )
    np.testing.assert_allclose(result.equivalent_plastic_strain, 0.0)
