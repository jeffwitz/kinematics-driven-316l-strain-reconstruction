import numpy as np

from fem_inhouse.config import CaseStudyConfig, MeshConfig, SolverConfig
from fem_inhouse.solver import run_case_study


def test_affine_elastic_field_is_reproduced_with_typed_api() -> None:
    mesh = MeshConfig(nx=4, ny=4)
    config = CaseStudyConfig(
        mesh,
        solver=SolverConfig(increments=2, residual_tolerance=1e-9, hardening_mode="ludwik"),
    )
    x = np.linspace(0.0, mesh.physical_size_mm[0], mesh.nx + 1)
    y = np.linspace(0.0, mesh.physical_size_mm[1], mesh.ny + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    strain_xx = 1e-4

    result = run_case_study(
        config,
        displacement_x_mm=strain_xx * xx,
        displacement_y_mm=np.zeros_like(yy),
        yield_stress_mpa=np.full((mesh.nx, mesh.ny), 250.0),
        hardening_coefficient_mpa=np.full((mesh.nx, mesh.ny), 500.0),
    )

    expected_s11 = (
        config.material.young_modulus_mpa * strain_xx / (1.0 - config.material.poisson_ratio**2)
    )
    np.testing.assert_allclose(result.displacement_mm[..., 0], strain_xx * xx, atol=1e-15)
    np.testing.assert_allclose(result.displacement_mm[..., 1], 0.0, atol=1e-15)
    np.testing.assert_allclose(result.total_strain[..., 0], strain_xx, rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(result.stress_mpa[..., 0], expected_s11, rtol=1e-11)
    np.testing.assert_allclose(result.equivalent_plastic_strain, 0.0)
