import numpy as np

from fem_inhouse.core.plane_stress_material import ConstitutiveTrial, InPlaneConstitutiveTrial
from fem_inhouse.spectral2d import (
    QUAD1_2D,
    Spectral2DConfig,
    StructuredGrid2D,
    solve_dirichlet_plane_stress_spectral,
)


class ElasticMaterial:
    def __init__(self, point_count: int) -> None:
        self.point_count = point_count
        self._tangent = np.array([[4.0, 1.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 1.0]])

    def evaluate_in_plane(self, in_plane_strain, *, time_increment, consistent_tangent=True):
        strain = np.asarray(in_plane_strain, dtype=float)
        return InPlaneConstitutiveTrial(
            stress_in_plane_mpa=np.einsum("ab,...b->...a", self._tangent, strain),
            tangent_in_plane_mpa=np.broadcast_to(self._tangent, (*strain.shape[:-1], 3, 3)),
        )

    def complete_trial(self, trial):
        shape = trial.stress_in_plane_mpa.shape[:-1]
        zeros = np.zeros((*shape, 3, 3))
        return ConstitutiveTrial(
            stress_in_plane_mpa=trial.stress_in_plane_mpa,
            tangent_in_plane_mpa=trial.tangent_in_plane_mpa,
            full_stress_tensor_mpa=zeros,
            full_strain_tensor=zeros,
            elastic_strain_tensor=zeros,
            plastic_strain_tensor=zeros,
            plane_stress_residual_mpa=np.zeros((*shape, 3)),
        )

    def commit(self):
        pass

    def revert(self):
        pass


def test_homogeneous_affine_elastic_field_converges_without_fluctuation() -> None:
    grid = StructuredGrid2D(4, 3, 2.0, 1.5)
    x, y = grid.coordinates
    boundary = np.zeros((2, *grid.node_shape, 2))
    boundary[1, ..., 0] = 0.2 * x[:, None] + 0.1 * y[None, :]
    boundary[1, ..., 1] = -0.1 * x[:, None] + 0.3 * y[None, :]
    result = solve_dirichlet_plane_stress_spectral(
        grid=grid,
        material=ElasticMaterial(QUAD1_2D(grid).material_point_count),
        boundary_displacement_history=boundary,
        config=Spectral2DConfig(anderson_enabled=False),
    )
    np.testing.assert_allclose(result.fluctuation_displacement, 0.0, atol=1.0e-12)
    assert result.diagnostics.iterations_per_increment == (1,)
    assert result.diagnostics.absolute_residual_history[-1] < 1.0e-12


def test_non_affine_elastic_field_converges() -> None:
    grid = StructuredGrid2D(4, 4, 2.0, 2.0)
    x, y = grid.coordinates
    boundary = np.zeros((2, *grid.node_shape, 2))
    boundary[1, ..., 0] = 0.1 * x[:, None] + 0.01 * np.sin(np.pi * y[None, :] / 2.0)
    boundary[1, ..., 1] = 0.1 * y[None, :] + 0.01 * np.sin(np.pi * x[:, None] / 2.0)
    result = solve_dirichlet_plane_stress_spectral(
        grid=grid,
        material=ElasticMaterial(16),
        boundary_displacement_history=boundary,
        config=Spectral2DConfig(anderson_enabled=False, maximum_fixed_point_iterations=100),
    )
    assert result.diagnostics.absolute_residual_history[-1] < 1.0e-7
