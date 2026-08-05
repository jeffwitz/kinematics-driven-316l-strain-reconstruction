import numpy as np
import pytest

from fem_inhouse.core.plane_stress_material import (
    ConstitutiveTrial,
    InPlaneConstitutiveTrial,
)
from fem_inhouse.spectral2d import (
    EBIPlaneStressElementBatch,
    EBISpectralSolverConfig,
    EBITwoTriangleKinematics2D,
    StructuredGrid2D,
    hookean_plane_stress_relative_error,
    pack_interior,
    solve_ebi_dirichlet_plane_stress,
    unpack_interior,
)
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig


class ElasticHookeanBatch:
    def __init__(self, point_count: int) -> None:
        self.point_count = point_count
        self.calls = 0
        self.commits = 0
        self.reverts = 0
        tangent = np.array([[4.0, 1.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 1.5]])
        self.elastic_tangent_in_plane_mpa = np.broadcast_to(tangent, (point_count, 3, 3)).copy()

    def evaluate_in_plane(
        self, in_plane_strain, *, time_increment, consistent_tangent=True
    ) -> InPlaneConstitutiveTrial:
        self.calls += 1
        strain = np.asarray(in_plane_strain, dtype=float)
        stress = np.einsum("pij,pj->pi", self.elastic_tangent_in_plane_mpa, strain.reshape(-1, 3))
        return InPlaneConstitutiveTrial(
            stress_in_plane_mpa=stress,
            tangent_in_plane_mpa=self.elastic_tangent_in_plane_mpa.copy(),
        )

    def complete_trial(self, trial: InPlaneConstitutiveTrial) -> ConstitutiveTrial:
        stress = np.asarray(trial.stress_in_plane_mpa)
        strain = np.linalg.solve(self.elastic_tangent_in_plane_mpa, stress[..., None])[..., 0]
        full_stress = np.zeros((self.point_count, 3, 3))
        full_stress[:, 0, 0] = stress[:, 0]
        full_stress[:, 1, 1] = stress[:, 1]
        full_stress[:, 0, 1] = full_stress[:, 1, 0] = stress[:, 2]
        full_strain = np.zeros_like(full_stress)
        full_strain[:, 0, 0] = strain[:, 0]
        full_strain[:, 1, 1] = strain[:, 1]
        full_strain[:, 0, 1] = full_strain[:, 1, 0] = 0.5 * strain[:, 2]
        return ConstitutiveTrial(
            stress_in_plane_mpa=stress,
            tangent_in_plane_mpa=trial.tangent_in_plane_mpa,
            full_stress_tensor_mpa=full_stress,
            full_strain_tensor=full_strain,
            elastic_strain_tensor=full_strain,
            plastic_strain_tensor=np.zeros_like(full_strain),
            plane_stress_residual_mpa=np.zeros((self.point_count, 3)),
        )

    def commit(self) -> None:
        self.commits += 1

    def revert(self) -> None:
        self.reverts += 1


def test_ebi_uses_two_samples_and_one_material_state_per_pixel() -> None:
    grid = StructuredGrid2D(4, 3, 2.0, 1.5)
    kinematics = EBITwoTriangleKinematics2D(grid)
    assert kinematics.kinematic_samples_per_pixel == 2
    assert kinematics.material_states_per_pixel == 1
    assert kinematics.material_point_count == 12
    assert kinematics.strain_samples(np.zeros((*grid.node_shape, 2))).shape == (4, 3, 2, 3)


def test_ebi_reconstruction_equals_traditional_two_triangle_elasticity() -> None:
    material = ElasticHookeanBatch(6)
    elements = EBIPlaneStressElementBatch(material, (3, 2))
    samples = np.random.default_rng(3).normal(size=(3, 2, 2, 3))
    trial = elements.evaluate_samples(samples, time_increment=1.0, consistent_tangent=True)
    traditional = np.einsum("xyij,xyqj->xyqi", trial.elastic_tangent_in_plane_mpa, samples)
    np.testing.assert_allclose(trial.sample_stress_mpa, traditional)
    np.testing.assert_allclose(trial.sample_stress_mpa.mean(axis=2), trial.mean_stress_mpa)
    assert material.calls == 1


def test_ebi_tangent_action_passes_directional_derivative() -> None:
    grid = StructuredGrid2D(4, 4, 2.0, 2.0)
    material = ElasticHookeanBatch(16)
    elements = EBIPlaneStressElementBatch(material, grid.pixel_shape)
    kinematics = EBITwoTriangleKinematics2D(grid)
    rng = np.random.default_rng(7)
    displacement = np.zeros((*grid.node_shape, 2))
    displacement[1:-1, 1:-1] = rng.normal(size=(3, 3, 2))
    direction = np.zeros_like(displacement)
    direction[1:-1, 1:-1] = rng.normal(size=(3, 3, 2))
    trial = elements.evaluate_samples(
        kinematics.strain_samples(displacement),
        time_increment=1.0,
        consistent_tangent=True,
    )
    analytical = elements.tangent_action(direction, kinematics=kinematics, trial=trial)
    calls_after_tangent = material.calls
    step = 1.0e-7
    perturbed = elements.evaluate_samples(
        kinematics.strain_samples(displacement + step * direction),
        time_increment=1.0,
        consistent_tangent=True,
    )
    base_residual = kinematics.divergence_from_sample_stress(trial.sample_stress_mpa)
    perturbed_residual = kinematics.divergence_from_sample_stress(perturbed.sample_stress_mpa)
    numerical = (perturbed_residual - base_residual) / step
    error = np.linalg.norm(numerical - analytical) / max(np.linalg.norm(analytical), 1.0)
    assert error < 1.0e-8
    assert calls_after_tangent == 1


@pytest.mark.parametrize("backend", ["scipy", "fftw"])
def test_ebi_newton_gmres_converges_non_affine_elastic_case(backend: str) -> None:
    if backend == "fftw":
        pytest.importorskip("pyfftw")
    grid = StructuredGrid2D(4, 4, 2.0, 2.0)
    x, y = grid.coordinates
    boundary = np.zeros((2, *grid.node_shape, 2))
    boundary[1, ..., 0] = 0.1 * x[:, None] + 0.01 * np.sin(np.pi * y[None, :] / 2.0)
    boundary[1, ..., 1] = 0.1 * y[None, :] + 0.01 * np.sin(np.pi * x[:, None] / 2.0)
    material = ElasticHookeanBatch(16)
    result = solve_ebi_dirichlet_plane_stress(
        grid=grid,
        material=material,
        boundary_displacement_history=boundary,
        config=EBISpectralSolverConfig(
            relative_equilibrium_tolerance=1.0e-10,
            transform=SpectralTransformConfig(
                backend=backend, fftw_planner_effort="estimate", fftw_use_wisdom=False
            ),
        ),
    )
    assert result.diagnostics.dimensionless_equilibrium_history[-1] < 1.0e-10
    assert result.diagnostics.material_points == 16
    assert result.strain_in_plane.shape == (4, 4, 2, 3)


def test_pack_unpack_preserves_only_interior_unknowns() -> None:
    grid = StructuredGrid2D(4, 3, 2.0, 1.5)
    field = np.zeros((*grid.node_shape, 2))
    field[1:-1, 1:-1] = 2.0
    np.testing.assert_array_equal(unpack_interior(pack_interior(field), grid), field)


def test_ebi_transactions_exist_only_on_the_mean_material_state() -> None:
    material = ElasticHookeanBatch(4)
    elements = EBIPlaneStressElementBatch(material, (2, 2))
    trial = elements.evaluate_samples(
        np.zeros((2, 2, 2, 3)), time_increment=1.0, consistent_tangent=True
    )
    elements.complete_trial(trial)
    elements.revert()
    elements.commit()
    assert material.calls == 1
    assert material.reverts == 1
    assert material.commits == 1


def test_ebi_newton_solution_is_invariant_to_length_units() -> None:
    def solve(scale: float):
        grid = StructuredGrid2D(4, 4, 2.0 * scale, 2.0 * scale)
        x, y = grid.coordinates
        boundary = np.zeros((2, *grid.node_shape, 2))
        boundary[1, ..., 0] = 0.1 * x[:, None] + 0.01 * scale * np.sin(
            np.pi * y[None, :] / (2.0 * scale)
        )
        boundary[1, ..., 1] = 0.1 * y[None, :] + 0.01 * scale * np.sin(
            np.pi * x[:, None] / (2.0 * scale)
        )
        return solve_ebi_dirichlet_plane_stress(
            grid=grid,
            material=ElasticHookeanBatch(16),
            boundary_displacement_history=boundary,
            config=EBISpectralSolverConfig(relative_equilibrium_tolerance=1.0e-10),
        )

    millimetres = solve(1.0)
    micrometres = solve(1.0e3)
    np.testing.assert_allclose(
        micrometres.displacement / 1.0e3,
        millimetres.displacement,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        micrometres.stress_in_plane_mpa,
        millimetres.stress_in_plane_mpa,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert (
        micrometres.diagnostics.iterations_per_increment
        == millimetres.diagnostics.iterations_per_increment
    )


def test_hookean_relation_uses_elastic_not_total_strain() -> None:
    material = ElasticHookeanBatch(1)
    trial = material.evaluate_in_plane([[0.1, -0.02, 0.03]], time_increment=1.0)
    completed = material.complete_trial(trial)
    assert (
        hookean_plane_stress_relative_error(completed, material.elastic_tangent_in_plane_mpa)
        < 1.0e-12
    )
