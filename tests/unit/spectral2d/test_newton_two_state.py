import numpy as np
import pytest

from fem_inhouse.core.plane_stress_material import ConstitutiveTrial, InPlaneConstitutiveTrial
from fem_inhouse.spectral2d import (
    AdaptiveStepConfig,
    EBISpectralSolverConfig,
    EBITwoTriangleKinematics2D,
    StructuredGrid2D,
)
from fem_inhouse.spectral2d.newton_two_state import (
    AcceptedTwoStateTrialCache,
    TraditionalTwoStateTriangleBatch,
    TwoStateJacobianWorkspace,
    pack_interior,
    solve_two_state_dirichlet_plane_stress,
)
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig
from scripts.qualify_ebi_state_sharing import moment, side_resultants


class NonlinearStateBatch:
    def __init__(self, point_count: int) -> None:
        self.point_count = point_count
        self.calls = 0
        self.commits = 0
        self.reverts = 0
        self._has_trial = False
        self.committed_stress = None
        self._stiffness = np.broadcast_to(
            np.array([[4.0, 0.5, 0.0], [0.5, 3.0, 0.0], [0.0, 0.0, 1.5]]),
            (point_count, 3, 3),
        ).copy()

    def evaluate_in_plane(self, strain, *, time_increment, consistent_tangent=True):
        self.calls += 1
        self._has_trial = True
        values = np.asarray(strain, dtype=float).reshape(-1, 3)
        stress = np.einsum("pij,pj->pi", self._stiffness, values) + 0.2 * values**2
        self._last_stress = stress.copy()
        tangent = self._stiffness + np.einsum("pi,ij->pij", 0.4 * values, np.eye(3))
        return InPlaneConstitutiveTrial(stress_in_plane_mpa=stress, tangent_in_plane_mpa=tangent)

    def complete_trial(self, trial):
        stress = np.asarray(trial.stress_in_plane_mpa)
        full_stress = np.zeros((self.point_count, 3, 3))
        full_stress[:, 0, 0] = stress[:, 0]
        full_stress[:, 1, 1] = stress[:, 1]
        full_stress[:, 0, 1] = full_stress[:, 1, 0] = stress[:, 2]
        strain = np.zeros_like(full_stress)
        return ConstitutiveTrial(
            stress_in_plane_mpa=stress,
            tangent_in_plane_mpa=trial.tangent_in_plane_mpa,
            full_stress_tensor_mpa=full_stress,
            full_strain_tensor=strain,
            elastic_strain_tensor=strain,
            plastic_strain_tensor=strain,
            plane_stress_residual_mpa=np.zeros((self.point_count, 3)),
            observables={"accumulated_slip": np.zeros(self.point_count)},
        )

    def commit(self):
        if not self._has_trial:
            raise RuntimeError("commit requires an active trial")
        self.commits += 1
        self.committed_stress = self._last_stress.copy()
        self._has_trial = False

    def revert(self):
        self.reverts += 1
        self._has_trial = False


def test_two_state_nonlinear_directional_tangent() -> None:
    grid = StructuredGrid2D(2, 2, 1.0, 1.0)
    material = NonlinearStateBatch(8)
    elements = TraditionalTwoStateTriangleBatch(material, grid.pixel_shape)
    kinematics = EBITwoTriangleKinematics2D(grid)
    displacement = np.zeros((*grid.node_shape, 2))
    displacement[1, 1] = (0.1, -0.06)
    direction = np.zeros_like(displacement)
    direction[1, 1] = (0.7, -0.4)
    trial = elements.evaluate_samples(kinematics.strain_samples(displacement), time_increment=0.1)
    analytical = elements.tangent_action(direction, kinematics=kinematics, trial=trial)
    step = 1.0e-7
    perturbed = elements.evaluate_samples(
        kinematics.strain_samples(displacement + step * direction), time_increment=0.1
    )
    base = kinematics.divergence_from_sample_stress(trial.sample_stress_mpa)
    updated = kinematics.divergence_from_sample_stress(perturbed.sample_stress_mpa)
    error = np.linalg.norm((updated - base) / step - analytical) / max(
        np.linalg.norm(analytical), 1.0
    )
    assert error < 1.0e-7
    assert material.calls == 2


def test_two_state_trial_revert_commit_is_transactional() -> None:
    grid = StructuredGrid2D(2, 2, 1.0, 1.0)
    material = NonlinearStateBatch(8)
    elements = TraditionalTwoStateTriangleBatch(material, grid.pixel_shape)
    trial_a = elements.evaluate_samples(np.zeros((2, 2, 2, 3)), time_increment=0.1)
    elements.complete_trial(trial_a)
    elements.revert()
    with pytest.raises(RuntimeError):
        elements.commit()
    trial_b = elements.evaluate_samples(np.full((2, 2, 2, 3), 0.1), time_increment=0.1)
    elements.complete_trial(trial_b)
    elements.commit()
    assert material.reverts == 1
    assert material.commits == 1
    np.testing.assert_allclose(
        material.committed_stress, trial_b.material_trial.stress_in_plane_mpa
    )


def test_accepted_trial_cache_owns_mutable_residual_and_is_single_use() -> None:
    grid = StructuredGrid2D(2, 2, 1.0, 1.0)
    material = NonlinearStateBatch(8)
    elements = TraditionalTwoStateTriangleBatch(material, grid.pixel_shape)
    sample_strain = np.zeros((*grid.pixel_shape, 2, 3))
    trial = elements.evaluate_samples(sample_strain, time_increment=0.1)
    source_residual = np.ones((*grid.node_shape, 2))
    cache = AcceptedTwoStateTrialCache()

    cache.store(
        trial=trial,
        sample_strain=sample_strain,
        residual=source_residual,
        relative=0.25,
        absolute=1.5,
    )
    source_residual[...] = 99.0
    sample_strain[...] = 77.0

    cached_trial, cached_strain, cached_residual, relative, absolute = cache.take()
    assert cached_trial is trial
    np.testing.assert_allclose(cached_strain, 0.0)
    np.testing.assert_allclose(cached_residual, 1.0)
    assert relative == 0.25
    assert absolute == 1.5
    assert not cache.populated
    with pytest.raises(RuntimeError, match="cache is empty"):
        cache.take()



def test_side_resultants_and_moment_use_nodal_coordinates() -> None:
    grid = StructuredGrid2D(2, 2, 2.0, 3.0)
    reaction = np.zeros((*grid.node_shape, 2))
    reaction[-1, :, 1] = 2.0
    reaction[:, 0, 0] = 1.0
    sides = side_resultants(reaction)
    np.testing.assert_allclose(sides[1], (1.0, 6.0))
    np.testing.assert_allclose(sides[2], (3.0, 2.0))
    assert moment(reaction, grid) == 12.0


def test_two_state_solver_backend_equivalence() -> None:
    pytest.importorskip("pyfftw")
    grid = StructuredGrid2D(4, 4, 2.0, 2.0)
    x, y = grid.coordinates
    boundary = np.zeros((3, *grid.node_shape, 2))
    boundary[1:, ..., 0] = 0.04 * x[:, None]
    boundary[1:, ..., 1] = 0.03 * y[None, :]
    results = {}
    for backend in ("scipy", "fftw"):
        results[backend] = solve_two_state_dirichlet_plane_stress(
            grid=grid,
            material=NonlinearStateBatch(32),
            boundary_displacement_history=boundary,
            config=EBISpectralSolverConfig(
                relative_equilibrium_tolerance=1.0e-10,
                transform=SpectralTransformConfig(
                    backend=backend, fftw_planner_effort="estimate", fftw_use_wisdom=False
                ),
            ),
        )
        assert results[backend].diagnostics.dimensionless_equilibrium_history[-1] < 1.0e-10
    np.testing.assert_allclose(
        results["fftw"].displacement, results["scipy"].displacement, rtol=1.0e-11, atol=1.0e-12
    )
    np.testing.assert_allclose(
        results["fftw"].stress_in_plane_mpa,
        results["scipy"].stress_in_plane_mpa,
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_two_state_solver_archives_linear_cost_breakdown() -> None:
    grid = StructuredGrid2D(4, 4, 2.0, 2.0)
    x, y = grid.coordinates
    boundary = np.zeros((3, *grid.node_shape, 2))
    boundary[1, ..., 0] = 0.02 * x[:, None] + 0.003 * y[None, :] ** 2
    boundary[2, ..., 1] = 0.03 * y[None, :] + 0.002 * x[:, None] ** 2
    result = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=NonlinearStateBatch(32),
        boundary_displacement_history=boundary,
        config=EBISpectralSolverConfig(
            relative_equilibrium_tolerance=1.0e-10,
            transform=SpectralTransformConfig(
                backend="scipy", fftw_planner_effort="estimate"
            ),
        ),
    )
    diagnostics = result.diagnostics
    assert diagnostics.linear_solves
    assert diagnostics.provenance["gmres_restart"] == 50
    assert all(entry.gmres_info == 0 for entry in diagnostics.linear_solves)
    assert all(entry.gmres_iterations > 0 for entry in diagnostics.linear_solves)
    assert all(entry.jacobian_calls > 0 for entry in diagnostics.linear_solves)
    assert all(entry.preconditioner_calls > 0 for entry in diagnostics.linear_solves)
    assert all(entry.krylov_overhead_seconds >= 0.0 for entry in diagnostics.linear_solves)
    gmres_total = sum(entry.gmres_seconds for entry in diagnostics.linear_solves)
    jacobian_total = sum(entry.jacobian_seconds for entry in diagnostics.linear_solves)
    preconditioner_total = sum(
        entry.preconditioner_seconds for entry in diagnostics.linear_solves
    )
    assert diagnostics.timings["gmres_seconds"] == pytest.approx(gmres_total)
    assert diagnostics.timings["jacobian_seconds"] == pytest.approx(jacobian_total)
    assert diagnostics.timings["preconditioner_seconds"] == pytest.approx(preconditioner_total)
    assert diagnostics.timings["jacobian_calls"] == sum(
        entry.jacobian_calls for entry in diagnostics.linear_solves
    )
    assert diagnostics.timings["preconditioner_calls"] == sum(
        entry.preconditioner_calls for entry in diagnostics.linear_solves
    )


def test_two_state_complete_trial_promotion_preserves_solution() -> None:
    grid = StructuredGrid2D(4, 4, 2.0, 2.0)
    x, y = grid.coordinates
    boundary = np.zeros((3, *grid.node_shape, 2))
    boundary[1:, ..., 0] = 0.02 * x[:, None]
    boundary[1:, ..., 1] = 0.03 * y[None, :]
    common = dict(
        relative_equilibrium_tolerance=1.0e-10,
        transform=SpectralTransformConfig(backend="scipy"),
    )
    qualification_material = NonlinearStateBatch(32)
    production_material = NonlinearStateBatch(32)
    qualification = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=qualification_material,
        boundary_displacement_history=boundary,
        config=EBISpectralSolverConfig(**common, verify_final_state=True),
    )
    production = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=production_material,
        boundary_displacement_history=boundary,
        config=EBISpectralSolverConfig(**common, verify_final_state=False),
    )
    np.testing.assert_allclose(production.displacement, qualification.displacement)
    np.testing.assert_allclose(production.stress_in_plane_mpa, qualification.stress_in_plane_mpa)
    np.testing.assert_allclose(production.reaction_forces, qualification.reaction_forces)
    assert production_material.calls < qualification_material.calls
    assert production.diagnostics.verification_residual <= 1.0e-10


def test_two_state_adaptive_path_reaches_same_proportional_solution() -> None:
    grid = StructuredGrid2D(4, 4, 2.0, 2.0)
    x, y = grid.coordinates
    boundary = np.zeros((3, *grid.node_shape, 2))
    boundary[1:, ..., 0] = 0.02 * x[:, None]
    boundary[1:, ..., 1] = 0.03 * y[None, :]
    common = dict(
        relative_equilibrium_tolerance=1.0e-10,
        transform=SpectralTransformConfig(backend="scipy"),
        verify_final_state=False,
    )
    fixed = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=NonlinearStateBatch(32),
        boundary_displacement_history=boundary,
        config=EBISpectralSolverConfig(**common),
    )
    adaptive = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=NonlinearStateBatch(32),
        boundary_displacement_history=boundary,
        config=EBISpectralSolverConfig(
            **common,
            adaptive_stepping_enabled=True,
            adaptive_step=AdaptiveStepConfig(
                initial_increment_fraction=0.5,
                minimum_increment_fraction=0.125,
                maximum_increment_fraction=0.5,
            ),
        ),
    )
    np.testing.assert_allclose(adaptive.displacement, fixed.displacement, rtol=1.0e-8)
    np.testing.assert_allclose(
        adaptive.stress_in_plane_mpa,
        fixed.stress_in_plane_mpa,
        rtol=1.0e-8,
    )
    assert adaptive.diagnostics.adaptive_stepping_enabled
    assert len(adaptive.diagnostics.adaptive_step_history) == 2
    assert all(item["accepted"] for item in adaptive.diagnostics.adaptive_step_history)


@pytest.mark.parametrize("mesh_size", (4, 12, 50, 100))
def test_two_state_inplace_jacobian_matches_reference(mesh_size: int) -> None:
    grid = StructuredGrid2D(mesh_size, mesh_size, 2.0, 2.0)
    elements = TraditionalTwoStateTriangleBatch(
        NonlinearStateBatch(2 * mesh_size * mesh_size), grid.pixel_shape
    )
    kinematics = EBITwoTriangleKinematics2D(grid)
    rng = np.random.default_rng(mesh_size)
    displacement = rng.normal(size=(*grid.node_shape, 2))
    trial = elements.evaluate_samples(kinematics.strain_samples(displacement), time_increment=0.1)
    workspace = TwoStateJacobianWorkspace.create(grid)
    for _ in range(20):
        vector = rng.normal(size=2 * (mesh_size - 1) ** 2)
        field = np.zeros((*grid.node_shape, 2))
        field[1:-1, 1:-1, :] = vector.reshape(mesh_size - 1, mesh_size - 1, 2)
        reference = elements.tangent_action(field, kinematics=kinematics, trial=trial)
        workspace.nodal_increment[...] = field
        inplace = elements.tangent_action_into(
            kinematics=kinematics,
            trial=trial,
            workspace=workspace,
            kernel="einsum",
        )
        np.testing.assert_allclose(
            inplace,
            pack_interior(reference),
            rtol=1.0e-13,
            atol=1.0e-13 * max(float(np.linalg.norm(reference)), 1.0),
        )
        explicit = elements.tangent_action_into(
            kinematics=kinematics,
            trial=trial,
            workspace=workspace,
            kernel="explicit",
        )
        np.testing.assert_allclose(
            explicit,
            pack_interior(reference),
            rtol=1.0e-13,
            atol=1.0e-13 * max(float(np.linalg.norm(reference)), 1.0),
        )


def test_two_state_inexact_newton_records_forcing_and_preserves_solution() -> None:
    grid = StructuredGrid2D(4, 4, 2.0, 2.0)
    x, y = grid.coordinates
    boundary = np.zeros((3, *grid.node_shape, 2))
    boundary[1, ..., 0] = 0.02 * x[:, None] + 0.003 * y[None, :] ** 2
    boundary[2, ..., 1] = 0.03 * y[None, :] + 0.002 * x[:, None] ** 2
    common = dict(
        relative_equilibrium_tolerance=1.0e-10,
        transform=SpectralTransformConfig(backend="scipy"),
    )
    fixed = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=NonlinearStateBatch(32),
        boundary_displacement_history=boundary,
        config=EBISpectralSolverConfig(**common),
    )
    inexact = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=NonlinearStateBatch(32),
        boundary_displacement_history=boundary,
        config=EBISpectralSolverConfig(
            **common,
            linear_tolerance_mode="eisenstat_walker",
            verify_linear_residual=True,
        ),
    )
    assert inexact.diagnostics.verification_residual <= 1.0e-10
    assert any(
        entry.requested_relative_tolerance < 1.0e-3
        for entry in inexact.diagnostics.linear_solves
    )
    assert all(
        entry.linear_residual_ratio is not None
        for entry in inexact.diagnostics.linear_solves
    )
    np.testing.assert_allclose(
        inexact.displacement,
        fixed.displacement,
        rtol=1.0e-8,
        atol=1.0e-10,
    )


def test_two_state_reference_updates_reuse_green_plan_and_archive_changes() -> None:
    grid = StructuredGrid2D(4, 4, 2.0, 2.0)
    x, y = grid.coordinates
    boundary = np.zeros((3, *grid.node_shape, 2))
    boundary[1, ..., 0] = 0.02 * x[:, None] + 0.003 * y[None, :] ** 2
    boundary[2, ..., 1] = 0.03 * y[None, :] + 0.002 * x[:, None] ** 2
    result = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=NonlinearStateBatch(32),
        boundary_displacement_history=boundary,
        config=EBISpectralSolverConfig(
            relative_equilibrium_tolerance=1.0e-10,
            reference_update_mode="per_newton",
            reference_minimum_relative_change=0.0,
            transform=SpectralTransformConfig(backend="scipy"),
        ),
    )
    assert result.diagnostics.verification_residual <= 1.0e-10
    assert result.diagnostics.reference_updates
    assert any(update["accepted"] for update in result.diagnostics.reference_updates)
    assert result.diagnostics.transform_planning_seconds >= 0.0
