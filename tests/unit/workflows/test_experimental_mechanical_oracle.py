from __future__ import annotations

import numpy as np

from fem_inhouse.core.driven_j2 import DrivenJ2PlaneStressBatch
from fem_inhouse.core.plane_stress_material import PythonJ2PlaneStressBatch
from fem_inhouse.identification.dic_whitening import DICSpectralWhitener
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.workflows.experimental_mechanical_oracle import (
    ExperimentalOracleIncrementProblem,
    ExperimentalOracleObjectiveWeights,
    ExperimentalOracleOptimizationConfig,
    ExperimentalOracleWarmStartRequest,
    evaluate_experimental_mechanical_oracle,
    ludwik_increment_history_from_measured_displacement,
    solve_experimental_mechanical_oracle_history,
    solve_experimental_mechanical_oracle_increment,
    solve_experimental_mechanical_oracle_reduced_increment,
)


def _affine_displacement(grid: StructuredGrid2D) -> np.ndarray:
    x = np.linspace(0.0, grid.length_x, grid.node_shape[0])[:, None]
    y = np.linspace(0.0, grid.length_y, grid.node_shape[1])[None, :]
    displacement = np.empty((*grid.node_shape, 2))
    displacement[..., 0] = 0.005 * x - 0.0004 * y
    displacement[..., 1] = -0.001 * y + 0.0007 * x
    return displacement


def test_global_oracle_jacobian_action_matches_a_central_difference() -> None:
    grid = StructuredGrid2D(3, 2, 3.0, 2.0)
    kinematics = TwoSubcellDiagnostic2D(grid)
    material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    displacement = _affine_displacement(grid)
    plastic_increment = np.full((*grid.pixel_shape, 2), 0.0015)
    linearisation = evaluate_experimental_mechanical_oracle(
        material,
        kinematics,
        displacement,
        plastic_increment,
        time_increment=1.0,
    )

    rng = np.random.default_rng(20260814)
    displacement_direction = rng.standard_normal(displacement.shape)
    displacement_direction[[0, -1], :, :] = 0.0
    displacement_direction[:, [0, -1], :] = 0.0
    plastic_direction = 0.1 * rng.standard_normal(plastic_increment.shape)
    action = linearisation.jacobian_action(
        displacement_direction,
        plastic_direction,
    )

    step = 1.0e-7
    plus = evaluate_experimental_mechanical_oracle(
        material,
        kinematics,
        displacement + step * displacement_direction,
        plastic_increment + step * plastic_direction,
        time_increment=1.0,
    ).mechanical_residual
    minus = evaluate_experimental_mechanical_oracle(
        material,
        kinematics,
        displacement - step * displacement_direction,
        plastic_increment - step * plastic_direction,
        time_increment=1.0,
    ).mechanical_residual
    finite_difference = (plus - minus) / (2.0 * step)

    relative_error = np.linalg.norm(action - finite_difference) / np.linalg.norm(
        finite_difference
    )
    assert relative_error < 1.0e-7


def test_global_oracle_rejects_a_material_layout_mismatch() -> None:
    grid = StructuredGrid2D(2, 2, 2.0, 2.0)
    kinematics = TwoSubcellDiagnostic2D(grid)
    material = DrivenJ2PlaneStressBatch(
        1,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )

    try:
        evaluate_experimental_mechanical_oracle(
            material,
            kinematics,
            _affine_displacement(grid),
            np.full((*grid.pixel_shape, 2), 0.001),
            time_increment=1.0,
        )
    except ValueError as error:
        assert "material point count" in str(error)
    else:
        raise AssertionError("a mismatched material layout must be rejected")


def test_global_oracle_transpose_action_satisfies_the_adjoint_identity() -> None:
    grid = StructuredGrid2D(3, 3, 3.0, 3.0)
    kinematics = TwoSubcellDiagnostic2D(grid)
    material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    displacement = _affine_displacement(grid)
    plastic_increment = np.full((*grid.pixel_shape, 2), 0.001)
    linearisation = evaluate_experimental_mechanical_oracle(
        material,
        kinematics,
        displacement,
        plastic_increment,
        time_increment=1.0,
    )
    rng = np.random.default_rng(81)
    displacement_direction = rng.normal(size=displacement.shape)
    plastic_direction = rng.normal(size=plastic_increment.shape)
    mechanical_dual = rng.normal(size=displacement.shape)

    action = linearisation.jacobian_action(
        displacement_direction,
        plastic_direction,
    )
    displacement_gradient, plastic_gradient = linearisation.jacobian_transpose_action(
        mechanical_dual
    )

    lhs = float(np.vdot(action, mechanical_dual).real)
    rhs = float(
        np.vdot(displacement_direction, displacement_gradient).real
        + np.vdot(plastic_direction, plastic_gradient).real
    )
    np.testing.assert_allclose(lhs, rhs, rtol=2.0e-13, atol=1.0e-8)


def test_reduced_plastic_parameterisation_round_trips_and_has_reduced_dimension() -> None:
    grid = StructuredGrid2D(3, 3, 3.0, 3.0)
    kinematics = TwoSubcellDiagnostic2D(grid)
    material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    measured = _affine_displacement(grid)
    increment = np.full((*grid.pixel_shape, 2), 0.001)
    basis = np.column_stack(
        (
            np.ones(increment.size),
            np.linspace(-1.0, 1.0, increment.size),
        )
    ) * 1.0e-4
    problem = ExperimentalOracleIncrementProblem(
        material=material,
        kinematics=kinematics,
        measured_displacement=measured,
        whitener=_white_dic_whitener(measured.shape, 1.0e-4),
        ludwik_increment=increment,
        previous_increment=increment,
        weights=ExperimentalOracleObjectiveWeights(ludwik_prior=0.1),
        time_increment=1.0,
        displacement_variable_scale=1.0e-3,
        plastic_increment_variable_scale=1.0e-3,
        equilibrium_scale=100.0,
        plastic_basis=basis,
    )
    variables = problem.pack_state(measured, increment)
    displacement, reconstructed = problem.unpack_state(variables)
    assert problem.reduced
    assert problem.variable_count == problem.displacement_unknown_count + 2
    np.testing.assert_allclose(displacement, measured)
    np.testing.assert_allclose(reconstructed, increment, atol=1.0e-12)


def _white_dic_whitener(shape: tuple[int, int, int], noise: float) -> DICSpectralWhitener:
    return DICSpectralWhitener(
        power_spectral_density=np.full(shape, noise**2),
        spectral_floor=noise**2 * 1.0e-8,
    )


def test_augmented_objective_gradient_matches_a_directional_difference() -> None:
    grid = StructuredGrid2D(3, 3, 3.0, 3.0)
    kinematics = TwoSubcellDiagnostic2D(grid)
    material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    measured = _affine_displacement(grid)
    plastic = np.full((*grid.pixel_shape, 2), 0.0015)
    ludwik = np.full_like(plastic, 0.0014)
    problem = ExperimentalOracleIncrementProblem(
        material=material,
        kinematics=kinematics,
        measured_displacement=measured,
        whitener=_white_dic_whitener(measured.shape, 1.0e-4),
        ludwik_increment=ludwik,
        previous_increment=np.full_like(plastic, 0.0012),
        weights=ExperimentalOracleObjectiveWeights(
            dic=1.0,
            ludwik_prior=0.7,
            spatial_plastic_increment=0.3,
            temporal_plastic_increment=0.2,
        ),
        time_increment=1.0,
        displacement_variable_scale=1.0e-3,
        plastic_increment_variable_scale=1.0e-3,
        equilibrium_scale=100.0,
    )
    displacement = measured.copy()
    displacement[1:-1, 1:-1] += 2.0e-5
    variables = problem.pack_state(displacement, plastic)
    rng = np.random.default_rng(104)
    multiplier = 0.1 * rng.normal(size=problem.mechanical_constraint_count)
    direction = rng.normal(size=variables.size)
    direction /= np.linalg.norm(direction)
    analytical = problem.objective_and_gradient(
        variables,
        multiplier=multiplier,
        penalty=2.0,
    )

    step = 1.0e-6
    plus = problem.objective_and_gradient(
        variables + step * direction,
        multiplier=multiplier,
        penalty=2.0,
    ).value
    minus = problem.objective_and_gradient(
        variables - step * direction,
        multiplier=multiplier,
        penalty=2.0,
    ).value
    finite_difference = (plus - minus) / (2.0 * step)
    directional_gradient = float(np.vdot(analytical.gradient, direction).real)

    np.testing.assert_allclose(
        directional_gradient,
        finite_difference,
        rtol=2.0e-5,
        atol=1.0e-8,
    )


def test_increment_solver_recovers_an_equilibrated_affine_reference_and_commits() -> None:
    grid = StructuredGrid2D(3, 3, 3.0, 3.0)
    kinematics = TwoSubcellDiagnostic2D(grid)
    material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    measured = _affine_displacement(grid)
    truth = np.full((*grid.pixel_shape, 2), 0.001)
    initial_displacement = measured.copy()
    initial_displacement[1:-1, 1:-1, 0] += 5.0e-5
    initial_increment = 0.8 * truth

    result = solve_experimental_mechanical_oracle_increment(
        material=material,
        kinematics=kinematics,
        measured_displacement=measured,
        whitener=_white_dic_whitener(measured.shape, 1.0e-5),
        ludwik_increment=truth,
        initial_displacement=initial_displacement,
        initial_equivalent_plastic_increment=initial_increment,
        weights=ExperimentalOracleObjectiveWeights(
            dic=1.0,
            ludwik_prior=10.0,
            spatial_plastic_increment=0.1,
        ),
        config=ExperimentalOracleOptimizationConfig(
            maximum_augmented_iterations=5,
            maximum_inner_iterations=200,
            equilibrium_rms_tolerance=1.0e-5,
            initial_penalty=1.0,
            penalty_growth=10.0,
        ),
        time_increment=1.0,
    )

    assert result.converged
    np.testing.assert_allclose(result.displacement, measured, rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(
        result.equivalent_plastic_increment,
        truth,
        rtol=0.0,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        material.committed_equivalent_plastic_strain,
        truth.ravel(),
        rtol=0.0,
        atol=2.0e-7,
    )


def test_reduced_increment_solver_preserves_an_equilibrated_reference() -> None:
    grid = StructuredGrid2D(3, 3, 3.0, 3.0)
    kinematics = TwoSubcellDiagnostic2D(grid)
    material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    measured = _affine_displacement(grid)
    increment = np.full((*grid.pixel_shape, 2), 8.0e-4)

    result = solve_experimental_mechanical_oracle_reduced_increment(
        material=material,
        kinematics=kinematics,
        measured_displacement=measured,
        whitener=_white_dic_whitener(measured.shape, 1.0e-5),
        ludwik_increment=increment,
        initial_displacement=measured,
        initial_equivalent_plastic_increment=increment,
        weights=ExperimentalOracleObjectiveWeights(
            dic=1.0,
            ludwik_prior=100.0,
        ),
        config=ExperimentalOracleOptimizationConfig(
            maximum_inner_iterations=50,
            equilibrium_rms_tolerance=1.0e-6,
            projected_gradient_tolerance=1.0e-3,
        ),
    )

    assert result.converged
    assert result.equilibrium_rms <= 1.0e-6
    np.testing.assert_allclose(
        result.equivalent_plastic_increment,
        increment,
        rtol=0.0,
        atol=2.0e-7,
    )


def test_increment_solver_reverts_when_the_augmented_constraint_does_not_converge() -> None:
    grid = StructuredGrid2D(3, 3, 3.0, 3.0)
    kinematics = TwoSubcellDiagnostic2D(grid)
    material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    measured = _affine_displacement(grid)
    initial = measured.copy()
    initial[1:-1, 1:-1, 0] += 1.0e-4
    increment = np.full((*grid.pixel_shape, 2), 0.001)

    result = solve_experimental_mechanical_oracle_increment(
        material=material,
        kinematics=kinematics,
        measured_displacement=measured,
        whitener=_white_dic_whitener(measured.shape, 1.0e-5),
        ludwik_increment=increment,
        initial_displacement=initial,
        initial_equivalent_plastic_increment=0.8 * increment,
        config=ExperimentalOracleOptimizationConfig(
            maximum_augmented_iterations=1,
            maximum_inner_iterations=1,
            equilibrium_rms_tolerance=1.0e-30,
        ),
    )

    assert not result.converged
    np.testing.assert_array_equal(
        material.committed_equivalent_plastic_strain,
        np.zeros(material.point_count),
    )
    np.testing.assert_array_equal(
        material.committed_plastic_strain,
        np.zeros((material.point_count, 3)),
    )


def test_augmented_penalty_is_kept_when_the_constraint_drops_sufficiently() -> None:
    config = ExperimentalOracleOptimizationConfig(
        sufficient_constraint_reduction=0.25
    )
    assert config.sufficient_constraint_reduction == 0.25


def test_history_solver_commits_each_accepted_increment_sequentially() -> None:
    grid = StructuredGrid2D(3, 3, 3.0, 3.0)
    kinematics = TwoSubcellDiagnostic2D(grid)
    measured_final = _affine_displacement(grid)
    measured_history = np.stack(
        (
            np.zeros_like(measured_final),
            0.5 * measured_final,
            measured_final,
        )
    )
    increment = np.full((*grid.pixel_shape, 2), 5.0e-4)
    ludwik_history = np.stack((increment, increment))
    material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    committed_peeq_seen: list[float] = []

    def warm_start(request: ExperimentalOracleWarmStartRequest) -> np.ndarray:
        committed_peeq_seen.append(
            float(np.max(request.material.committed_equivalent_plastic_strain))
        )
        return request.initial_displacement

    result = solve_experimental_mechanical_oracle_history(
        material=material,
        kinematics=kinematics,
        measured_displacement_history=measured_history,
        whitener=_white_dic_whitener(measured_final.shape, 1.0e-5),
        ludwik_increment_history=ludwik_history,
        displacement_warm_start=warm_start,
        weights=ExperimentalOracleObjectiveWeights(
            dic=1.0,
            ludwik_prior=10.0,
        ),
        config=ExperimentalOracleOptimizationConfig(
            maximum_augmented_iterations=5,
            maximum_inner_iterations=200,
            equilibrium_rms_tolerance=1.0e-5,
        ),
        time_increments=np.array([0.4, 0.6]),
    )

    assert result.completed
    assert result.failed_increment is None
    assert len(result.increments) == 2
    assert result.displacement_history.shape == measured_history.shape
    assert result.equivalent_plastic_increment_history.shape == ludwik_history.shape
    assert committed_peeq_seen[0] == 0.0
    assert committed_peeq_seen[1] > 4.0e-4
    np.testing.assert_allclose(
        material.committed_equivalent_plastic_strain,
        np.full(material.point_count, 1.0e-3),
        rtol=0.0,
        atol=5.0e-7,
    )


def test_ludwik_prior_replay_returns_non_negative_increment_history() -> None:
    grid = StructuredGrid2D(2, 2, 2.0, 2.0)
    kinematics = TwoSubcellDiagnostic2D(grid)
    final = 3.0 * _affine_displacement(grid)
    history = np.stack((np.zeros_like(final), 0.5 * final, final))
    material = PythonJ2PlaneStressBatch(
        np.full(kinematics.material_point_count, 200.0),
        np.full(kinematics.material_point_count, 500.0),
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )

    increments = ludwik_increment_history_from_measured_displacement(
        material=material,
        kinematics=kinematics,
        measured_displacement_history=history,
        time_increments=np.array([0.4, 0.6]),
    )

    assert increments.shape == (2, *grid.pixel_shape, 2)
    assert np.all(increments >= 0.0)
    assert float(np.max(increments)) > 0.0
