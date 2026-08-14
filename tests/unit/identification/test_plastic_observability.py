from __future__ import annotations

import numpy as np

from fem_inhouse.core.driven_j2 import DrivenJ2PlaneStressBatch
from fem_inhouse.identification.dic_whitening import DICSpectralWhitener
from fem_inhouse.identification.plastic_observability import (
    PlasticMetric,
    PlasticObservabilityOperator,
    PlasticObservabilityState,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.workflows.experimental_mechanical_oracle import (
    evaluate_experimental_mechanical_oracle,
)


def _operator() -> PlasticObservabilityOperator:
    grid = StructuredGrid2D(3, 3, 3.0, 3.0)
    kinematics = TwoSubcellDiagnostic2D(grid)
    material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    displacement = np.zeros((*grid.node_shape, 2))
    displacement[..., 0] = np.linspace(0.0, 0.01, grid.node_shape[0])[:, None]
    displacement[..., 1] = np.linspace(0.0, -0.004, grid.node_shape[1])[None, :]
    plastic = np.full((*grid.pixel_shape, 2), 0.0015)
    linearisation = evaluate_experimental_mechanical_oracle(
        material,
        kinematics,
        displacement,
        plastic,
        time_increment=1.0,
    )
    whitener = DICSpectralWhitener(
        power_spectral_density=np.ones(displacement.shape),
        spectral_floor=1.0e-8,
    )
    return PlasticObservabilityOperator(
        (PlasticObservabilityState(linearisation),),
        grid,
        whitener,
    )


def test_plastic_and_observation_adjoint_checks() -> None:
    operator = _operator()
    errors = operator.adjoint_errors()
    assert errors["gp_relative_error"] < 1.0e-11
    assert errors["observation_relative_error"] < 1.0e-8


def test_information_operator_is_positive_semidefinite() -> None:
    operator = _operator()
    rng = np.random.default_rng(42)
    plastic = rng.normal(size=operator.plastic_shape)
    action = operator.information_action(plastic)
    quadratic = float(np.vdot(plastic, action).real)
    assert quadratic >= -1.0e-9


def test_spatial_plastic_metric_is_positive_definite() -> None:
    operator = _operator()
    metric = PlasticMetric(amplitude_weight=1.0, spatial_weight=2.0)
    rng = np.random.default_rng(43)
    plastic = rng.normal(size=operator.plastic_shape)
    quadratic = float(np.vdot(plastic, metric.action(plastic)).real)
    assert quadratic > 0.0


def test_generalized_modes_are_sorted_and_metric_normalized() -> None:
    operator = _operator()
    eigenvalues, modes = operator.generalized_modes(2, metric=PlasticMetric())
    assert eigenvalues.shape == (2,)
    assert modes.shape == (operator.plastic_size, 2)
    assert eigenvalues[0] >= eigenvalues[1] >= -1.0e-10
    metric_modes = modes.T @ np.stack(
        [PlasticMetric().action(modes[:, index].reshape(operator.plastic_shape)).ravel()
         for index in range(2)],
        axis=1,
    )
    np.testing.assert_allclose(metric_modes, np.eye(2), rtol=1.0e-7, atol=1.0e-7)


def test_gp_action_matches_a_constitutive_central_difference() -> None:
    grid = StructuredGrid2D(3, 3, 3.0, 3.0)
    kinematics = TwoSubcellDiagnostic2D(grid)
    displacement = np.zeros((*grid.node_shape, 2))
    displacement[..., 0] = np.linspace(0.0, 0.01, grid.node_shape[0])[:, None]
    displacement[..., 1] = np.linspace(0.0, -0.004, grid.node_shape[1])[None, :]
    plastic = np.full((*grid.pixel_shape, 2), 0.0015)
    direction = np.random.default_rng(7).normal(size=plastic.shape)
    direction *= 0.1
    step = 1.0e-7

    def evaluate(increment: np.ndarray) -> np.ndarray:
        material = DrivenJ2PlaneStressBatch(
            kinematics.material_point_count,
            young_modulus_mpa=205_000.0,
            poisson_ratio=0.30,
        )
        return evaluate_experimental_mechanical_oracle(
            material,
            kinematics,
            displacement,
            increment,
            time_increment=1.0,
        ).mechanical_residual

    base_material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    base = evaluate_experimental_mechanical_oracle(
        base_material,
        kinematics,
        displacement,
        plastic,
        time_increment=1.0,
    )
    finite_difference = (evaluate(plastic + step * direction) - evaluate(
        plastic - step * direction
    )) / (2.0 * step)
    analytical = base.plastic_residual_action(direction)
    relative_error = np.linalg.norm(analytical - finite_difference) / np.linalg.norm(
        finite_difference
    )
    assert relative_error < 1.0e-6
