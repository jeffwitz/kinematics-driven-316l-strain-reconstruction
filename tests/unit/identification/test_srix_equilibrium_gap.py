from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.core.plane_stress_material import InPlaneConstitutiveTrial
from fem_inhouse.identification.srix_equilibrium_gap import (
    SrixEquilibriumGapProblem,
    SrixTheta4,
)
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior


class _Identity:
    def apply(self, values):
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values):
        return np.asarray(values, dtype=np.float64)


class _LinearBatch:
    def __init__(self, point_count: int, scale: float) -> None:
        self.point_count = point_count
        self.scale = scale
        self.tangent = plane_stress_elasticity(205_000.0, 0.3)
        self.committed_calls = 0

    @property
    def backend_name(self):
        return "test-linear"

    @property
    def completion_strategy(self):
        return "test"

    @property
    def linear_system_matrix_type(self):
        return 2

    @property
    def statistics(self):
        raise NotImplementedError

    def evaluate_in_plane(self, in_plane_strain, *, time_increment, consistent_tangent=True):
        values = np.asarray(in_plane_strain, dtype=np.float64)
        stress = self.scale * (values @ self.tangent.T)
        return InPlaneConstitutiveTrial(
            stress_in_plane_mpa=stress,
            tangent_in_plane_mpa=(
                np.broadcast_to(self.scale * self.tangent, (self.point_count, 3, 3)).copy()
                if consistent_tangent
                else None
            ),
        )

    def evaluate_in_plane_response(
        self,
        in_plane_strain,
        *,
        time_increment,
        response_level,
        consistent_tangent=True,
    ):
        return self.evaluate_in_plane(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=response_level == "tangent" and consistent_tangent,
        )

    def complete_trial(self, trial):
        raise NotImplementedError

    def commit(self):
        self.committed_calls += 1

    def revert(self):
        return None


def _operator(pixels: int = 5) -> TensorPlasticObservabilityOperator:
    grid = StructuredGrid2D(pixels, pixels, 0.00184 * pixels, 0.00184 * pixels)
    return TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.3,
        transfer=_Identity(),
        whitener=_Identity(),
    )


def _theta(scale: float = 1.0) -> SrixTheta4:
    return SrixTheta4(100.0 * scale, 200.0, 300.0, 10.0)


def _factory(operator: TensorPlasticObservabilityOperator):
    def create(overrides: Mapping[str, float]):
        return _LinearBatch(
            operator.kinematics.material_point_count,
            float(overrides["tau0_mpa"]) / 100.0,
        )

    return create


def test_uniform_stress_has_zero_interior_weak_residual() -> None:
    operator = _operator()
    stress = np.broadcast_to(
        np.array([120.0, 80.0, 15.0]), (*operator.grid.pixel_shape, 2, 3)
    )
    residual = operator.weak_equilibrium_residual(stress)
    np.testing.assert_allclose(residual, 0.0, atol=5e-12)


def test_affine_elastic_field_has_zero_gap() -> None:
    operator = _operator()
    x = np.linspace(0.0, operator.grid.length_x, operator.grid.nx + 1)
    y = np.linspace(0.0, operator.grid.length_y, operator.grid.ny + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    displacement = np.stack((0.01 * xx + 0.002 * yy, -0.003 * xx + 0.005 * yy), axis=-1)
    history = np.stack((np.zeros_like(displacement), displacement))
    problem = SrixEquilibriumGapProblem(
        operator=operator,
        displacement_history=history,
        state_indices=(1,),
        scored_states={1},
        material_factory=_factory(operator),
    )
    result = problem.evaluate(_theta())
    # Dividing the assembled force by the 1.69e-6 mm2 sub-cell area exposes a
    # harmless round-off force density; the displacement correction is the
    # physically scaled quantity used by REGM.
    assert result.states[0].raw_equilibrium_norm < 2e-8
    assert result.states[0].pseudo_displacement_norm < 1e-14


def test_reconditioned_sign_reduces_the_elastic_residual() -> None:
    operator = _operator(6)
    generator = np.random.default_rng(12)
    free = generator.normal(size=operator.free_size) * 1e-6
    displacement = unpack_interior(free, operator.grid)
    strain = operator.kinematics.strain(displacement)
    elasticity = plane_stress_elasticity(205_000.0, 0.3)
    stress = np.einsum("ij,xyqj->xyqi", elasticity, strain)
    before = operator.weak_equilibrium_residual(stress)
    correction = operator.reconditioned_correction(stress)
    corrected_strain = operator.kinematics.strain(displacement + correction)
    corrected_stress = np.einsum("ij,xyqj->xyqi", elasticity, corrected_strain)
    after = operator.weak_equilibrium_residual(corrected_stress)
    wrong_strain = operator.kinematics.strain(displacement - correction)
    wrong_stress = np.einsum("ij,xyqj->xyqi", elasticity, wrong_strain)
    wrong = operator.weak_equilibrium_residual(wrong_stress)
    assert np.linalg.norm(after) < 1e-10 * np.linalg.norm(before)
    assert np.linalg.norm(wrong) > np.linalg.norm(before)
    np.testing.assert_allclose(pack_interior(correction), -free, rtol=1e-9, atol=1e-16)


def test_replay_is_deterministic_and_runtime_parameter_changes_response() -> None:
    operator = _operator()
    generator = np.random.default_rng(3)
    displacement = unpack_interior(generator.normal(size=operator.free_size) * 2e-6, operator.grid)
    history = np.stack((np.zeros_like(displacement), displacement, 1.5 * displacement))
    problem = SrixEquilibriumGapProblem(
        operator=operator,
        displacement_history=history,
        state_indices=(1, 2),
        scored_states={1, 2},
        material_factory=_factory(operator),
    )
    first = problem.residual_vector(_theta())
    second = problem.residual_vector(_theta())
    changed = problem.residual_vector(_theta(1.1))
    np.testing.assert_array_equal(first, second)
    assert not np.allclose(first, changed)


def test_log_parameter_conversion_and_fd_sensitivity() -> None:
    operator = _operator()
    generator = np.random.default_rng(7)
    displacement = unpack_interior(generator.normal(size=operator.free_size) * 1e-6, operator.grid)
    problem = SrixEquilibriumGapProblem(
        operator=operator,
        displacement_history=np.stack((np.zeros_like(displacement), displacement)),
        state_indices=(1,),
        scored_states={1},
        material_factory=_factory(operator),
    )
    theta = _theta()
    recovered = SrixTheta4.from_log_coordinates(theta.log_coordinates())
    np.testing.assert_allclose(recovered.as_array(), theta.as_array(), rtol=1e-15)
    coarse = problem.jacobian_fd(theta.log_coordinates(), relative_step=1e-3)
    fine = problem.jacobian_fd(theta.log_coordinates(), relative_step=3e-4)
    np.testing.assert_allclose(coarse, fine, rtol=2e-7, atol=1e-12)
    assert np.linalg.norm(fine[:, 0]) > 0.0
    np.testing.assert_allclose(fine[:, 1:], 0.0, atol=1e-12)
    svd = problem.sensitivity_svd(fine)
    assert svd.numerical_rank == 1
    assert svd.normalized_singular_values[0] == pytest.approx(1.0)
