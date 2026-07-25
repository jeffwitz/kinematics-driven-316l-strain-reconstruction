from dataclasses import replace

import numpy as np
import pytest

from fem_inhouse.config import CaseStudyConfig, SolverConfig
from fem_inhouse.core import nonlinear
from fem_inhouse.core.linear_solver import LinearSolverStatistics
from fem_inhouse.core.nonlocal_plasticity import NonlocalCouplingConvergenceError
from fem_inhouse.core.plane_stress_material import (
    LocalPlaneStressConvergenceError,
    PlaneStressBatchStatistics,
)
from fem_inhouse.examples import reduced_biaxial_case
from fem_inhouse.postprocessing import von_mises_stress
from fem_inhouse.solver import run_case_study


def _solve_case(case, config=None, yield_map=None, hardening_map=None):
    return run_case_study(
        case.config if config is None else config,
        displacement_x_mm=case.displacement_x_mm,
        displacement_y_mm=case.displacement_y_mm,
        yield_stress_mpa=case.yield_stress_mpa if yield_map is None else yield_map,
        hardening_coefficient_mpa=(
            case.hardening_coefficient_mpa if hardening_map is None else hardening_map
        ),
    )


def test_increment_refinement_is_stable() -> None:
    case = reduced_biaxial_case(nx=4, ny=4, constitutive_backend="python")
    responses = []
    for increments in (5, 10, 20):
        config = replace(
            case.config,
            solver=replace(case.config.solver, increments=increments),
        )
        result = _solve_case(case, config=config)
        stress = von_mises_stress(
            result.stress_mpa[..., 0],
            result.stress_mpa[..., 1],
            result.stress_mpa[..., 2],
        )
        responses.append((float(stress.mean()), float(result.equivalent_plastic_strain.mean())))

    response_values = np.asarray(responses)
    relative_spread = np.ptp(response_values, axis=0) / np.abs(response_values[-1])
    assert np.all(relative_spread < 5e-4)


def test_heterogeneous_case_converges_to_finite_balanced_result() -> None:
    case = reduced_biaxial_case(nx=6, ny=6, constitutive_backend="python")
    indices = np.indices((6, 6))
    modulation = (indices[0] + indices[1]) % 2
    yield_map = np.where(modulation, 220.0, 280.0)
    hardening_map = np.where(modulation, 420.0, 580.0)

    result = _solve_case(case, yield_map=yield_map, hardening_map=hardening_map)

    assert all(np.isfinite(field).all() for field in result.arrays())
    assert result.equivalent_plastic_strain.max() > 0
    assert result.diagnostics is not None
    assert result.diagnostics.converged_increments > 0
    assert result.diagnostics.total_newton_iterations >= result.diagnostics.converged_increments
    assert (
        result.diagnostics.final_residual_norm < 1e-10
        or result.diagnostics.final_relative_residual < case.config.solver.residual_tolerance
    )
    assert result.diagnostics.final_convergence_criterion in {
        "absolute_residual",
        "relative_residual",
    }
    assert result.diagnostics.elapsed_seconds > 0
    assert result.diagnostics.linear_solve_seconds > 0
    assert result.diagnostics.pardiso_analysis_calls == 1
    assert result.diagnostics.pardiso_factorization_calls >= 1
    assert (
        result.diagnostics.pardiso_factorization_calls
        == result.diagnostics.pardiso_solve_calls
    )
    assert result.diagnostics.constitutive_seconds > 0
    assert result.diagnostics.output_seconds > 0
    assert result.diagnostics.linear_system_matrix_type == (
        "symmetric_positive_definite"
    )
    assert result.diagnostics.maximum_relative_constitutive_tangent_asymmetry < 1e-12
    assert "mtype=2" in result.diagnostics.backend
    net_reaction = np.linalg.norm(result.reaction_force.sum(axis=(0, 1)))
    reaction_scale = np.linalg.norm(result.reaction_force, axis=-1).sum()
    assert net_reaction / reaction_scale < 1e-10


def test_nonconvergence_raises_diagnostic_error(monkeypatch) -> None:
    case = reduced_biaxial_case(nx=4, ny=4, constitutive_backend="python")
    config = CaseStudyConfig(
        mesh=case.config.mesh,
        material=case.config.material,
        solver=SolverConfig(
            increments=1,
            max_newton_iterations=2,
            minimum_step_divisor=2,
            require_pypardiso=False,
            hardening_mode="tabular",
            constitutive_backend="python",
        ),
    )

    class NonfiniteLinearSolver:
        backend_name = "test non-finite solver"
        matrix_storage = "full"
        statistics = LinearSolverStatistics()

        @staticmethod
        def factorize_and_solve(_matrix, right_hand_side):
            return np.full_like(right_hand_side, np.nan)

        @staticmethod
        def close() -> None:
            return None

    monkeypatch.setattr(
        nonlinear,
        "create_linear_solver",
        lambda _matrix_type: NonfiniteLinearSolver(),
    )
    with pytest.raises(RuntimeError, match="increment cutback below minimum"):
        _solve_case(case, config=config)


def test_local_constitutive_failure_triggers_clean_global_cutback(monkeypatch) -> None:
    case = reduced_biaxial_case(nx=4, ny=4, constitutive_backend="python")
    original_factory = nonlinear.create_plane_stress_material_batch
    wrapper_holder = {}

    class FailFirstTrial:
        def __init__(self, delegate):
            self.delegate = delegate
            self.failed = False
            self.reverts = 0

        @property
        def point_count(self):
            return self.delegate.point_count

        @property
        def backend_name(self):
            return self.delegate.backend_name

        @property
        def completion_strategy(self):
            return self.delegate.completion_strategy

        @property
        def statistics(self):
            return PlaneStressBatchStatistics(local_plane_stress_failures=int(self.failed))

        def evaluate(self, *args, **kwargs):
            if not self.failed:
                self.failed = True
                raise LocalPlaneStressConvergenceError("injected local failure")
            return self.delegate.evaluate(*args, **kwargs)

        def commit(self):
            self.delegate.commit()

        def revert(self):
            self.reverts += 1
            self.delegate.revert()

    def factory(*args, **kwargs):
        wrapper = FailFirstTrial(original_factory(*args, **kwargs))
        wrapper_holder["batch"] = wrapper
        return wrapper

    monkeypatch.setattr(nonlinear, "create_plane_stress_material_batch", factory)
    result = _solve_case(case)

    wrapper = wrapper_holder["batch"]
    assert result.diagnostics is not None
    assert result.diagnostics.cutbacks == 1
    assert result.diagnostics.local_plane_stress_failures == 1
    assert wrapper.reverts >= 1
    assert all(np.isfinite(field).all() for field in result.arrays())


def test_nonlocal_failure_restarts_from_committed_chi_after_cutback(monkeypatch) -> None:
    case = reduced_biaxial_case(nx=4, ny=4, constitutive_backend="python")
    config = replace(
        case.config,
        solver=replace(
            case.config.solver,
            constitutive_backend="mfront-native-plane-stress",
        ),
        nonlocal_plasticity=replace(
            case.config.nonlocal_plasticity,
            enabled=True,
            coupling_modulus_mpa=0.0,
        ),
    )
    original_factory = nonlinear.create_plane_stress_material_batch
    original_fixed_point = nonlinear.evaluate_nonlocal_fixed_point
    wrapper_holder = {}
    initial_fields = []

    class NonlocalPythonWrapper:
        def __init__(self, delegate):
            self.delegate = delegate
            self.reverts = 0
            self.external_chi = np.zeros(delegate.point_count)
            self.committed_chi = self.external_chi.copy()

        def __getattr__(self, name):
            return getattr(self.delegate, name)

        def set_nonlocal_equivalent_plastic_strain(self, values):
            self.external_chi = np.asarray(values, dtype=float).copy()

        def commit(self):
            self.delegate.commit()
            self.committed_chi = self.external_chi.copy()

        def revert(self):
            self.reverts += 1
            self.delegate.revert()
            self.external_chi = self.committed_chi.copy()

    def factory(*args, **kwargs):
        delegate = original_factory("python", *args[1:], **kwargs)
        wrapper = NonlocalPythonWrapper(delegate)
        wrapper_holder["batch"] = wrapper
        return wrapper

    def fail_first_fixed_point(*args, **kwargs):
        initial_fields.append(np.asarray(kwargs["initial_nonlocal_peeq"]).copy())
        if len(initial_fields) == 1:
            raise NonlocalCouplingConvergenceError("injected coupling failure")
        return original_fixed_point(*args, **kwargs)

    monkeypatch.setattr(nonlinear, "create_plane_stress_material_batch", factory)
    monkeypatch.setattr(
        nonlinear,
        "evaluate_nonlocal_fixed_point",
        fail_first_fixed_point,
    )

    result = _solve_case(case, config=config)

    wrapper = wrapper_holder["batch"]
    assert result.diagnostics is not None
    assert result.diagnostics.cutbacks == 1
    assert result.diagnostics.nonlocal_coupling_failures == 1
    assert wrapper.reverts >= 1
    np.testing.assert_array_equal(initial_fields[0], 0.0)
    np.testing.assert_array_equal(initial_fields[1], 0.0)
    assert result.nonlocal_equivalent_plastic_strain is not None
    assert all(np.isfinite(field).all() for field in result.arrays())
