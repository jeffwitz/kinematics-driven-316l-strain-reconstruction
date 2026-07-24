from dataclasses import replace

import numpy as np
import pytest

from fem_inhouse.config import CaseStudyConfig, SolverConfig
from fem_inhouse.core import nonlinear
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
    case = reduced_biaxial_case(nx=4, ny=4)
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
    case = reduced_biaxial_case(nx=6, ny=6)
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
    assert result.diagnostics.constitutive_seconds > 0
    assert result.diagnostics.output_seconds > 0
    net_reaction = np.linalg.norm(result.reaction_force.sum(axis=(0, 1)))
    reaction_scale = np.linalg.norm(result.reaction_force, axis=-1).sum()
    assert net_reaction / reaction_scale < 1e-10


def test_nonconvergence_raises_diagnostic_error(monkeypatch) -> None:
    case = reduced_biaxial_case(nx=4, ny=4)
    config = CaseStudyConfig(
        mesh=case.config.mesh,
        material=case.config.material,
        solver=SolverConfig(
            increments=1,
            max_newton_iterations=2,
            minimum_step_divisor=2,
            require_pypardiso=False,
            hardening_mode="tabular",
        ),
    )

    def nonfinite_solution(_matrix, right_hand_side):
        return np.full_like(right_hand_side, np.nan)

    monkeypatch.setattr(nonlinear, "_solve", nonfinite_solution)
    with pytest.raises(RuntimeError, match="increment cutback below minimum"):
        _solve_case(case, config=config)
