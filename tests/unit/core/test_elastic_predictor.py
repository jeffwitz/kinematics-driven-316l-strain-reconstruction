"""Regression guard for the elastic predictor of the boundary-history branch.

`FixedCSRAssembler.assemble` deliberately reuses one CSR buffer. `run_fem` must
therefore own a private copy of the elastic operator, otherwise the
history-branch elastic predictor is solved against the last elastoplastic
tangent. See `validation/dic_multistep_p0043_newton_instrumentation_results.md`.
"""

from __future__ import annotations

import numpy as np

from fem_inhouse.core.assembly import FixedCSRAssembler
from fem_inhouse.examples import reduced_biaxial_case
from fem_inhouse.solver import run_case_study


def test_assembler_reuses_one_buffer() -> None:
    """Document the contract the predictor has to defend itself against."""

    location_matrix = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 8, 9, 10, 11, 4, 5]])
    assembler = FixedCSRAssembler.from_location_matrix(location_matrix, np.arange(12))

    first = assembler.assemble(np.eye(8))
    owned = first.copy()
    baseline = owned.data.copy()
    second = assembler.assemble(np.eye(8) * 7.0)

    assert first is second
    assert not np.array_equal(first.data, baseline)
    np.testing.assert_array_equal(owned.data, baseline)


def _heterogeneous_case():
    case = reduced_biaxial_case(nx=6, ny=6, constitutive_backend="python")
    generator = np.random.default_rng(4)
    yield_map = np.asarray(case.yield_stress_mpa) * (
        1.0 + 0.4 * generator.random(np.shape(case.yield_stress_mpa))
    )
    return case, yield_map


def _tangent_assemblies(trace: list[dict[str, object]]) -> int:
    return len([record for record in trace if "tangent_diagonal_minimum" in record])


def test_linear_history_matches_the_proportional_ramp() -> None:
    """An identical boundary path must cost the same whichever branch drives it."""

    case, yield_map = _heterogeneous_case()
    increments = case.config.solver.increments
    displacement_x = np.asarray(case.displacement_x_mm)
    displacement_y = np.asarray(case.displacement_y_mm)
    final = np.stack((displacement_x, displacement_y), axis=-1)
    history = np.stack([final * (step / increments) for step in range(increments + 1)])
    shared = {
        "displacement_x_mm": displacement_x,
        "displacement_y_mm": displacement_y,
        "yield_stress_mpa": yield_map,
        "hardening_coefficient_mpa": case.hardening_coefficient_mpa,
    }

    proportional_trace: list[dict[str, object]] = []
    history_trace: list[dict[str, object]] = []
    proportional = run_case_study(
        case.config, newton_trace=proportional_trace, **shared
    )
    driven = run_case_study(
        case.config,
        boundary_displacement_history_mm=history,
        newton_trace=history_trace,
        **shared,
    )

    # A corrupted predictor still converges here, but needs far more iterations.
    assert _tangent_assemblies(history_trace) == _tangent_assemblies(proportional_trace)
    assert _tangent_assemblies(history_trace) > 0
    np.testing.assert_allclose(
        driven.stress_mpa, proportional.stress_mpa, rtol=0.0, atol=1.0e-6
    )
    np.testing.assert_allclose(
        driven.equivalent_plastic_strain,
        proportional.equivalent_plastic_strain,
        rtol=0.0,
        atol=1.0e-11,
    )


def test_history_predictor_stays_elastic_across_increments() -> None:
    """Trial strains must never leave the physical range of the imposed path."""

    case, yield_map = _heterogeneous_case()
    increments = case.config.solver.increments
    displacement_x = np.asarray(case.displacement_x_mm)
    displacement_y = np.asarray(case.displacement_y_mm)
    final = np.stack((displacement_x, displacement_y), axis=-1)
    history = np.stack([final * (step / increments) for step in range(increments + 1)])

    trace: list[dict[str, object]] = []
    run_case_study(
        case.config,
        displacement_x_mm=displacement_x,
        displacement_y_mm=displacement_y,
        yield_stress_mpa=yield_map,
        hardening_coefficient_mpa=case.hardening_coefficient_mpa,
        boundary_displacement_history_mm=history,
        newton_trace=trace,
    )

    imposed = float(np.max(np.abs(final))) / case.config.mesh.base_pixel_size_mm
    largest = max(float(record["total_strain_maximum"]) for record in trace)

    assert largest < imposed
    assert all(record["outcome"] in {"converged", "corrected"} for record in trace)
