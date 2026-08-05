import numpy as np
import pytest

from fem_inhouse.spectral2d import (
    LoadStepAttempt,
    StepDoublingErrorConfig,
    StepObservables,
    estimate_step_error,
    estimate_step_error_by_doubling,
    next_step_factor,
)


def _observables(scale: float) -> StepObservables:
    base = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    return StepObservables(
        displacement=scale * base[..., :2],
        stress_in_plane_mpa=scale * base[..., :3],
        reaction_forces=scale * base[:2, :2, :2],
        plastic_slip=scale * base,
        equivalent_plastic_slip=scale * base,
        accumulated_slip=scale * base,
    )


def test_step_doubling_is_zero_for_identical_states() -> None:
    state = _observables(1.0)
    estimate = estimate_step_error(state, state, StepDoublingErrorConfig())
    assert estimate.maximum_ratio == 0.0
    assert estimate.controlling_quantity == "stress"


def test_step_doubling_reports_system_resolved_error() -> None:
    fine = _observables(1.0)
    coarse = _observables(1.0)
    coarse.plastic_slip[..., 2] += 0.1
    estimate = estimate_step_error(fine, coarse, StepDoublingErrorConfig())
    assert estimate.signed_slip_ratio_per_system[2] > 0.0
    assert estimate.controlling_system == 2


def test_step_factor_shrinks_and_grows_with_error() -> None:
    config = StepDoublingErrorConfig()
    shrink, shrink_reason = next_step_factor(4.0, config, accepted=False)
    grow, grow_reason = next_step_factor(0.01, config, accepted=True)
    assert 0.25 <= shrink <= 0.8
    assert 1.0 <= grow <= 2.0
    assert shrink_reason == "rejected_error_controlled_step"
    assert grow_reason == "accepted_error_controlled_step"


def test_step_doubling_rejects_invalid_shapes() -> None:
    state = _observables(1.0)
    broken = StepObservables(
        displacement=state.displacement,
        stress_in_plane_mpa=state.stress_in_plane_mpa,
        reaction_forces=state.reaction_forces,
        plastic_slip=state.plastic_slip[..., :2],
        equivalent_plastic_slip=state.equivalent_plastic_slip,
        accumulated_slip=state.accumulated_slip,
    )
    with pytest.raises(ValueError, match="matching"):
        estimate_step_error(state, broken, StepDoublingErrorConfig())


def test_step_doubling_runs_branches_from_one_snapshot_and_returns_fine_state() -> None:
    calls: list[tuple[float, float, float]] = []

    def attempt(start: float, end: float, snapshot: object) -> LoadStepAttempt:
        value = float(snapshot) + (end - start) * float(snapshot)
        calls.append((start, end, float(snapshot)))
        return LoadStepAttempt(
            succeeded=True,
            start_fraction=start,
            end_fraction=end,
            state=value,
            observables=_observables(value),
            diagnostics={"value": value},
        )

    result = estimate_step_error_by_doubling(
        0.0,
        0.1,
        1.0,
        attempt_solver=attempt,
        config=StepDoublingErrorConfig(
            stress_relative_tolerance=1.0,
            reaction_relative_tolerance=1.0,
            displacement_relative_tolerance=1.0,
            signed_slip_relative_tolerance=1.0,
            accumulated_slip_relative_tolerance=1.0,
        ),
    )

    assert result.accepted
    assert result.fine_state == pytest.approx(1.1025)
    assert calls == [(0.0, 0.1, 1.0), (0.0, 0.05, 1.0), (0.05, 0.1, 1.05)]


def test_step_doubling_rejects_excess_error_without_fine_commit() -> None:
    def attempt(start: float, end: float, snapshot: object) -> LoadStepAttempt:
        value = float(snapshot) * (1.0 + end - start)
        return LoadStepAttempt(
            succeeded=True,
            start_fraction=start,
            end_fraction=end,
            state=value,
            observables=_observables(value),
            diagnostics=None,
        )

    result = estimate_step_error_by_doubling(
        0.0,
        1.0,
        1.0,
        attempt_solver=attempt,
        config=StepDoublingErrorConfig(),
    )

    assert not result.accepted
    assert result.fine_state is None
    assert result.decision_reason == "rejected_error_controlled_step"
