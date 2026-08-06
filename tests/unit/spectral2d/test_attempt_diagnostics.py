import pytest

from fem_inhouse.spectral2d.diagnostics import (
    LoadStepAttemptDiagnostics,
    summarize_load_step_attempts,
)


def _attempt(*, accepted: bool, scale: int) -> LoadStepAttemptDiagnostics:
    return LoadStepAttemptDiagnostics(
        attempt_index=scale,
        load_fraction_start=0.0,
        load_fraction_end=0.1 * scale,
        accepted=accepted,
        failure_reason=None if accepted else "line_search_failure",
        newton_iterations=scale,
        linear_solves=scale,
        krylov_outer_callbacks=2 * scale,
        jacobian_matvec_calls=3 * scale,
        preconditioner_calls=4 * scale,
        krylov_seconds=5.0 * scale,
        jacobian_seconds=2.0 * scale,
        preconditioner_seconds=1.0 * scale,
        krylov_overhead_seconds=2.0 * scale,
        material_seconds=7.0 * scale,
        material_evaluations=5 * scale,
        material_integration_seconds=6.0 * scale,
        material_condensation_seconds=7.0 * scale,
        mgis_integrations=8 * scale,
        line_search_rejections=scale,
        minimum_line_search_factor=0.5,
    )


def test_attempt_summary_separates_accepted_and_rejected_work() -> None:
    summary = summarize_load_step_attempts(
        [_attempt(accepted=True, scale=1), _attempt(accepted=False, scale=2)]
    )

    assert summary["accepted"]["attempts"] == 1
    assert summary["rejected"]["attempts"] == 1
    assert summary["accepted"]["jacobian_matvec_calls"] == 3
    assert summary["rejected"]["jacobian_matvec_calls"] == 6
    assert summary["total"]["jacobian_matvec_calls"] == 9
    assert summary["total"]["krylov_seconds"] == pytest.approx(15.0)
    assert summary["total"]["krylov_overhead_seconds"] == pytest.approx(6.0)


def test_attempt_krylov_breakdown_is_exclusive() -> None:
    attempt = _attempt(accepted=True, scale=1)

    assert attempt.krylov_seconds == pytest.approx(
        attempt.jacobian_seconds
        + attempt.preconditioner_seconds
        + attempt.krylov_overhead_seconds
    )
