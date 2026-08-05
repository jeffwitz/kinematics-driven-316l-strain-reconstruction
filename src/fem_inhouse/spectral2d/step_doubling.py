"""Error estimation primitives for SRIX step doubling.

This module contains no Newton, MGIS, or FFT logic.  It compares two complete
step results and returns a scalar acceptance ratio plus per-observable metrics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StepDoublingErrorConfig:
    enabled: bool = False
    stress_relative_tolerance: float = 1.0e-3
    stress_absolute_tolerance_mpa: float = 1.0e-3
    reaction_relative_tolerance: float = 1.0e-3
    reaction_absolute_tolerance: float = 1.0e-10
    signed_slip_relative_tolerance: float = 1.0e-3
    signed_slip_absolute_tolerance: float = 1.0e-8
    accumulated_slip_relative_tolerance: float = 1.0e-3
    accumulated_slip_absolute_tolerance: float = 1.0e-8
    displacement_relative_tolerance: float = 1.0e-5
    displacement_absolute_tolerance: float = 1.0e-12
    activity_threshold: float = 1.0e-8
    safety_factor: float = 0.8
    minimum_shrink_factor: float = 0.25
    maximum_shrink_factor: float = 0.8
    minimum_growth_factor: float = 1.0
    maximum_growth_factor: float = 2.0
    assumed_method_order: float = 1.0

    def __post_init__(self) -> None:
        tolerances = (
            self.stress_relative_tolerance,
            self.stress_absolute_tolerance_mpa,
            self.reaction_relative_tolerance,
            self.reaction_absolute_tolerance,
            self.signed_slip_relative_tolerance,
            self.signed_slip_absolute_tolerance,
            self.accumulated_slip_relative_tolerance,
            self.accumulated_slip_absolute_tolerance,
            self.displacement_relative_tolerance,
            self.displacement_absolute_tolerance,
            self.activity_threshold,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in tolerances):
            raise ValueError("step-doubling tolerances must be finite and non-negative")
        if not 0.0 < self.safety_factor <= 1.0:
            raise ValueError("step-doubling safety factor must be in (0, 1]")
        if not 0.0 < self.minimum_shrink_factor <= self.maximum_shrink_factor <= 1.0:
            raise ValueError("invalid step-doubling shrink bounds")
        if not 1.0 <= self.minimum_growth_factor <= self.maximum_growth_factor:
            raise ValueError("invalid step-doubling growth bounds")
        if self.assumed_method_order < 0.0:
            raise ValueError("assumed method order must be non-negative")


@dataclass(frozen=True, slots=True)
class StepObservables:
    displacement: FloatArray
    stress_in_plane_mpa: FloatArray
    reaction_forces: FloatArray
    plastic_slip: FloatArray
    equivalent_plastic_slip: FloatArray
    accumulated_slip: FloatArray

    def __post_init__(self) -> None:
        for name in (
            "displacement",
            "stress_in_plane_mpa",
            "reaction_forces",
            "plastic_slip",
            "equivalent_plastic_slip",
            "accumulated_slip",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
            object.__setattr__(self, name, value.copy())


@dataclass(frozen=True, slots=True)
class LoadStepAttempt:
    """Result of one isolated solve on one load-path interval.

    The callback used by :func:`estimate_step_error_by_doubling` must treat
    ``start_snapshot`` as immutable and return a state candidate that can be
    passed as the snapshot of the following half-step.
    """

    succeeded: bool
    start_fraction: float
    end_fraction: float
    state: object | None
    observables: StepObservables | None
    diagnostics: object
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class StepDoublingResult:
    """Complete coarse/fine comparison and the controller decision."""

    coarse: LoadStepAttempt
    first_half: LoadStepAttempt | None
    second_half: LoadStepAttempt | None
    error: StepErrorEstimate | None
    accepted: bool
    decision_reason: str
    fine_state: object | None
    next_step_factor: float


AttemptSolver = Callable[[float, float, object], LoadStepAttempt]


@dataclass(frozen=True, slots=True)
class ObservableError:
    relative_l2: float
    relative_linf: float
    maximum_absolute: float
    ratio: float


@dataclass(frozen=True, slots=True)
class StepErrorEstimate:
    stress: ObservableError
    reactions: ObservableError
    displacement: ObservableError
    signed_slip: ObservableError
    accumulated_slip: ObservableError
    signed_slip_ratio_per_system: FloatArray
    accumulated_slip_ratio_per_system: FloatArray
    maximum_ratio: float
    controlling_quantity: str
    controlling_system: int | None


def _error(
    fine: FloatArray,
    coarse: FloatArray,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> ObservableError:
    difference = np.asarray(fine, dtype=np.float64) - np.asarray(coarse, dtype=np.float64)
    fine_values = np.asarray(fine, dtype=np.float64)
    l2_error = float(np.linalg.norm(difference))
    l2_scale = absolute_tolerance + relative_tolerance * float(np.linalg.norm(fine_values))
    linf_error = float(np.max(np.abs(difference), initial=0.0))
    linf_scale = absolute_tolerance + relative_tolerance * float(
        np.max(np.abs(fine_values), initial=0.0)
    )
    relative_l2 = l2_error / max(float(np.linalg.norm(fine_values)), 1.0e-30)
    relative_linf = linf_error / max(float(np.max(np.abs(fine_values), initial=0.0)), 1.0e-30)
    return ObservableError(
        relative_l2=relative_l2,
        relative_linf=relative_linf,
        maximum_absolute=linf_error,
        ratio=max(l2_error / max(l2_scale, 1.0e-30), linf_error / max(linf_scale, 1.0e-30)),
    )


def _per_system_error(
    fine: FloatArray,
    coarse: FloatArray,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[ObservableError, FloatArray]:
    if fine.shape != coarse.shape or fine.ndim < 1 or fine.shape[-1] == 0:
        raise ValueError("per-system observables must have matching non-empty system axes")
    ratios = np.empty(fine.shape[-1], dtype=np.float64)
    l2_values: list[float] = []
    linf_values: list[float] = []
    maximum_absolute = 0.0
    for system in range(fine.shape[-1]):
        metric = _error(
            fine[..., system],
            coarse[..., system],
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        ratios[system] = metric.ratio
        l2_values.append(metric.relative_l2)
        linf_values.append(metric.relative_linf)
        maximum_absolute = max(maximum_absolute, metric.maximum_absolute)
    return (
        ObservableError(
            relative_l2=max(l2_values),
            relative_linf=max(linf_values),
            maximum_absolute=maximum_absolute,
            ratio=float(np.max(ratios)),
        ),
        ratios,
    )


def estimate_step_error(
    fine: StepObservables,
    coarse: StepObservables,
    config: StepDoublingErrorConfig,
) -> StepErrorEstimate:
    """Compare the two final states, using the fine state as the scale."""

    stress = _error(
        fine.stress_in_plane_mpa,
        coarse.stress_in_plane_mpa,
        relative_tolerance=config.stress_relative_tolerance,
        absolute_tolerance=config.stress_absolute_tolerance_mpa,
    )
    reactions = _error(
        fine.reaction_forces,
        coarse.reaction_forces,
        relative_tolerance=config.reaction_relative_tolerance,
        absolute_tolerance=config.reaction_absolute_tolerance,
    )
    displacement = _error(
        fine.displacement,
        coarse.displacement,
        relative_tolerance=config.displacement_relative_tolerance,
        absolute_tolerance=config.displacement_absolute_tolerance,
    )
    signed_slip, signed_ratios = _per_system_error(
        fine.plastic_slip,
        coarse.plastic_slip,
        relative_tolerance=config.signed_slip_relative_tolerance,
        absolute_tolerance=config.signed_slip_absolute_tolerance,
    )
    accumulated_slip, accumulated_ratios = _per_system_error(
        fine.accumulated_slip,
        coarse.accumulated_slip,
        relative_tolerance=config.accumulated_slip_relative_tolerance,
        absolute_tolerance=config.accumulated_slip_absolute_tolerance,
    )
    candidates: list[tuple[str, float, int | None]] = [
        ("stress", stress.ratio, None),
        ("reactions", reactions.ratio, None),
        ("displacement", displacement.ratio, None),
        ("signed_slip", signed_slip.ratio, int(np.argmax(signed_ratios))),
        ("accumulated_slip", accumulated_slip.ratio, int(np.argmax(accumulated_ratios))),
    ]
    controlling_quantity, maximum_ratio, controlling_system = max(
        candidates, key=lambda item: item[1]
    )
    return StepErrorEstimate(
        stress=stress,
        reactions=reactions,
        displacement=displacement,
        signed_slip=signed_slip,
        accumulated_slip=accumulated_slip,
        signed_slip_ratio_per_system=signed_ratios,
        accumulated_slip_ratio_per_system=accumulated_ratios,
        maximum_ratio=maximum_ratio,
        controlling_quantity=controlling_quantity,
        controlling_system=controlling_system,
    )


def next_step_factor(
    error_ratio: float,
    config: StepDoublingErrorConfig,
    *,
    accepted: bool,
) -> tuple[float, str]:
    """Return a bounded step factor for the next attempt."""

    if not np.isfinite(error_ratio):
        return config.minimum_shrink_factor, "non_finite_error_estimate"
    if error_ratio <= 1.0e-30:
        raw_factor = config.maximum_growth_factor
    else:
        raw_factor = config.safety_factor * error_ratio ** (
            -1.0 / (config.assumed_method_order + 1.0)
        )
    if accepted:
        return (
            float(np.clip(raw_factor, config.minimum_growth_factor, config.maximum_growth_factor)),
            "accepted_error_controlled_step",
        )
    return (
        float(np.clip(raw_factor, config.minimum_shrink_factor, config.maximum_shrink_factor)),
        "rejected_error_controlled_step",
    )


def estimate_step_error_by_doubling(
    start_fraction: float,
    end_fraction: float,
    start_snapshot: object,
    *,
    attempt_solver: AttemptSolver,
    config: StepDoublingErrorConfig,
) -> StepDoublingResult:
    """Solve one coarse step and two half-steps from one transaction root.

    ``attempt_solver`` owns the material and global-state snapshot mechanics.
    It must not mutate ``start_snapshot``.  This separation keeps MGIS and
    Newton details out of the numerical error controller and makes rollback a
    testable contract at the solver boundary.
    """

    if not end_fraction > start_fraction:
        raise ValueError("step-doubling interval must have positive length")
    midpoint = start_fraction + 0.5 * (end_fraction - start_fraction)
    coarse = attempt_solver(start_fraction, end_fraction, start_snapshot)
    if not coarse.succeeded:
        factor, _ = next_step_factor(np.inf, config, accepted=False)
        return StepDoublingResult(
            coarse=coarse,
            first_half=None,
            second_half=None,
            error=None,
            accepted=False,
            decision_reason=f"coarse_step_failure:{coarse.failure_reason or 'unknown'}",
            fine_state=None,
            next_step_factor=factor,
        )

    first_half = attempt_solver(start_fraction, midpoint, start_snapshot)
    if not first_half.succeeded or first_half.state is None:
        factor, _ = next_step_factor(np.inf, config, accepted=False)
        return StepDoublingResult(
            coarse=coarse,
            first_half=first_half,
            second_half=None,
            error=None,
            accepted=False,
            decision_reason=(
                f"first_half_failure:{first_half.failure_reason or 'missing_state'}"
            ),
            fine_state=None,
            next_step_factor=factor,
        )

    second_half = attempt_solver(midpoint, end_fraction, first_half.state)
    if not second_half.succeeded or second_half.state is None:
        factor, _ = next_step_factor(np.inf, config, accepted=False)
        return StepDoublingResult(
            coarse=coarse,
            first_half=first_half,
            second_half=second_half,
            error=None,
            accepted=False,
            decision_reason=(
                f"second_half_failure:{second_half.failure_reason or 'missing_state'}"
            ),
            fine_state=None,
            next_step_factor=factor,
        )

    if coarse.observables is None or second_half.observables is None:
        raise ValueError("successful attempts must return observables")
    error = estimate_step_error(second_half.observables, coarse.observables, config)
    accepted = error.maximum_ratio <= 1.0
    factor, factor_reason = next_step_factor(
        error.maximum_ratio,
        config,
        accepted=accepted,
    )
    return StepDoublingResult(
        coarse=coarse,
        first_half=first_half,
        second_half=second_half,
        error=error,
        accepted=accepted,
        decision_reason=factor_reason,
        fine_state=second_half.state if accepted else None,
        next_step_factor=factor,
    )
