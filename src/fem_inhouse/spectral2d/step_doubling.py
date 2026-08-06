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
    linf_relative_tolerance_factor: float = 5.0
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
            self.linf_relative_tolerance_factor,
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
        if self.linf_relative_tolerance_factor < 1.0:
            raise ValueError("L-infinity tolerance factor must be at least one")


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
        if self.plastic_slip.shape != self.equivalent_plastic_slip.shape:
            raise ValueError(
                "plastic_slip and equivalent_plastic_slip must have matching shapes"
            )
        if self.plastic_slip.ndim < 1 or self.plastic_slip.shape[-1] != 12:
            raise ValueError(
                "per-system slip observables must have exactly 12 systems on the last axis"
            )


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


class StepDoublingFailureError(RuntimeError):
    """Unrecoverable step-doubling failure with its partial history."""

    def __init__(self, message: str, history: list[dict[str, object]]) -> None:
        super().__init__(message)
        self.history = tuple(history)


@dataclass(frozen=True, slots=True)
class ObservableError:
    relative_l2: float
    relative_linf: float
    maximum_absolute: float
    ratio: float
    weighted_rms_ratio: float
    weighted_linf_ratio: float
    absolute_l2: float
    absolute_linf: float
    fine_l2_norm: float
    coarse_l2_norm: float
    maximum_fine_amplitude: float
    maximum_coarse_amplitude: float


@dataclass(frozen=True, slots=True)
class PerSystemError:
    """Per-system error metrics and activity classification."""

    ratios: FloatArray
    weighted_rms_ratios: FloatArray
    weighted_linf_ratios: FloatArray
    fine_l2_norms: FloatArray
    coarse_l2_norms: FloatArray
    difference_l2_norms: FloatArray
    fine_linf_amplitudes: FloatArray
    coarse_linf_amplitudes: FloatArray
    active_fine: NDArray[np.bool_]
    active_coarse: NDArray[np.bool_]
    active_set_mismatch: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class StepErrorEstimate:
    stress: ObservableError
    reactions: ObservableError
    displacement: ObservableError
    signed_slip: ObservableError
    accumulated_slip: ObservableError
    signed_slip_ratio_per_system: FloatArray
    accumulated_slip_ratio_per_system: FloatArray
    signed_slip_details: PerSystemError
    accumulated_slip_details: PerSystemError
    equivalent_plastic_slip: ObservableError
    equivalent_plastic_slip_ratio_per_system: FloatArray
    equivalent_plastic_slip_details: PerSystemError
    maximum_ratio: float
    controlling_quantity: str
    controlling_system: int | None


def _observable_error_record(metric: ObservableError) -> dict[str, float]:
    return {
        "relative_l2": metric.relative_l2,
        "relative_linf": metric.relative_linf,
        "maximum_absolute": metric.maximum_absolute,
        "weighted_rms_ratio": metric.weighted_rms_ratio,
        "weighted_linf_ratio": metric.weighted_linf_ratio,
        "ratio": metric.ratio,
        "absolute_l2": metric.absolute_l2,
        "absolute_linf": metric.absolute_linf,
        "fine_l2_norm": metric.fine_l2_norm,
        "coarse_l2_norm": metric.coarse_l2_norm,
        "maximum_fine_amplitude": metric.maximum_fine_amplitude,
        "maximum_coarse_amplitude": metric.maximum_coarse_amplitude,
    }


def _per_system_error_record(details: PerSystemError) -> dict[str, object]:
    return {
        "ratio_per_system": details.ratios.tolist(),
        "weighted_rms_ratio_per_system": details.weighted_rms_ratios.tolist(),
        "weighted_linf_ratio_per_system": details.weighted_linf_ratios.tolist(),
        "fine_norm_per_system": details.fine_l2_norms.tolist(),
        "coarse_norm_per_system": details.coarse_l2_norms.tolist(),
        "difference_norm_per_system": details.difference_l2_norms.tolist(),
        "maximum_fine_amplitude_per_system": details.fine_linf_amplitudes.tolist(),
        "maximum_coarse_amplitude_per_system": details.coarse_linf_amplitudes.tolist(),
        "active_fine_per_system": details.active_fine.tolist(),
        "active_coarse_per_system": details.active_coarse.tolist(),
        "active_set_mismatch_per_system": details.active_set_mismatch.tolist(),
    }


def step_error_to_record(error: StepErrorEstimate) -> dict[str, object]:
    """Return a JSON-compatible diagnostic record for one comparison."""

    return {
        "maximum_error_ratio": error.maximum_ratio,
        "controlling_quantity": error.controlling_quantity,
        "controlling_system": error.controlling_system,
        "stress": _observable_error_record(error.stress),
        "reactions": _observable_error_record(error.reactions),
        "displacement": _observable_error_record(error.displacement),
        "signed_slip": {
            **_observable_error_record(error.signed_slip),
            **_per_system_error_record(error.signed_slip_details),
        },
        "equivalent_plastic_slip": {
            **_observable_error_record(error.equivalent_plastic_slip),
            **_per_system_error_record(error.equivalent_plastic_slip_details),
        },
    }


def _error(
    fine: FloatArray,
    coarse: FloatArray,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    linf_relative_tolerance: float | None = None,
) -> ObservableError:
    difference = np.asarray(fine, dtype=np.float64) - np.asarray(coarse, dtype=np.float64)
    fine_values = np.asarray(fine, dtype=np.float64)
    l2_error = float(np.linalg.norm(difference))
    if linf_relative_tolerance is None:
        linf_relative_tolerance = relative_tolerance
    scale = absolute_tolerance + relative_tolerance * np.maximum(
        np.abs(fine_values), np.abs(np.asarray(coarse, dtype=np.float64))
    )
    weighted = np.abs(difference) / np.maximum(scale, 1.0e-30)
    weighted_rms = float(np.sqrt(np.mean(weighted**2)))
    linf_scale = absolute_tolerance + linf_relative_tolerance * np.maximum(
        np.abs(fine_values), np.abs(np.asarray(coarse, dtype=np.float64))
    )
    weighted_linf = float(
        np.max(np.abs(difference) / np.maximum(linf_scale, 1.0e-30), initial=0.0)
    )
    linf_error = float(np.max(np.abs(difference), initial=0.0))
    fine_l2_norm = float(np.linalg.norm(fine_values))
    coarse_l2_norm = float(np.linalg.norm(coarse))
    fine_linf = float(np.max(np.abs(fine_values), initial=0.0))
    coarse_linf = float(np.max(np.abs(coarse), initial=0.0))
    relative_l2 = l2_error / max(fine_l2_norm, 1.0e-30)
    relative_linf = linf_error / max(fine_linf, 1.0e-30)
    return ObservableError(
        relative_l2=relative_l2,
        relative_linf=relative_linf,
        maximum_absolute=linf_error,
        ratio=max(weighted_rms, weighted_linf),
        weighted_rms_ratio=weighted_rms,
        weighted_linf_ratio=weighted_linf,
        absolute_l2=l2_error,
        absolute_linf=linf_error,
        fine_l2_norm=fine_l2_norm,
        coarse_l2_norm=coarse_l2_norm,
        maximum_fine_amplitude=fine_linf,
        maximum_coarse_amplitude=coarse_linf,
    )


def _per_system_error(
    fine: FloatArray,
    coarse: FloatArray,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    linf_relative_tolerance: float,
    activity_threshold: float,
) -> tuple[ObservableError, PerSystemError]:
    if (
        fine.shape != coarse.shape
        or fine.ndim < 1
        or fine.shape[-1] != 12
    ):
        raise ValueError("per-system observables must have matching 12-system axes")
    system_count = fine.shape[-1]
    ratios = np.empty(system_count, dtype=np.float64)
    weighted_rms = np.empty(system_count, dtype=np.float64)
    weighted_linf = np.empty(system_count, dtype=np.float64)
    fine_l2 = np.empty(system_count, dtype=np.float64)
    coarse_l2 = np.empty(system_count, dtype=np.float64)
    difference_l2 = np.empty(system_count, dtype=np.float64)
    fine_linf = np.empty(system_count, dtype=np.float64)
    coarse_linf = np.empty(system_count, dtype=np.float64)
    active_fine = np.empty(system_count, dtype=bool)
    active_coarse = np.empty(system_count, dtype=bool)
    active_mismatch = np.empty(system_count, dtype=bool)
    metrics: list[ObservableError] = []
    for system in range(system_count):
        fine_system = fine[..., system]
        coarse_system = coarse[..., system]
        active_fine[system] = bool(
            np.max(np.abs(fine_system), initial=0.0) > activity_threshold
        )
        active_coarse[system] = bool(
            np.max(np.abs(coarse_system), initial=0.0) > activity_threshold
        )
        active_mismatch[system] = active_fine[system] != active_coarse[system]
        effective_relative_tolerance = (
            relative_tolerance
            if active_fine[system] or active_coarse[system]
            else 0.0
        )
        metric = _error(
            fine_system,
            coarse_system,
            relative_tolerance=effective_relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            linf_relative_tolerance=(
                linf_relative_tolerance
                if active_fine[system] or active_coarse[system]
                else 0.0
            ),
        )
        metrics.append(metric)
        ratios[system] = metric.ratio
        weighted_rms[system] = metric.weighted_rms_ratio
        weighted_linf[system] = metric.weighted_linf_ratio
        fine_l2[system] = metric.fine_l2_norm
        coarse_l2[system] = metric.coarse_l2_norm
        difference_l2[system] = metric.absolute_l2
        fine_linf[system] = metric.maximum_fine_amplitude
        coarse_linf[system] = metric.maximum_coarse_amplitude
    maximum_absolute = max(metric.maximum_absolute for metric in metrics)
    return (
        ObservableError(
            relative_l2=max(metric.relative_l2 for metric in metrics),
            relative_linf=max(metric.relative_linf for metric in metrics),
            maximum_absolute=maximum_absolute,
            ratio=float(np.max(ratios)),
            weighted_rms_ratio=float(np.max(weighted_rms)),
            weighted_linf_ratio=float(np.max(weighted_linf)),
            absolute_l2=float(np.max(difference_l2)),
            absolute_linf=maximum_absolute,
            fine_l2_norm=float(np.max(fine_l2)),
            coarse_l2_norm=float(np.max(coarse_l2)),
            maximum_fine_amplitude=float(np.max(fine_linf)),
            maximum_coarse_amplitude=float(np.max(coarse_linf)),
        ),
        PerSystemError(
            ratios=ratios,
            weighted_rms_ratios=weighted_rms,
            weighted_linf_ratios=weighted_linf,
            fine_l2_norms=fine_l2,
            coarse_l2_norms=coarse_l2,
            difference_l2_norms=difference_l2,
            fine_linf_amplitudes=fine_linf,
            coarse_linf_amplitudes=coarse_linf,
            active_fine=active_fine,
            active_coarse=active_coarse,
            active_set_mismatch=active_mismatch,
        ),
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
        linf_relative_tolerance=(
            config.linf_relative_tolerance_factor * config.stress_relative_tolerance
        ),
    )
    reactions = _error(
        fine.reaction_forces,
        coarse.reaction_forces,
        relative_tolerance=config.reaction_relative_tolerance,
        absolute_tolerance=config.reaction_absolute_tolerance,
        linf_relative_tolerance=(
            config.linf_relative_tolerance_factor * config.reaction_relative_tolerance
        ),
    )
    displacement = _error(
        fine.displacement,
        coarse.displacement,
        relative_tolerance=config.displacement_relative_tolerance,
        absolute_tolerance=config.displacement_absolute_tolerance,
        linf_relative_tolerance=(
            config.linf_relative_tolerance_factor
            * config.displacement_relative_tolerance
        ),
    )
    signed_slip, signed_details = _per_system_error(
        fine.plastic_slip,
        coarse.plastic_slip,
        relative_tolerance=config.signed_slip_relative_tolerance,
        absolute_tolerance=config.signed_slip_absolute_tolerance,
        linf_relative_tolerance=(
            config.linf_relative_tolerance_factor * config.signed_slip_relative_tolerance
        ),
        activity_threshold=config.activity_threshold,
    )
    equivalent_plastic_slip, equivalent_details = _per_system_error(
        fine.equivalent_plastic_slip,
        coarse.equivalent_plastic_slip,
        relative_tolerance=config.accumulated_slip_relative_tolerance,
        absolute_tolerance=config.accumulated_slip_absolute_tolerance,
        linf_relative_tolerance=(
            config.linf_relative_tolerance_factor
            * config.accumulated_slip_relative_tolerance
        ),
        activity_threshold=config.activity_threshold,
    )
    candidates: list[tuple[str, float, int | None]] = [
        ("stress", stress.ratio, None),
        ("reactions", reactions.ratio, None),
        ("displacement", displacement.ratio, None),
        ("signed_slip", signed_slip.ratio, int(np.argmax(signed_details.ratios))),
        (
            "equivalent_plastic_slip",
            equivalent_plastic_slip.ratio,
            int(np.argmax(equivalent_details.ratios)),
        ),
    ]
    controlling_quantity, maximum_ratio, controlling_system = max(
        candidates, key=lambda item: item[1]
    )
    return StepErrorEstimate(
        stress=stress,
        reactions=reactions,
        displacement=displacement,
        signed_slip=signed_slip,
        accumulated_slip=equivalent_plastic_slip,
        signed_slip_ratio_per_system=signed_details.ratios,
        accumulated_slip_ratio_per_system=equivalent_details.ratios,
        signed_slip_details=signed_details,
        accumulated_slip_details=equivalent_details,
        equivalent_plastic_slip=equivalent_plastic_slip,
        equivalent_plastic_slip_ratio_per_system=equivalent_details.ratios,
        equivalent_plastic_slip_details=equivalent_details,
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
