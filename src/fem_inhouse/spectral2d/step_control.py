"""Adaptive load-step decisions for rate-independent spectral solves.

The controller is deliberately independent of constitutive integration.  It
only consumes the outcome of an already transactional increment attempt.  The
solver can therefore qualify the policy before making it responsible for
material rollback.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdaptiveStepConfig:
    """Policy parameters for a rate-independent load path."""

    initial_increment_fraction: float = 0.25
    minimum_increment_fraction: float = 1.0 / 256.0
    maximum_increment_fraction: float = 0.5
    increment_growth_factor: float = 1.5
    increment_cutback_factor: float = 0.5
    target_newton_iterations_min: int = 4
    target_newton_iterations_max: int = 7
    maximum_cutbacks_per_step: int = 8

    def __post_init__(self) -> None:
        if self.initial_increment_fraction <= 0.0:
            raise ValueError("initial increment fraction must be positive")
        if self.minimum_increment_fraction <= 0.0:
            raise ValueError("minimum increment fraction must be positive")
        if self.maximum_increment_fraction < self.minimum_increment_fraction:
            raise ValueError("maximum increment fraction must not be below minimum")
        if self.initial_increment_fraction > self.maximum_increment_fraction:
            raise ValueError("initial increment fraction must not exceed maximum")
        if self.increment_growth_factor <= 1.0:
            raise ValueError("increment growth factor must be greater than one")
        if not 0.0 < self.increment_cutback_factor < 1.0:
            raise ValueError("increment cutback factor must be in (0, 1)")
        if self.target_newton_iterations_min < 1:
            raise ValueError("minimum target Newton iterations must be positive")
        if self.target_newton_iterations_max < self.target_newton_iterations_min:
            raise ValueError("maximum target Newton iterations must not be below minimum")
        if self.maximum_cutbacks_per_step < 0:
            raise ValueError("maximum cutbacks per step must be non-negative")


@dataclass(frozen=True, slots=True)
class LoadStepObservation:
    """Summary of one completed load-step attempt."""

    converged: bool
    newton_iterations: int
    minimum_line_search_factor: float = 1.0
    maximum_local_iterations: int = 0


@dataclass(frozen=True, slots=True)
class LoadStepDecision:
    """Decision returned after an accepted or rejected attempt."""

    accepted: bool
    next_increment_fraction: float
    reason: str
    cutbacks_for_current_step: int


class AdaptiveLoadStepController:
    """Stateful growth/cutback policy with mandatory path-node support."""

    def __init__(self, config: AdaptiveStepConfig) -> None:
        self.config = config
        self.increment_fraction = config.initial_increment_fraction
        self.cutbacks_for_current_step = 0

    def propose(self, current_fraction: float, segment_end: float = 1.0) -> float:
        """Return the next path fraction without crossing ``segment_end``."""

        if not 0.0 <= current_fraction <= segment_end <= 1.0:
            raise ValueError("path fractions must satisfy 0 <= current <= end <= 1")
        return min(segment_end, current_fraction + self.increment_fraction)

    def accept(self, observation: LoadStepObservation) -> LoadStepDecision:
        """Record an accepted attempt and select the next step size."""

        if not observation.converged:
            raise ValueError("an accepted decision requires a converged observation")
        difficult = (
            observation.newton_iterations > self.config.target_newton_iterations_max
            or observation.minimum_line_search_factor < 1.0
            or observation.maximum_local_iterations > 0
        )
        easy = (
            observation.newton_iterations <= self.config.target_newton_iterations_min
            and observation.minimum_line_search_factor >= 1.0
            and observation.maximum_local_iterations == 0
        )
        if difficult:
            self.increment_fraction = max(
                self.config.minimum_increment_fraction,
                self.increment_fraction * self.config.increment_cutback_factor,
            )
            reason = "accepted_difficult_step"
        elif easy:
            self.increment_fraction = min(
                self.config.maximum_increment_fraction,
                self.increment_fraction * self.config.increment_growth_factor,
            )
            reason = "accepted_easy_step"
        else:
            reason = "accepted_normal_step"
        self.cutbacks_for_current_step = 0
        return LoadStepDecision(
            accepted=True,
            next_increment_fraction=self.increment_fraction,
            reason=reason,
            cutbacks_for_current_step=0,
        )

    def reject(self, reason: str = "attempt_failed") -> LoadStepDecision:
        """Reject an attempt and return a smaller retry step."""

        self.cutbacks_for_current_step += 1
        if self.cutbacks_for_current_step > self.config.maximum_cutbacks_per_step:
            raise RuntimeError("maximum cutbacks per load step exceeded")
        self.increment_fraction = max(
            self.config.minimum_increment_fraction,
            self.increment_fraction * self.config.increment_cutback_factor,
        )
        return LoadStepDecision(
            accepted=False,
            next_increment_fraction=self.increment_fraction,
            reason=reason,
            cutbacks_for_current_step=self.cutbacks_for_current_step,
        )
