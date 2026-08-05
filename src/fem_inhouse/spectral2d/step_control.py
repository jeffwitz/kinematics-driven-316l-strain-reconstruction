"""Adaptive load-step decisions for rate-independent spectral solves.

The controller is deliberately independent of constitutive integration.  It
only consumes the outcome of an already transactional increment attempt.  The
solver can therefore qualify the policy before making it responsible for
material rollback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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


@dataclass(frozen=True, slots=True)
class LoadPathStep:
    """One proposed interval on a normalized boundary path."""

    index: int
    start_fraction: float
    end_fraction: float
    boundary: NDArray[np.float64]
    time_increment: float


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


class AdaptiveLoadPath:
    """Iterator that couples a normalized boundary path to the controller."""

    def __init__(self, history: NDArray[np.float64], config: AdaptiveStepConfig) -> None:
        if history.ndim != 4 or history.shape[0] < 2:
            raise ValueError("adaptive history must have shape (n, nx, ny, 2)")
        if not np.allclose(history[0], 0.0):
            raise ValueError("adaptive history must start at zero")
        self.history = np.asarray(history, dtype=np.float64)
        self.controller = AdaptiveLoadStepController(config)
        self.current_fraction = 0.0
        self._pending: LoadPathStep | None = None
        self._index = 0
        fractions = np.linspace(0.0, 1.0, history.shape[0])[:, None, None, None]
        self._proportional = bool(
            np.allclose(history, fractions * history[-1], rtol=1.0e-12, atol=1.0e-15)
        )

    def __iter__(self) -> AdaptiveLoadPath:
        return self

    @property
    def pending(self) -> LoadPathStep | None:
        return self._pending

    def __next__(self) -> LoadPathStep:
        if self._pending is not None:
            raise RuntimeError("accept or reject the pending adaptive load step first")
        if self.current_fraction >= 1.0 - 1.0e-14:
            raise StopIteration
        segment_end = 1.0 if self._proportional else self._next_required_knot()
        end_fraction = self.controller.propose(self.current_fraction, segment_end)
        boundary = self._interpolate(end_fraction)
        self._index += 1
        self._pending = LoadPathStep(
            index=self._index,
            start_fraction=self.current_fraction,
            end_fraction=end_fraction,
            boundary=boundary,
            time_increment=end_fraction - self.current_fraction,
        )
        return self._pending

    def accept(self, observation: LoadStepObservation) -> LoadStepDecision:
        if self._pending is None:
            raise RuntimeError("no adaptive load step is pending")
        pending = self._pending
        decision = self.controller.accept(observation)
        self.current_fraction = pending.end_fraction
        self._pending = None
        return decision

    def reject(self, reason: str = "attempt_failed") -> LoadStepDecision:
        if self._pending is None:
            raise RuntimeError("no adaptive load step is pending")
        decision = self.controller.reject(reason)
        self._pending = None
        self._index -= 1
        return decision

    def _next_required_knot(self) -> float:
        count = self.history.shape[0] - 1
        next_index = int(np.floor(self.current_fraction * count + 1.0e-12)) + 1
        return min(1.0, next_index / count)

    def _interpolate(self, fraction: float) -> NDArray[np.float64]:
        position = fraction * (self.history.shape[0] - 1)
        lower = min(int(np.floor(position)), self.history.shape[0] - 2)
        weight = position - lower
        return ((1.0 - weight) * self.history[lower] + weight * self.history[lower + 1]).copy()
