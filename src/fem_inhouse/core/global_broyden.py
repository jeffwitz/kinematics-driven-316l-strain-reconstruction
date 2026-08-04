"""Safeguarded global multisecant direction for the nonlinear solve.

Status: ``qualified_negative_result``. On the registered SRIX case the
correction is transparent but does not reduce Newton iterations; it remains
available for reproducibility and is disabled by default.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class GlobalInverseBroyden:
    """Limited-memory global multisecant accelerator with hard safeguards."""

    name = "global_broyden"

    def __init__(self, memory: int = 3, maximum_step_factor: float = 2.0) -> None:
        if not 1 <= memory <= 5:
            raise ValueError("global Broyden memory must lie in 1..5")
        if maximum_step_factor <= 1.0:
            raise ValueError("maximum_step_factor must be greater than one")
        self.memory = int(memory)
        self.maximum_step_factor = float(maximum_step_factor)
        self._previous_state: FloatArray | None = None
        self._previous_residual: FloatArray | None = None
        self._steps: list[FloatArray] = []
        self._residual_changes: list[FloatArray] = []
        self._accepted = 0
        self._proposed = 0
        self._accepted_by_safeguards = 0
        self._used_at_full_step = 0
        self._rejected_by_norm = 0
        self._rejected_by_descent = 0
        self._rejected_by_line_search = 0
        self._newton_fallbacks = 0
        self._last_direction_proposed = False

    def begin_increment(self) -> None:
        self._previous_state = None
        self._previous_residual = None
        self._steps.clear()
        self._residual_changes.clear()
        self._last_direction_proposed = False

    def discard(self) -> None:
        self.begin_increment()

    def observe(self, state: ArrayLike, residual: ArrayLike) -> None:
        current_state = np.asarray(state, dtype=np.float64).reshape(-1).copy()
        current_residual = np.asarray(residual, dtype=np.float64).reshape(-1).copy()
        if current_state.size != current_residual.size:
            raise ValueError("global Broyden state and residual must have equal size")
        if self._previous_state is not None and self._previous_residual is not None:
            step = current_state - self._previous_state
            residual_change = current_residual - self._previous_residual
            step_norm = np.linalg.norm(step)
            change_norm = np.linalg.norm(residual_change)
            if (
                np.isfinite(step_norm)
                and np.isfinite(change_norm)
                and step_norm > 1e-14
                and change_norm > 1e-14
            ):
                self._steps.append(step)
                self._residual_changes.append(residual_change)
                del self._steps[:-self.memory]
                del self._residual_changes[:-self.memory]
                self._accepted += 1
        self._previous_state = current_state
        self._previous_residual = current_residual

    def direction(
        self,
        base_direction: ArrayLike,
        residual: ArrayLike,
        solve_columns: Callable[[FloatArray], FloatArray] | None = None,
        base_residual_solutions: ArrayLike | None = None,
    ) -> FloatArray:
        """Apply the multisecant inverse update around the current Newton solve."""

        base = np.asarray(base_direction, dtype=np.float64).reshape(-1)
        rhs = -np.asarray(residual, dtype=np.float64).reshape(-1)
        self._last_direction_proposed = False
        if not self._steps or not np.isfinite(base).all() or not np.isfinite(rhs).all():
            return base.copy()
        steps = np.column_stack(self._steps)
        residual_changes = np.column_stack(self._residual_changes)
        coefficients, *_ = np.linalg.lstsq(residual_changes, rhs, rcond=1e-10)
        if base_residual_solutions is None:
            if solve_columns is None:
                raise ValueError("base residual solves are required")
            base_residual_solutions = solve_columns(residual_changes)
        base_residual_solutions = np.asarray(base_residual_solutions, dtype=np.float64)
        if base_residual_solutions.shape != steps.shape:
            raise ValueError("global Broyden base solves have an incompatible shape")
        candidate = base + (steps - base_residual_solutions) @ coefficients
        base_norm = float(np.linalg.norm(base))
        candidate_norm = float(np.linalg.norm(candidate))
        if (
            not np.isfinite(candidate).all()
            or base_norm == 0.0
            or candidate_norm > self.maximum_step_factor * base_norm
        ):
            self._rejected_by_norm += 1
            return base.copy()
        self._proposed += 1
        self._last_direction_proposed = True
        return candidate

    def residual_change_matrix(self) -> FloatArray | None:
        """Return the active residual-change columns for a block solve."""

        if not self._residual_changes:
            return None
        return np.column_stack(self._residual_changes)

    def reject(self) -> None:
        """Record a descent rejection for compatibility with older callers."""

        self.reject_descent()

    @property
    def last_direction_proposed(self) -> bool:
        return self._last_direction_proposed

    def accept(self) -> None:
        """Record a direction that passed the solver safeguards."""

        self._accepted_by_safeguards += 1

    def mark_full_step(self) -> None:
        """Record a safeguarded correction accepted without damping."""

        self._used_at_full_step += 1

    def reject_descent(self) -> None:
        """Record rejection by the predicted-descent safeguard."""

        self._rejected_by_descent += 1

    def reject_line_search(self) -> None:
        """Record rejection by Armijo line search."""

        self._rejected_by_line_search += 1

    def fallback_to_newton(self) -> None:
        """Record a Newton fallback after a proposed correction."""

        self._newton_fallbacks += 1

    @property
    def diagnostics(self) -> dict[str, float]:
        return {
            "global_broyden_memory": float(self.memory),
            "global_broyden_pairs_accepted": float(self._accepted),
            "global_broyden_directions_proposed": float(self._proposed),
            "global_broyden_directions_accepted_by_safeguards": float(
                self._accepted_by_safeguards
            ),
            "global_broyden_directions_used_at_full_step": float(self._used_at_full_step),
            "global_broyden_directions_rejected_by_norm": float(self._rejected_by_norm),
            "global_broyden_directions_rejected_by_descent": float(
                self._rejected_by_descent
            ),
            "global_broyden_directions_rejected_by_line_search": float(
                self._rejected_by_line_search
            ),
            "global_broyden_newton_fallbacks": float(self._newton_fallbacks),
        }
