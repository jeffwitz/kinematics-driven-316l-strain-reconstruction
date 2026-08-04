"""Safeguarded global multisecant direction for the nonlinear solve."""

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
        self._used = 0
        self._rejected = 0

    def begin_increment(self) -> None:
        self._previous_state = None
        self._previous_residual = None
        self._steps.clear()
        self._residual_changes.clear()

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
        solve_columns: Callable[[FloatArray], FloatArray],
    ) -> FloatArray:
        """Apply the multisecant inverse update around the current Newton solve."""

        base = np.asarray(base_direction, dtype=np.float64).reshape(-1)
        rhs = -np.asarray(residual, dtype=np.float64).reshape(-1)
        if not self._steps or not np.isfinite(base).all() or not np.isfinite(rhs).all():
            return base.copy()
        steps = np.column_stack(self._steps)
        residual_changes = np.column_stack(self._residual_changes)
        coefficients, *_ = np.linalg.lstsq(residual_changes, rhs, rcond=1e-10)
        base_residual_solutions = np.asarray(solve_columns(residual_changes), dtype=np.float64)
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
            self._rejected += 1
            return base.copy()
        self._used += 1
        return candidate

    @property
    def diagnostics(self) -> dict[str, float]:
        return {
            "global_broyden_memory": float(self.memory),
            "global_broyden_pairs_accepted": float(self._accepted),
            "global_broyden_directions_used": float(self._used),
            "global_broyden_directions_rejected": float(self._rejected),
        }
