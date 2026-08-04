"""Small, transaction-free Anderson accelerator for spectral fixed points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AndersonDiagnostics:
    proposals: int
    accelerated_proposals: int
    resets: int


class DisplacementAndersonAccelerator:
    """Limited-memory Anderson proposal generator.

    The accelerator stores only accepted fixed-point iterates supplied through
    ``propose``.  It never evaluates or commits a constitutive state; the
    caller owns acceptance, relaxation and reset policy.
    """

    def __init__(self, memory: int = 4, regularization: float = 1.0e-12) -> None:
        if memory < 1:
            raise ValueError("Anderson memory must be positive")
        if regularization < 0.0:
            raise ValueError("Anderson regularization must be nonnegative")
        self.memory = int(memory)
        self.regularization = float(regularization)
        self._states: list[FloatArray] = []
        self._images: list[FloatArray] = []
        self._residuals: list[FloatArray] = []
        self._proposals = 0
        self._accelerated_proposals = 0
        self._resets = 0

    def reset(self) -> None:
        self._states.clear()
        self._images.clear()
        self._residuals.clear()
        self._resets += 1

    def propose(
        self,
        state: ArrayLike,
        fixed_point_image: ArrayLike,
        residual: ArrayLike,
    ) -> FloatArray:
        """Return an unrelaxed Anderson candidate for the current iterate."""

        x = np.asarray(state, dtype=np.float64).reshape(-1)
        image = np.asarray(fixed_point_image, dtype=np.float64).reshape(-1)
        current_residual = np.asarray(residual, dtype=np.float64).reshape(-1)
        if not (x.size == image.size == current_residual.size):
            raise ValueError("Anderson state, image and residual sizes must match")
        if not (
            np.isfinite(x).all()
            and np.isfinite(image).all()
            and np.isfinite(current_residual).all()
        ):
            raise ValueError("Anderson inputs must be finite")

        self._proposals += 1
        candidate = image.copy()
        if self._residuals:
            residual_history = [*self._residuals, current_residual]
            image_history = [*self._images, image]
            first_pair = max(0, len(residual_history) - self.memory)
            delta_residuals = np.column_stack(
                [
                    residual_history[index + 1] - residual_history[index]
                    for index in range(first_pair, len(residual_history) - 1)
                ]
            )
            delta_images = np.column_stack(
                [
                    image_history[index + 1] - image_history[index]
                    for index in range(first_pair, len(image_history) - 1)
                ]
            )
            left, singular_values, right_transpose = np.linalg.svd(
                delta_residuals, full_matrices=False
            )
            threshold = self.regularization * max(float(singular_values[0]), 1.0)
            active = singular_values > threshold
            if not np.any(active):
                coefficients = np.zeros(delta_residuals.shape[1])
            else:
                coefficients = right_transpose[active].T @ (
                    (left[:, active].T @ current_residual) / singular_values[active]
                )
            candidate = image - delta_images @ coefficients
            self._accelerated_proposals += 1

        self._states.append(x.copy())
        self._images.append(image.copy())
        self._residuals.append(current_residual.copy())
        del self._states[: -self.memory]
        del self._images[: -self.memory]
        del self._residuals[: -self.memory]
        return candidate.reshape(np.asarray(fixed_point_image).shape)

    @property
    def diagnostics(self) -> AndersonDiagnostics:
        return AndersonDiagnostics(
            proposals=self._proposals,
            accelerated_proposals=self._accelerated_proposals,
            resets=self._resets,
        )
