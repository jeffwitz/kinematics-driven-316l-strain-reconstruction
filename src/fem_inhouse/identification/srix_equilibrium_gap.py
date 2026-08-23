"""Reconditioned weak-equilibrium identification for the FCC SRIX law.

The measured displacement history is differentiated once with the qualified
TRI2 kinematics.  Each parameter evaluation then consists only of a causal
local constitutive replay, weak residual assembly, one factorised elastic
solve per scored state, and the qualified observation/whitening actions.  It
contains no nonlinear global mechanical solve.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.plane_stress_material import (
    PlaneStressMaterialBatch,
    evaluate_in_plane_response,
)
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)

FloatArray = NDArray[np.float64]
MaterialFactory = Callable[[Mapping[str, float]], PlaneStressMaterialBatch]
THETA4_NAMES = ("tau0_mpa", "R_mpa", "Q_mpa", "b")


@dataclass(frozen=True, slots=True)
class SrixTheta4:
    """The first four positive SRIX parameters, in physical units."""

    tau0_mpa: float
    r_mpa: float
    q_mpa: float
    b: float

    def __post_init__(self) -> None:
        values = self.as_array()
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("all theta4 parameters must be finite and strictly positive")

    def as_array(self) -> FloatArray:
        return np.asarray(
            (self.tau0_mpa, self.r_mpa, self.q_mpa, self.b), dtype=np.float64
        )

    def as_runtime_overrides(self) -> dict[str, float]:
        """Return names accepted by ``resolve_srix_parameters``."""

        return {
            "tau0_mpa": self.tau0_mpa,
            "R_mpa": self.r_mpa,
            "Q_mpa": self.q_mpa,
            "b": self.b,
        }

    def log_coordinates(self) -> FloatArray:
        return np.log(self.as_array())

    @classmethod
    def from_log_coordinates(cls, eta: ArrayLike) -> SrixTheta4:
        values = np.asarray(eta, dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("eta must contain four finite log parameters")
        physical = np.exp(values)
        return cls(
            tau0_mpa=float(physical[0]),
            r_mpa=float(physical[1]),
            q_mpa=float(physical[2]),
            b=float(physical[3]),
        )


@dataclass(frozen=True, slots=True)
class EquilibriumGapTiming:
    material_seconds: float
    weak_residual_seconds: float
    reconditioner_seconds: float
    observation_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class EquilibriumGapState:
    state_index: int
    scored: bool
    raw_equilibrium_norm: float
    pseudo_displacement_norm: float
    relative_pseudo_displacement_norm: float
    whitened_residual_rms: float | None
    stress_in_plane_mpa: FloatArray | None = None
    weak_residual: FloatArray | None = None
    pseudo_displacement: FloatArray | None = None


@dataclass(frozen=True, slots=True)
class EquilibriumGapEvaluation:
    theta: SrixTheta4
    residual_vector: FloatArray
    cost: float
    residual_rms: float
    states: tuple[EquilibriumGapState, ...]
    timing: EquilibriumGapTiming
    material_evaluations: int


@dataclass(frozen=True, slots=True)
class SensitivitySVD:
    singular_values: FloatArray
    normalized_singular_values: FloatArray
    right_singular_vectors: FloatArray
    numerical_rank: int
    condition_number: float
    relative_threshold: float


class SrixEquilibriumGapProblem:
    """Causal local SRIX replay followed by elastic reconditioning."""

    def __init__(
        self,
        *,
        operator: TensorPlasticObservabilityOperator,
        displacement_history: ArrayLike,
        state_indices: Sequence[int],
        scored_states: set[int],
        material_factory: MaterialFactory,
        time_increments: ArrayLike | None = None,
        debug: bool = False,
    ) -> None:
        history = np.asarray(displacement_history, dtype=np.float64)
        expected_tail = (*operator.grid.node_shape, 2)
        if history.ndim != 4 or history.shape[1:] != expected_tail:
            raise ValueError(
                f"displacement_history must have shape (states, {expected_tail})"
            )
        if history.shape[0] < 2 or not np.isfinite(history).all():
            raise ValueError("displacement_history must contain finite initial and final states")
        indices = tuple(int(value) for value in state_indices)
        if len(indices) != history.shape[0] - 1 or len(set(indices)) != len(indices):
            raise ValueError("state_indices must uniquely label every replayed increment")
        if not scored_states.issubset(indices):
            raise ValueError("scored_states must be a subset of state_indices")
        if time_increments is None:
            increments = np.ones(len(indices), dtype=np.float64)
        else:
            increments = np.asarray(time_increments, dtype=np.float64)
            if increments.shape != (len(indices),):
                raise ValueError("time_increments must have one entry per replayed state")
        if not np.isfinite(increments).all() or np.any(increments <= 0.0):
            raise ValueError("time increments must be finite and strictly positive")

        self.operator = operator
        self.displacement_history = history.copy()
        self.state_indices = indices
        self.scored_states = frozenset(scored_states)
        self.material_factory = material_factory
        self.time_increments = increments.copy()
        self.debug = bool(debug)
        # The expensive differentiation of measured displacement is immutable
        # with respect to theta and is therefore cached once.
        self.strain_history = np.stack(
            [operator.kinematics.strain(history[index]) for index in range(1, history.shape[0])]
        )

    def replay(self, theta: SrixTheta4) -> EquilibriumGapEvaluation:
        started_total = time.perf_counter()
        material = self.material_factory(theta.as_runtime_overrides())
        if material.point_count != self.operator.kinematics.material_point_count:
            raise ValueError("material point count does not match the TRI2 kinematics")
        residual_blocks: list[FloatArray] = []
        state_records: list[EquilibriumGapState] = []
        material_seconds = 0.0
        weak_seconds = 0.0
        reconditioner_seconds = 0.0
        observation_seconds = 0.0

        try:
            for offset, (state, time_increment) in enumerate(
                zip(self.state_indices, self.time_increments, strict=True)
            ):
                strain = self.strain_history[offset]
                material_started = time.perf_counter()
                trial = evaluate_in_plane_response(
                    material,
                    strain.reshape(-1, 3),
                    time_increment=float(time_increment),
                    response_level="residual",
                    consistent_tangent=False,
                )
                material.commit()
                material_seconds += time.perf_counter() - material_started

                stress = np.asarray(trial.stress_in_plane_mpa, dtype=np.float64).reshape(
                    *self.operator.grid.pixel_shape, 2, 3
                )
                weak_started = time.perf_counter()
                weak = self.operator.weak_equilibrium_residual(stress)
                weak_seconds += time.perf_counter() - weak_started

                reconditioner_started = time.perf_counter()
                correction = self.operator.correction_from_weak_residual(weak)
                reconditioner_seconds += time.perf_counter() - reconditioner_started

                scored = state in self.scored_states
                whitened: FloatArray | None = None
                if scored:
                    observation_started = time.perf_counter()
                    observed = self.operator.transfer.apply(correction)
                    whitened = np.asarray(
                        self.operator.whitener.apply(observed), dtype=np.float64
                    )
                    observation_seconds += time.perf_counter() - observation_started
                    residual_blocks.append(whitened.reshape(-1))

                measured_norm = float(np.linalg.norm(self.displacement_history[offset + 1]))
                correction_norm = float(np.linalg.norm(correction))
                state_records.append(
                    EquilibriumGapState(
                        state_index=state,
                        scored=scored,
                        raw_equilibrium_norm=float(np.linalg.norm(weak)),
                        pseudo_displacement_norm=correction_norm,
                        relative_pseudo_displacement_norm=correction_norm
                        / max(measured_norm, np.finfo(np.float64).tiny),
                        whitened_residual_rms=(
                            None
                            if whitened is None
                            else float(np.sqrt(np.mean(whitened**2)))
                        ),
                        stress_in_plane_mpa=stress.copy() if self.debug else None,
                        weak_residual=weak.copy() if self.debug else None,
                        pseudo_displacement=correction.copy() if self.debug else None,
                    )
                )
        except Exception:
            material.revert()
            raise

        residual = (
            np.concatenate(residual_blocks)
            if residual_blocks
            else np.empty(0, dtype=np.float64)
        )
        cost = 0.5 * float(residual @ residual)
        total_seconds = time.perf_counter() - started_total
        return EquilibriumGapEvaluation(
            theta=theta,
            residual_vector=residual,
            cost=cost,
            residual_rms=(
                0.0 if residual.size == 0 else float(np.sqrt(np.mean(residual**2)))
            ),
            states=tuple(state_records),
            timing=EquilibriumGapTiming(
                material_seconds=material_seconds,
                weak_residual_seconds=weak_seconds,
                reconditioner_seconds=reconditioner_seconds,
                observation_seconds=observation_seconds,
                total_seconds=total_seconds,
            ),
            material_evaluations=len(self.state_indices),
        )

    def evaluate(self, theta: SrixTheta4) -> EquilibriumGapEvaluation:
        return self.replay(theta)

    def residual_vector(self, theta: SrixTheta4) -> FloatArray:
        return self.replay(theta).residual_vector

    def jacobian_fd(self, eta: ArrayLike, *, relative_step: float) -> FloatArray:
        """Central finite-difference Jacobian of the full residual vector."""

        coordinates = np.asarray(eta, dtype=np.float64)
        if coordinates.shape != (4,) or not np.isfinite(coordinates).all():
            raise ValueError("eta must contain four finite log parameters")
        if not np.isfinite(relative_step) or relative_step <= 0.0:
            raise ValueError("relative_step must be finite and positive")
        columns: list[FloatArray] = []
        for index in range(4):
            plus = coordinates.copy()
            minus = coordinates.copy()
            plus[index] += relative_step
            minus[index] -= relative_step
            columns.append(
                (
                    self.residual_vector(SrixTheta4.from_log_coordinates(plus))
                    - self.residual_vector(SrixTheta4.from_log_coordinates(minus))
                )
                / (2.0 * relative_step)
            )
        if not columns:
            return np.empty((0, 4), dtype=np.float64)
        return np.column_stack(columns)

    @staticmethod
    def sensitivity_svd(
        jacobian: ArrayLike, *, relative_threshold: float = 1.0e-8
    ) -> SensitivitySVD:
        matrix = np.asarray(jacobian, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != 4 or not np.isfinite(matrix).all():
            raise ValueError("jacobian must have finite shape (residuals, 4)")
        if not 0.0 < relative_threshold < 1.0:
            raise ValueError("relative_threshold must lie strictly between zero and one")
        _, singular, right_transposed = np.linalg.svd(matrix, full_matrices=False)
        if singular.size == 0 or singular[0] == 0.0:
            normalized = np.zeros_like(singular)
            rank = 0
            condition = float("inf")
        else:
            normalized = singular / singular[0]
            rank = int(np.count_nonzero(normalized > relative_threshold))
            condition = (
                float(singular[0] / singular[rank - 1]) if rank else float("inf")
            )
        return SensitivitySVD(
            singular_values=singular,
            normalized_singular_values=normalized,
            right_singular_vectors=right_transposed.T,
            numerical_rank=rank,
            condition_number=condition,
            relative_threshold=relative_threshold,
        )
