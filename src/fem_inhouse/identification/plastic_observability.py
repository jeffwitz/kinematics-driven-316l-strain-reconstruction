"""Matrix-free observability operators for the experimental plastic oracle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator, eigsh, gmres

from fem_inhouse.core.constitutive import PLANE_STRESS_VON_MISES_METRIC, von_mises
from fem_inhouse.identification.dic_whitening import DICSpectralTransfer, DICSpectralWhitener
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior
from fem_inhouse.workflows.experimental_mechanical_oracle import (
    ExperimentalMechanicalOracleLinearisation,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PlasticMetric:
    """Mesh-normalized amplitude/spatial metric for plastic perturbations.

    ``reference_scale`` is the documented RMS plastic-increment scale. The
    amplitude term is normalized by the number of plastic unknowns, so its
    quadratic form is a mean-square relative perturbation.
    """

    amplitude_weight: float = 1.0
    spatial_weight: float = 0.0
    reference_scale: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.amplitude_weight) or self.amplitude_weight <= 0.0:
            raise ValueError("amplitude_weight must be finite and positive")
        if not np.isfinite(self.spatial_weight) or self.spatial_weight < 0.0:
            raise ValueError("spatial_weight must be finite and non-negative")
        if not np.isfinite(self.reference_scale) or self.reference_scale <= 0.0:
            raise ValueError("reference_scale must be finite and positive")

    def action(self, plastic: ArrayLike) -> FloatArray:
        values = np.asarray(plastic, dtype=np.float64)
        normalization = values.size * self.reference_scale**2
        result = self.amplitude_weight * values / normalization
        if self.spatial_weight == 0.0:
            return result
        for axis in (0, 1):
            difference = np.diff(values, axis=axis)
            if difference.size == 0:
                continue
            contribution = difference / (difference.size * self.reference_scale**2)
            lower = [slice(None)] * values.ndim
            upper = [slice(None)] * values.ndim
            lower[axis] = slice(0, -1)
            upper[axis] = slice(1, None)
            result[tuple(lower)] -= self.spatial_weight * contribution
            result[tuple(upper)] += self.spatial_weight * contribution
        return result


@dataclass(frozen=True, slots=True)
class DirectionMetric:
    """J2 metric for raw flow-direction perturbations."""

    states: tuple[PlasticObservabilityState, ...]
    null_regularisation: float = 1.0e-10

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("DirectionMetric requires at least one state")
        if not np.isfinite(self.null_regularisation) or self.null_regularisation <= 0.0:
            raise ValueError("null_regularisation must be finite and positive")

    def action(self, direction: ArrayLike) -> FloatArray:
        value = np.asarray(direction, dtype=np.float64)
        if value.ndim != 4 or value.shape[-1] != 3:
            raise ValueError("direction must have shape (nx, ny, 2, 3)")
        inverse_metric = np.linalg.inv(PLANE_STRESS_VON_MISES_METRIC)
        result = self.null_regularisation * value / value.size
        total_weight = sum(state.weight for state in self.states)
        normalisation = total_weight if total_weight > 0.0 else 1.0
        for state in self.states:
            trial = state.linearisation.trial
            stress = trial.stress_in_plane_mpa
            q = von_mises(stress)
            safe = np.where(q > 0.0, q, 1.0)
            flow = np.asarray(trial.observables["flow_direction"], dtype=np.float64)
            raw = value.reshape(-1, 3)
            tangent = raw - flow * (np.einsum("pi,pi->p", stress, raw) / safe)[:, None]
            tangent[q <= 0.0] = 0.0
            weighted = np.einsum("ij,pj->pi", inverse_metric, tangent)
            result += (state.weight / normalisation) * weighted.reshape(value.shape) / value.size
        return result


@dataclass(frozen=True, slots=True)
class PlasticObservabilityState:
    """One mechanically equilibrated linearisation used for mode construction."""

    linearisation: ExperimentalMechanicalOracleLinearisation
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("state weight must be finite and non-negative")


class PlasticObservabilityOperator:
    """Apply ``G_p``, ``O = -W K^{-1} G_p`` and their adjoints.

    The displacement remains full-field. Only the plastic increment is supplied
    as an operator vector; no global matrix is assembled.
    """

    def __init__(
        self,
        states: tuple[PlasticObservabilityState, ...],
        grid: StructuredGrid2D,
        whitener: DICSpectralWhitener,
        *,
        transfer: DICSpectralTransfer | None = None,
        normalize_state_weights: bool = True,
        gmres_rtol: float = 1.0e-8,
        gmres_maxiter: int = 500,
    ) -> None:
        if not states:
            raise ValueError("at least one observability state is required")
        if not 0.0 < gmres_rtol < 1.0:
            raise ValueError("gmres_rtol must lie in (0, 1)")
        self.states = states
        self.grid = grid
        self.whitener = whitener
        self.transfer = transfer
        self.normalize_state_weights = normalize_state_weights
        self.gmres_rtol = gmres_rtol
        self.gmres_maxiter = gmres_maxiter
        displacement_shape = states[0].linearisation.displacement_shape
        plastic_shape = states[0].linearisation.plastic_increment_shape
        if displacement_shape != whitener.field_shape:
            raise ValueError("whitener shape must match the displacement field")
        if any(
            state.linearisation.displacement_shape != displacement_shape
            or state.linearisation.plastic_increment_shape != plastic_shape
            for state in states
        ):
            raise ValueError("all observability states must have identical shapes")
        self.displacement_shape = displacement_shape
        self.plastic_shape = plastic_shape
        self.plastic_size = int(np.prod(plastic_shape))
        self.displacement_size = int(np.prod(grid.interior_shape) * 2)

    def _plastic(self, value: ArrayLike) -> FloatArray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape == self.plastic_shape:
            return result
        if result.size == self.plastic_size:
            return result.reshape(self.plastic_shape)
        raise ValueError(f"plastic value must have shape {self.plastic_shape}")

    def _displacement_vector(self, value: ArrayLike) -> FloatArray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape == self.displacement_shape:
            return pack_interior(result)
        if result.size == self.displacement_size:
            return result.reshape(-1)
        raise ValueError("displacement value has an incompatible shape")

    def _displacement_field(self, value: ArrayLike) -> FloatArray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape == self.displacement_shape:
            return result
        if result.size == self.displacement_size:
            return unpack_interior(result.reshape(-1), self.grid)
        raise ValueError("displacement value has an incompatible shape")

    def _k_operator(
        self, state: PlasticObservabilityState, transpose: bool = False
    ) -> LinearOperator:
        lin = state.linearisation

        def action(value: NDArray[np.float64]) -> NDArray[np.float64]:
            field = unpack_interior(value, self.grid)
            if transpose:
                result = lin.mechanical_jacobian_transpose_action(field)
            else:
                result = lin.mechanical_jacobian_action(field)
            return pack_interior(result)

        return LinearOperator(
            (self.displacement_size, self.displacement_size),
            matvec=action,
            dtype=np.float64,
        )

    def solve_mechanical(
        self,
        state: PlasticObservabilityState,
        rhs: ArrayLike,
        *,
        transpose: bool = False,
    ) -> FloatArray:
        """Solve ``K du = rhs`` or ``K.T du = rhs`` and return a full field."""

        packed_rhs = self._displacement_vector(rhs)
        solution, info = gmres(
            self._k_operator(state, transpose=transpose),
            packed_rhs,
            rtol=self.gmres_rtol,
            atol=0.0,
            maxiter=self.gmres_maxiter,
        )
        if info != 0:
            raise RuntimeError(f"mechanical sensitivity solve failed (info={info})")
        return unpack_interior(solution, self.grid)

    def gp(self, state: PlasticObservabilityState, plastic: ArrayLike) -> FloatArray:
        """Apply ``G_p`` and return the full mechanical residual field."""

        return state.linearisation.plastic_residual_action(self._plastic(plastic))

    def gp_transpose(self, state: PlasticObservabilityState, dual: ArrayLike) -> FloatArray:
        """Apply ``G_p.T`` to a full mechanical dual field."""

        field = self._displacement_field(dual)
        return state.linearisation.plastic_residual_transpose_action(field)

    def sensitivity(self, state: PlasticObservabilityState, plastic: ArrayLike) -> FloatArray:
        """Apply ``S_p = -K^{-1}G_p``."""

        return self.solve_mechanical(state, -self.gp(state, plastic))

    def observation(self, state: PlasticObservabilityState, plastic: ArrayLike) -> FloatArray:
        """Apply ``O = W_D M_D S_p``."""

        displacement = self.sensitivity(state, plastic)
        if self.transfer is not None:
            displacement = self.transfer.apply(displacement)
        return self.whitener.apply(displacement)

    def observation_transpose(
        self, state: PlasticObservabilityState, dual: ArrayLike
    ) -> FloatArray:
        """Apply ``O.T = -G_p.T K.T^{-1} M_D.T W_D.T``."""

        whitened_dual = self.whitener.adjoint(dual)
        if self.transfer is not None:
            whitened_dual = self.transfer.adjoint(whitened_dual)
        mechanical_dual = self.solve_mechanical(state, whitened_dual, transpose=True)
        return -self.gp_transpose(state, mechanical_dual)

    def information_action(self, plastic: ArrayLike) -> FloatArray:
        """Apply the weighted information operator.

        By default the state weights are normalized to sum to one, so spectra
        describe average information per retained state rather than growing
        solely because more snapshots were selected.
        """

        value = self._plastic(plastic)
        result = np.zeros_like(value)
        total_weight = sum(state.weight for state in self.states)
        normalization = (
            total_weight if self.normalize_state_weights and total_weight > 0.0 else 1.0
        )
        for state in self.states:
            result += (state.weight / normalization) * self.observation_transpose(
                state,
                self.observation(state, value),
            )
        return result

    def information_operator(self) -> LinearOperator:
        """Return a matrix-free ``LinearOperator`` for the information action."""

        return LinearOperator(
            (self.plastic_size, self.plastic_size),
            matvec=lambda value: self.information_action(value).reshape(-1),
            dtype=np.float64,
        )

    def metric_operator(self, metric: PlasticMetric | None = None) -> LinearOperator:
        """Return the matrix-free SPD metric operator ``H_p``."""

        selected = metric or PlasticMetric()
        return LinearOperator(
            (self.plastic_size, self.plastic_size),
            matvec=lambda value: selected.action(self._plastic(value)).reshape(-1),
            dtype=np.float64,
        )

    def generalized_modes(
        self,
        rank: int,
        *,
        metric: PlasticMetric | None = None,
        tolerance: float = 1.0e-8,
        maximum_iterations: int | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute leading generalized observability modes matrix-free.

        Returns descending eigenvalues and modes whose columns are normalized
        with respect to ``H_p``. The default metric is deliberately explicit
        and amplitude-only until a calibrated spatial prior is available.
        """

        if rank < 1 or rank >= self.plastic_size:
            raise ValueError("rank must satisfy 1 <= rank < plastic_size")
        if not 0.0 < tolerance < 1.0:
            raise ValueError("tolerance must lie in (0, 1)")
        values, vectors = eigsh(
            self.information_operator(),
            k=rank,
            M=self.metric_operator(metric),
            which="LM",
            tol=tolerance,
            maxiter=maximum_iterations,
        )
        order = np.argsort(values)[::-1]
        return values[order], vectors[:, order]

    def adjoint_errors(self, *, seed: int = 20260814) -> dict[str, float]:
        """Return relative adjoint errors for ``G_p`` and ``O``."""

        rng = np.random.default_rng(seed)
        state = self.states[0]
        plastic = rng.normal(size=self.plastic_shape)
        dual = rng.normal(size=self.displacement_shape)
        gp_error = abs(
            np.vdot(self.gp(state, plastic), dual)
            - np.vdot(plastic, self.gp_transpose(state, dual))
        )
        observed = self.observation(state, plastic)
        observation_dual = rng.normal(size=observed.shape)
        observation_error = abs(
            np.vdot(observed, observation_dual)
            - np.vdot(plastic, self.observation_transpose(state, observation_dual))
        )
        gp_scale = max(float(abs(np.vdot(self.gp(state, plastic), dual))), 1.0e-30)
        observation_scale = max(
            float(abs(np.vdot(observed, observation_dual))), 1.0e-30
        )
        return {
            "gp_relative_error": float(gp_error / gp_scale),
            "observation_relative_error": float(observation_error / observation_scale),
        }


class DirectionObservabilityOperator(PlasticObservabilityOperator):
    """Observability operator for signed perturbations of the J2 flow direction.

    The input is a three-component raw direction field per material point. The
    constitutive linearisation projects it onto ``sigma.T delta_n = 0`` before
    applying the mechanical residual operator. Unlike amplitude increments,
    these variables are signed and have no ``Delta p >= 0`` constraint.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.direction_shape = (*self.plastic_shape, 3)
        self.direction_size = int(np.prod(self.direction_shape))
        self.plastic_size = self.direction_size

    def _plastic(self, value: ArrayLike) -> FloatArray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape == self.direction_shape:
            return result
        if result.size == self.direction_size:
            return result.reshape(self.direction_shape)
        raise ValueError(f"direction value must have shape {self.direction_shape}")

    def gp(self, state: PlasticObservabilityState, plastic: ArrayLike) -> FloatArray:
        return state.linearisation.direction_residual_action(self._plastic(plastic))

    def gp_transpose(self, state: PlasticObservabilityState, dual: ArrayLike) -> FloatArray:
        field = self._displacement_field(dual)
        return state.linearisation.direction_residual_transpose_action(field)

    def adjoint_errors(self, *, seed: int = 20260814) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        state = self.states[0]
        direction = rng.normal(size=self.direction_shape)
        dual = rng.normal(size=self.displacement_shape)
        gp_value = self.gp(state, direction)
        gp_error = abs(np.vdot(gp_value, dual) - np.vdot(direction, self.gp_transpose(state, dual)))
        observed = self.observation(state, direction)
        observation_dual = rng.normal(size=observed.shape)
        observation_error = abs(
            np.vdot(observed, observation_dual)
            - np.vdot(direction, self.observation_transpose(state, observation_dual))
        )
        return {
            "gp_relative_error": float(gp_error / max(abs(np.vdot(gp_value, dual)), 1.0e-30)),
            "observation_relative_error": float(
                observation_error / max(abs(np.vdot(observed, observation_dual)), 1.0e-30)
            ),
        }
