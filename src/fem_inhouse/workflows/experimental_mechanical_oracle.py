"""Matrix-free mechanical core of the DIC-compatible experimental oracle.

It connects the qualified spectral kinematics to the transactional driven-J2
material, exposes the exact residual/Jacobian actions, and provides the first
bounded augmented-Lagrangian increment solver used by the experimental oracle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse.linalg import LinearOperator, gmres

from fem_inhouse.core.driven_j2 import DrivenJ2PlaneStressBatch, DrivenJ2Trial
from fem_inhouse.core.plane_stress_material import (
    ConstitutiveIntegrationError,
    PlaneStressMaterialBatch,
)
from fem_inhouse.identification.dic_whitening import DICSpectralWhitener
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import DiscreteKinematics2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ExperimentalOracleObjectiveWeights:
    """Dimensionless weights of the first experimental-oracle objective."""

    dic: float = 1.0
    ludwik_prior: float = 1.0
    spatial_plastic_increment: float = 0.0
    temporal_plastic_increment: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "dic",
            "ludwik_prior",
            "spatial_plastic_increment",
            "temporal_plastic_increment",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} weight must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ExperimentalOracleOptimizationConfig:
    """Numerical controls for one sequential oracle increment."""

    maximum_augmented_iterations: int = 6
    maximum_inner_iterations: int = 200
    inner_gradient_tolerance: float = 1.0e-7
    projected_gradient_tolerance: float = 1.0e-3
    inner_function_tolerance: float = 1.0e-12
    equilibrium_rms_tolerance: float = 1.0e-6
    initial_penalty: float = 1.0
    penalty_growth: float = 10.0
    sufficient_constraint_reduction: float = 0.5
    displacement_variable_scale: float = 1.0e-3
    plastic_increment_variable_scale: float = 1.0e-3
    equilibrium_scale: float | None = None
    initialise_feasible_multiplier: bool = True
    multiplier_krylov_relative_tolerance: float = 1.0e-6
    maximum_multiplier_krylov_iterations: int = 500

    def __post_init__(self) -> None:
        if self.maximum_augmented_iterations < 1 or self.maximum_inner_iterations < 1:
            raise ValueError("oracle iteration limits must be positive")
        for name in (
            "inner_gradient_tolerance",
            "projected_gradient_tolerance",
            "inner_function_tolerance",
            "equilibrium_rms_tolerance",
            "initial_penalty",
            "penalty_growth",
            "displacement_variable_scale",
            "plastic_increment_variable_scale",
            "multiplier_krylov_relative_tolerance",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.penalty_growth <= 1.0:
            raise ValueError("penalty_growth must be greater than one")
        if not 0.0 < self.sufficient_constraint_reduction < 1.0:
            raise ValueError("sufficient_constraint_reduction must lie between zero and one")
        if self.equilibrium_scale is not None and (
            not isfinite(self.equilibrium_scale) or self.equilibrium_scale <= 0.0
        ):
            raise ValueError("equilibrium_scale must be finite and positive")
        if self.maximum_multiplier_krylov_iterations < 1:
            raise ValueError("maximum_multiplier_krylov_iterations must be positive")


@dataclass(frozen=True, slots=True)
class ExperimentalOracleObjectiveEvaluation:
    """One objective/gradient evaluation and its physical diagnostics."""

    value: float
    gradient: FloatArray
    dic_misfit: float
    ludwik_prior: float
    spatial_regularisation: float
    temporal_regularisation: float
    augmented_equilibrium: float
    equilibrium_rms: float
    linearisation: ExperimentalMechanicalOracleLinearisation


@dataclass(frozen=True, slots=True)
class ExperimentalOracleAugmentedIteration:
    index: int
    penalty: float
    objective: float
    equilibrium_rms: float
    inner_iterations: int
    inner_success: bool
    projected_gradient_inf: float


@dataclass(frozen=True, slots=True)
class ExperimentalOracleIncrementResult:
    """Accepted or reverted result of one DIC-history increment."""

    converged: bool
    displacement: FloatArray
    equivalent_plastic_increment: FloatArray
    linearisation: ExperimentalMechanicalOracleLinearisation
    objective: float
    dic_misfit: float
    ludwik_prior: float
    spatial_regularisation: float
    temporal_regularisation: float
    augmented_equilibrium: float
    equilibrium_rms: float
    augmented_iterations: tuple[ExperimentalOracleAugmentedIteration, ...]
    constitutive_rejections: int
    message: str


@dataclass(frozen=True, slots=True)
class ExperimentalOracleHistoryResult:
    """Sequentially committed mechanically admissible DIC history."""

    completed: bool
    increments: tuple[ExperimentalOracleIncrementResult, ...]
    displacement_history: FloatArray
    equivalent_plastic_increment_history: FloatArray
    equivalent_plastic_strain_history: FloatArray
    failed_increment: int | None


@dataclass(frozen=True, slots=True)
class ExperimentalOracleWarmStartRequest:
    """Inputs for an optional transaction-preserving per-increment warm start."""

    increment_index: int
    material: DrivenJ2PlaneStressBatch
    kinematics: DiscreteKinematics2D
    measured_displacement: FloatArray
    ludwik_increment: FloatArray
    initial_displacement: FloatArray
    time_increment: float


ExperimentalOracleWarmStart = Callable[
    [ExperimentalOracleWarmStartRequest], ArrayLike
]
ExperimentalOracleProgress = Callable[[int, ExperimentalOracleIncrementResult], None]


@dataclass(frozen=True, slots=True)
class FixedIncrementEquilibriumResult:
    """Equilibrated displacement at a prescribed plastic increment."""

    displacement: FloatArray
    linearisation: ExperimentalMechanicalOracleLinearisation
    newton_iterations: int
    krylov_iterations: tuple[int, ...]
    line_search_steps: tuple[float, ...]
    equilibrium_rms: float


@dataclass(frozen=True, slots=True)
class ExperimentalMechanicalOracleLinearisation:
    """One non-committed ``(u, Delta p)`` residual and linearisation."""

    kinematics: DiscreteKinematics2D
    trial: DrivenJ2Trial
    displacement_shape: tuple[int, int, int]
    plastic_increment_shape: tuple[int, int, int]
    mechanical_residual: FloatArray

    def mechanical_jacobian_action(self, displacement_increment: ArrayLike) -> FloatArray:
        """Apply the fixed-plastic-increment mechanical Jacobian."""

        return self.jacobian_action(
            displacement_increment,
            np.zeros(self.plastic_increment_shape, dtype=np.float64),
        )

    def plastic_residual_action(self, plastic_increment_increment: ArrayLike) -> FloatArray:
        """Apply ``G_p = d(B^T sigma)/d(Delta p)``."""

        return self.jacobian_action(
            np.zeros(self.displacement_shape, dtype=np.float64),
            plastic_increment_increment,
        )

    def mechanical_jacobian_transpose_action(self, mechanical_dual: ArrayLike) -> FloatArray:
        """Apply the transpose of the fixed-plastic mechanical Jacobian."""

        return self.jacobian_transpose_action(mechanical_dual)[0]

    def plastic_residual_transpose_action(self, mechanical_dual: ArrayLike) -> FloatArray:
        """Apply ``G_p^T`` without assembling the global operator."""

        return self.jacobian_transpose_action(mechanical_dual)[1]

    def jacobian_action(
        self,
        displacement_increment: ArrayLike,
        plastic_increment_increment: ArrayLike,
    ) -> FloatArray:
        """Apply ``d(B^T sigma)/d(u,Delta p)`` without global assembly."""

        displacement = np.asarray(displacement_increment, dtype=np.float64)
        if displacement.shape != self.displacement_shape:
            raise ValueError(
                f"displacement_increment must have shape {self.displacement_shape}"
            )
        plastic = np.asarray(plastic_increment_increment, dtype=np.float64)
        if plastic.shape != self.plastic_increment_shape:
            raise ValueError(
                "plastic_increment_increment must have shape "
                f"{self.plastic_increment_shape}"
            )
        strain_increment = self.kinematics.strain_samples(displacement).reshape(-1, 3)
        tangent = self.trial.tangent_in_plane_mpa
        if tangent is None:
            raise RuntimeError("oracle linearisation requires the strain tangent")
        stress_increment = np.einsum("pij,pj->pi", tangent, strain_increment)
        stress_increment += (
            self.trial.stress_equivalent_plastic_increment_tangent_mpa
            * plastic.reshape(-1, 1)
        )
        stress_shape = (*self.plastic_increment_shape, 3)
        return self.kinematics.divergence_from_sample_stress(
            stress_increment.reshape(stress_shape)
        )

    def jacobian_transpose_action(
        self,
        mechanical_dual: ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        """Apply the Euclidean transpose of the mechanical Jacobian.

        ``divergence_from_sample_stress`` is the negative weighted adjoint of
        ``strain_samples``.  Keeping that sign and quadrature weight explicit
        here avoids assuming that the constitutive tangent is symmetric.
        """

        dual = np.asarray(mechanical_dual, dtype=np.float64)
        if dual.shape != self.displacement_shape:
            raise ValueError(f"mechanical_dual must have shape {self.displacement_shape}")
        dual_strain = self.kinematics.strain_samples(dual).reshape(-1, 3)
        tangent = self.trial.tangent_in_plane_mpa
        if tangent is None:
            raise RuntimeError("oracle linearisation requires the strain tangent")
        stress_dual = np.einsum("pji,pj->pi", tangent, dual_strain)
        stress_shape = (*self.plastic_increment_shape, 3)
        displacement_gradient = self.kinematics.divergence_from_sample_stress(
            stress_dual.reshape(stress_shape)
        )
        plastic_gradient = -self.kinematics.sample_quadrature_weight * np.einsum(
            "pi,pi->p",
            self.trial.stress_equivalent_plastic_increment_tangent_mpa,
            dual_strain,
        )
        return displacement_gradient, plastic_gradient.reshape(
            self.plastic_increment_shape
        )


def evaluate_experimental_mechanical_oracle(
    material: DrivenJ2PlaneStressBatch,
    kinematics: DiscreteKinematics2D,
    displacement: ArrayLike,
    equivalent_plastic_increment: ArrayLike,
    *,
    time_increment: float,
) -> ExperimentalMechanicalOracleLinearisation:
    """Evaluate one trial state of the mechanics-only oracle constraint."""

    displacement_values = np.asarray(displacement, dtype=np.float64)
    if displacement_values.ndim != 3 or displacement_values.shape[-1] != 2:
        raise ValueError("displacement must have shape (nx+1, ny+1, 2)")
    strain_samples = kinematics.strain_samples(displacement_values)
    expected_increment_shape = strain_samples.shape[:-1]
    increment = np.asarray(equivalent_plastic_increment, dtype=np.float64)
    if increment.shape != expected_increment_shape:
        raise ValueError(
            f"equivalent_plastic_increment must have shape {expected_increment_shape}"
        )
    if material.point_count != int(np.prod(expected_increment_shape)):
        raise ValueError("material point count does not match the kinematic samples")

    trial = material.evaluate(
        strain_samples.reshape(-1, 3),
        increment.reshape(-1),
        time_increment=time_increment,
        consistent_tangent=True,
    )
    stress = trial.stress_in_plane_mpa.reshape((*expected_increment_shape, 3))
    residual = kinematics.divergence_from_sample_stress(stress)
    return ExperimentalMechanicalOracleLinearisation(
        kinematics=kinematics,
        trial=trial,
        displacement_shape=displacement_values.shape,
        plastic_increment_shape=expected_increment_shape,
        mechanical_residual=residual,
    )


def _fixed_displacement_jacobian_action(
    linearisation: ExperimentalMechanicalOracleLinearisation,
    zero_plastic_direction: FloatArray,
    grid: StructuredGrid2D,
    vector: NDArray[np.float64],
) -> NDArray[np.float64]:
    displacement_increment = unpack_interior(vector, grid)
    return pack_interior(
        linearisation.jacobian_action(
            displacement_increment,
            zero_plastic_direction,
        )
    )


def solve_fixed_plastic_increment_equilibrium(
    *,
    material: DrivenJ2PlaneStressBatch,
    kinematics: DiscreteKinematics2D,
    boundary_displacement: ArrayLike,
    equivalent_plastic_increment: ArrayLike,
    initial_displacement: ArrayLike | None = None,
    time_increment: float = 1.0,
    equilibrium_rms_tolerance: float = 1.0e-9,
    maximum_newton_iterations: int = 30,
    maximum_line_search_iterations: int = 12,
    krylov_relative_tolerance: float = 1.0e-10,
    maximum_krylov_iterations: int = 500,
) -> FixedIncrementEquilibriumResult:
    """Solve mechanical equilibrium at prescribed ``Delta p`` without commit."""

    boundary = np.asarray(boundary_displacement, dtype=np.float64)
    if boundary.ndim != 3 or boundary.shape[-1] != 2:
        raise ValueError("boundary_displacement must have shape (nx+1, ny+1, 2)")
    displacement = boundary.copy()
    if initial_displacement is not None:
        initial = np.asarray(initial_displacement, dtype=np.float64)
        if initial.shape != boundary.shape:
            raise ValueError("initial_displacement and boundary shapes must match")
        displacement[1:-1, 1:-1] = initial[1:-1, 1:-1]
    increment = np.asarray(equivalent_plastic_increment, dtype=np.float64)
    sample_shape = kinematics.strain_samples(boundary).shape[:-1]
    if increment.shape != sample_shape:
        raise ValueError(f"equivalent_plastic_increment must have shape {sample_shape}")
    if np.any(increment < 0.0) or not np.isfinite(increment).all():
        raise ValueError("equivalent_plastic_increment must be finite and non-negative")
    if maximum_newton_iterations < 1 or maximum_line_search_iterations < 1:
        raise ValueError("Newton and line-search limits must be positive")
    grid = getattr(kinematics, "grid", None)
    if not isinstance(grid, StructuredGrid2D):
        raise TypeError("fixed-increment equilibrium requires StructuredGrid2D")
    zero_plastic_direction = np.zeros_like(increment)
    krylov_counts: list[int] = []
    accepted_steps: list[float] = []

    try:
        for newton in range(maximum_newton_iterations + 1):
            linearisation = evaluate_experimental_mechanical_oracle(
                material,
                kinematics,
                displacement,
                increment,
                time_increment=time_increment,
            )
            residual_vector = pack_interior(linearisation.mechanical_residual)
            constraint_count = max(residual_vector.size, 1)
            residual_rms = float(
                np.linalg.norm(residual_vector) / np.sqrt(constraint_count)
            )
            if residual_rms <= equilibrium_rms_tolerance:
                return FixedIncrementEquilibriumResult(
                    displacement=displacement,
                    linearisation=linearisation,
                    newton_iterations=newton,
                    krylov_iterations=tuple(krylov_counts),
                    line_search_steps=tuple(accepted_steps),
                    equilibrium_rms=residual_rms,
                )
            if newton == maximum_newton_iterations:
                break
            iteration_count = 0

            def callback(_: NDArray[np.float64]) -> None:
                nonlocal iteration_count
                iteration_count += 1

            correction, info = gmres(
                LinearOperator(
                    (residual_vector.size, residual_vector.size),
                    matvec=partial(
                        _fixed_displacement_jacobian_action,
                        linearisation,
                        zero_plastic_direction,
                        grid,
                    ),
                    dtype=np.float64,
                ),
                -residual_vector,
                rtol=krylov_relative_tolerance,
                atol=0.0,
                maxiter=maximum_krylov_iterations,
                callback=callback,
                callback_type="pr_norm",
            )
            krylov_counts.append(iteration_count)
            if info != 0:
                raise RuntimeError(f"fixed-increment GMRES failed with info={info}")
            correction_field = unpack_interior(correction, grid)
            step = 1.0
            for _ in range(maximum_line_search_iterations):
                candidate = displacement + step * correction_field
                candidate[0, :, :] = boundary[0, :, :]
                candidate[-1, :, :] = boundary[-1, :, :]
                candidate[:, 0, :] = boundary[:, 0, :]
                candidate[:, -1, :] = boundary[:, -1, :]
                try:
                    candidate_linearisation = evaluate_experimental_mechanical_oracle(
                        material,
                        kinematics,
                        candidate,
                        increment,
                        time_increment=time_increment,
                    )
                except ConstitutiveIntegrationError:
                    step *= 0.5
                    continue
                candidate_residual = pack_interior(
                    candidate_linearisation.mechanical_residual
                )
                if np.linalg.norm(candidate_residual) < np.linalg.norm(residual_vector):
                    displacement = candidate
                    accepted_steps.append(step)
                    break
                step *= 0.5
            else:
                raise RuntimeError("fixed-increment Newton line search failed")
    except Exception:
        material.revert()
        raise
    material.revert()
    raise RuntimeError("fixed-increment Newton did not converge")


def _quadratic_difference_regularisation(
    values: FloatArray,
    *,
    scale: float,
) -> tuple[float, FloatArray]:
    """Return a mesh-size-normalised nearest-neighbour quadratic prior."""

    scaled = values / scale
    gradient = np.zeros_like(values)
    objective = 0.0
    for axis in (0, 1):
        difference = np.diff(scaled, axis=axis)
        if difference.size == 0:
            continue
        objective += 0.5 * float(np.mean(difference**2))
        contribution = difference / (difference.size * scale)
        lower = [slice(None)] * values.ndim
        upper = [slice(None)] * values.ndim
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        gradient[tuple(lower)] -= contribution
        gradient[tuple(upper)] += contribution
    return objective, gradient


class ExperimentalOracleIncrementProblem:
    """Differentiable bounded P0 problem for one accepted load increment."""

    def __init__(
        self,
        *,
        material: DrivenJ2PlaneStressBatch,
        kinematics: DiscreteKinematics2D,
        measured_displacement: ArrayLike,
        whitener: DICSpectralWhitener,
        ludwik_increment: ArrayLike,
        previous_increment: ArrayLike | None,
        weights: ExperimentalOracleObjectiveWeights,
        time_increment: float,
        displacement_variable_scale: float,
        plastic_increment_variable_scale: float,
        equilibrium_scale: float,
        plastic_basis: ArrayLike | None = None,
    ) -> None:
        measured = np.asarray(measured_displacement, dtype=np.float64)
        if measured.shape != whitener.field_shape or measured.shape[-1] != 2:
            raise ValueError("measured displacement and DIC whitener shapes must match")
        sample_shape = kinematics.strain_samples(measured).shape[:-1]
        ludwik = np.asarray(ludwik_increment, dtype=np.float64)
        if ludwik.shape != sample_shape or np.any(ludwik < 0.0):
            raise ValueError(f"ludwik_increment must be non-negative with shape {sample_shape}")
        if not np.isfinite(ludwik).all():
            raise ValueError("ludwik_increment must be finite")
        previous = np.zeros(sample_shape, dtype=np.float64)
        if previous_increment is not None:
            previous = np.asarray(previous_increment, dtype=np.float64)
            if previous.shape != sample_shape or not np.isfinite(previous).all():
                raise ValueError(f"previous_increment must be finite with shape {sample_shape}")
        if material.point_count != int(np.prod(sample_shape)):
            raise ValueError("material point count does not match the oracle samples")
        self.material = material
        self.kinematics = kinematics
        self.measured_displacement = measured.copy()
        self.whitener = whitener
        self.ludwik_increment = ludwik.copy()
        self.previous_increment = previous.copy()
        self.weights = weights
        self.time_increment = float(time_increment)
        self.displacement_variable_scale = float(displacement_variable_scale)
        self.plastic_increment_variable_scale = float(plastic_increment_variable_scale)
        self.equilibrium_scale = float(equilibrium_scale)
        self.displacement_shape = measured.shape
        self.plastic_increment_shape = sample_shape
        self.interior_shape = measured[1:-1, 1:-1].shape
        self.displacement_unknown_count = int(np.prod(self.interior_shape))
        if plastic_basis is None:
            self.plastic_basis: FloatArray | None = None
            self.reduced_plastic_count = int(np.prod(sample_shape))
        else:
            basis = np.asarray(plastic_basis, dtype=np.float64)
            expected_rows = int(np.prod(sample_shape))
            if basis.ndim != 2 or basis.shape[0] != expected_rows or basis.shape[1] < 1:
                raise ValueError(
                    "plastic_basis must have shape (plastic_points, rank)"
                )
            if not np.isfinite(basis).all():
                raise ValueError("plastic_basis must be finite")
            self.plastic_basis = basis.copy()
            self.reduced_plastic_count = basis.shape[1]
        self.plastic_unknown_count = self.reduced_plastic_count
        self.mechanical_constraint_count = self.displacement_unknown_count
        self.constitutive_rejections = 0
        self.last_admissible_variables: FloatArray | None = None
        self.last_admissible_value: float | None = None
        self.last_admissible_gradient: FloatArray | None = None

    @property
    def variable_count(self) -> int:
        return self.displacement_unknown_count + self.plastic_unknown_count

    @property
    def reduced(self) -> bool:
        return self.plastic_basis is not None

    def pack_state(
        self,
        displacement: ArrayLike,
        equivalent_plastic_increment: ArrayLike,
    ) -> FloatArray:
        displacement_values = np.asarray(displacement, dtype=np.float64)
        increment = np.asarray(equivalent_plastic_increment, dtype=np.float64)
        if displacement_values.shape != self.displacement_shape:
            raise ValueError(f"displacement must have shape {self.displacement_shape}")
        if increment.shape != self.plastic_increment_shape:
            raise ValueError(
                f"equivalent_plastic_increment must have shape {self.plastic_increment_shape}"
            )
        displacement_variables = (
            (displacement_values[1:-1, 1:-1] - self.measured_displacement[1:-1, 1:-1])
            / self.displacement_variable_scale
        ).ravel()
        if self.plastic_basis is None:
            plastic_variables = (increment / self.plastic_increment_variable_scale).ravel()
        else:
            coefficients, *_ = np.linalg.lstsq(
                self.plastic_basis,
                (increment - self.ludwik_increment).ravel(),
                rcond=None,
            )
            plastic_variables = coefficients / self.plastic_increment_variable_scale
        return np.concatenate((displacement_variables, plastic_variables))

    def unpack_state(self, variables: ArrayLike) -> tuple[FloatArray, FloatArray]:
        values = np.asarray(variables, dtype=np.float64)
        if values.shape != (self.variable_count,):
            raise ValueError(f"variables must have shape {(self.variable_count,)}")
        displacement = self.measured_displacement.copy()
        displacement[1:-1, 1:-1] += (
            values[: self.displacement_unknown_count].reshape(self.interior_shape)
            * self.displacement_variable_scale
        )
        plastic_values = values[self.displacement_unknown_count :]
        if self.plastic_basis is None:
            increment = plastic_values.reshape(self.plastic_increment_shape) * (
                self.plastic_increment_variable_scale
            )
        else:
            coefficient = plastic_values * self.plastic_increment_variable_scale
            increment = (
                self.ludwik_increment.ravel() + self.plastic_basis @ coefficient
            ).reshape(self.plastic_increment_shape)
        return displacement, increment

    def objective_and_gradient(
        self,
        variables: ArrayLike,
        *,
        multiplier: ArrayLike,
        penalty: float,
    ) -> ExperimentalOracleObjectiveEvaluation:
        displacement, increment = self.unpack_state(variables)
        multiplier_values = np.asarray(multiplier, dtype=np.float64)
        if multiplier_values.shape != (self.mechanical_constraint_count,):
            raise ValueError(
                f"multiplier must have shape {(self.mechanical_constraint_count,)}"
            )
        if not isfinite(penalty) or penalty <= 0.0:
            raise ValueError("penalty must be finite and positive")
        linearisation = evaluate_experimental_mechanical_oracle(
            self.material,
            self.kinematics,
            displacement,
            increment,
            time_increment=self.time_increment,
        )
        displacement_difference = displacement - self.measured_displacement
        data_size = displacement_difference.size
        dic_value = self.weights.dic * self.whitener.quadratic_misfit(
            displacement_difference
        ) / data_size
        displacement_gradient = (
            self.weights.dic
            * self.whitener.normal_action(displacement_difference)
            / data_size
        )

        plastic_scale = self.plastic_increment_variable_scale
        prior_difference = (increment - self.ludwik_increment) / plastic_scale
        prior_value = 0.5 * self.weights.ludwik_prior * float(
            np.mean(prior_difference**2)
        )
        increment_gradient = (
            self.weights.ludwik_prior
            * prior_difference
            / (prior_difference.size * plastic_scale)
        )
        temporal_difference = (increment - self.previous_increment) / plastic_scale
        temporal_value = 0.5 * self.weights.temporal_plastic_increment * float(
            np.mean(temporal_difference**2)
        )
        increment_gradient += (
            self.weights.temporal_plastic_increment
            * temporal_difference
            / (temporal_difference.size * plastic_scale)
        )
        spatial_raw, spatial_gradient = _quadratic_difference_regularisation(
            increment,
            scale=plastic_scale,
        )
        spatial_value = self.weights.spatial_plastic_increment * spatial_raw
        increment_gradient += self.weights.spatial_plastic_increment * spatial_gradient

        residual = linearisation.mechanical_residual[1:-1, 1:-1].ravel()
        scaled_residual = residual / self.equilibrium_scale
        constraint_count = max(self.mechanical_constraint_count, 1)
        augmented_value = float(
            (
                np.vdot(multiplier_values, scaled_residual).real
                + 0.5 * penalty * np.vdot(scaled_residual, scaled_residual).real
            )
            / constraint_count
        )
        mechanical_dual = np.zeros(self.displacement_shape, dtype=np.float64)
        mechanical_dual[1:-1, 1:-1] = (
            (multiplier_values + penalty * scaled_residual)
            / (self.equilibrium_scale * constraint_count)
        ).reshape(self.interior_shape)
        equilibrium_u_gradient, equilibrium_p_gradient = (
            linearisation.jacobian_transpose_action(mechanical_dual)
        )
        displacement_gradient += equilibrium_u_gradient
        increment_gradient += equilibrium_p_gradient

        if self.plastic_basis is None:
            plastic_gradient = increment_gradient.ravel() * (
                self.plastic_increment_variable_scale
            )
        else:
            plastic_gradient = (
                self.plastic_basis.T @ increment_gradient.ravel()
            ) * self.plastic_increment_variable_scale
        physical_gradient = np.concatenate(
            (
                displacement_gradient[1:-1, 1:-1].ravel()
                * self.displacement_variable_scale,
                plastic_gradient,
            )
        )
        total = dic_value + prior_value + temporal_value + spatial_value + augmented_value
        equilibrium_rms = float(np.linalg.norm(residual) / np.sqrt(constraint_count))
        evaluation = ExperimentalOracleObjectiveEvaluation(
            value=total,
            gradient=physical_gradient,
            dic_misfit=dic_value,
            ludwik_prior=prior_value,
            spatial_regularisation=spatial_value,
            temporal_regularisation=temporal_value,
            augmented_equilibrium=augmented_value,
            equilibrium_rms=equilibrium_rms,
            linearisation=linearisation,
        )
        self.last_admissible_variables = np.asarray(variables, dtype=np.float64).copy()
        self.last_admissible_value = total
        self.last_admissible_gradient = physical_gradient.copy()
        return evaluation


def _objective_value_gradient(
    problem: ExperimentalOracleIncrementProblem,
    candidate: FloatArray,
    *,
    multiplier: FloatArray,
    penalty: float,
) -> tuple[float, FloatArray]:
    try:
        evaluation = problem.objective_and_gradient(
            candidate,
            multiplier=multiplier,
            penalty=penalty,
        )
    except ConstitutiveIntegrationError:
        problem.material.revert()
        problem.constitutive_rejections += 1
        reference = problem.last_admissible_variables
        if reference is None:
            reference = np.zeros_like(candidate)
        difference = np.asarray(candidate, dtype=np.float64) - reference
        barrier_scale = max(abs(problem.last_admissible_value or 0.0), 1.0)
        value = (problem.last_admissible_value or 0.0) + barrier_scale * (
            1.0 + 0.5 * float(np.vdot(difference, difference).real)
        )
        gradient = problem.last_admissible_gradient
        if gradient is None:
            gradient = barrier_scale * difference
        return value, gradient.copy()
    return evaluation.value, evaluation.gradient


def _initial_feasible_multiplier(
    problem: ExperimentalOracleIncrementProblem,
    evaluation: ExperimentalOracleObjectiveEvaluation,
    config: ExperimentalOracleOptimizationConfig,
) -> FloatArray:
    """Balance the DIC/prior displacement gradient at a feasible warm start."""

    count = problem.mechanical_constraint_count
    zero = np.zeros(count, dtype=np.float64)
    if (
        not config.initialise_feasible_multiplier
        or evaluation.equilibrium_rms > config.equilibrium_rms_tolerance
    ):
        return zero
    right_hand_side = -evaluation.gradient[: problem.displacement_unknown_count]
    if np.linalg.norm(right_hand_side) == 0.0:
        return zero

    linearisation = evaluation.linearisation

    def transpose_action(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        dual = np.zeros(problem.displacement_shape, dtype=np.float64)
        dual[1:-1, 1:-1] = vector.reshape(problem.interior_shape)
        displacement_gradient, _ = linearisation.jacobian_transpose_action(dual)
        return (
            displacement_gradient[1:-1, 1:-1].ravel()
            * problem.displacement_variable_scale
        )

    dual, info = gmres(
        LinearOperator((count, count), matvec=transpose_action, dtype=np.float64),
        right_hand_side,
        rtol=config.multiplier_krylov_relative_tolerance,
        atol=0.0,
        maxiter=config.maximum_multiplier_krylov_iterations,
    )
    if info != 0 or not np.isfinite(dual).all():
        return zero
    return dual * problem.equilibrium_scale * count


def solve_experimental_mechanical_oracle_increment(
    *,
    material: DrivenJ2PlaneStressBatch,
    kinematics: DiscreteKinematics2D,
    measured_displacement: ArrayLike,
    whitener: DICSpectralWhitener,
    ludwik_increment: ArrayLike,
    initial_displacement: ArrayLike,
    initial_equivalent_plastic_increment: ArrayLike,
    previous_increment: ArrayLike | None = None,
    weights: ExperimentalOracleObjectiveWeights | None = None,
    config: ExperimentalOracleOptimizationConfig | None = None,
    time_increment: float = 1.0,
    commit_on_success: bool = True,
    plastic_basis: ArrayLike | None = None,
) -> ExperimentalOracleIncrementResult:
    """Solve and transactionally accept one experimental-oracle increment."""

    measured = np.asarray(measured_displacement, dtype=np.float64)
    initial_u = np.asarray(initial_displacement, dtype=np.float64)
    initial_p = np.asarray(initial_equivalent_plastic_increment, dtype=np.float64)
    if weights is None:
        weights = ExperimentalOracleObjectiveWeights()
    if config is None:
        config = ExperimentalOracleOptimizationConfig()
    if initial_u.shape != measured.shape:
        raise ValueError("initial and measured displacement shapes must match")
    initial_p = np.maximum(initial_p, 0.0)

    provisional_scale = config.equilibrium_scale or 1.0
    problem = ExperimentalOracleIncrementProblem(
        material=material,
        kinematics=kinematics,
        measured_displacement=measured,
        whitener=whitener,
        ludwik_increment=ludwik_increment,
        previous_increment=previous_increment,
        weights=weights,
        time_increment=time_increment,
        displacement_variable_scale=config.displacement_variable_scale,
        plastic_increment_variable_scale=config.plastic_increment_variable_scale,
        equilibrium_scale=provisional_scale,
        plastic_basis=plastic_basis,
    )
    variables = problem.pack_state(initial_u, initial_p)
    if config.equilibrium_scale is None:
        initial_linearisation = evaluate_experimental_mechanical_oracle(
            material,
            kinematics,
            *problem.unpack_state(variables),
            time_increment=time_increment,
        )
        initial_residual = initial_linearisation.mechanical_residual[1:-1, 1:-1]
        automatic_scale = max(
            float(np.linalg.norm(initial_residual) / np.sqrt(max(initial_residual.size, 1))),
            1.0,
        )
        problem.equilibrium_scale = automatic_scale

    multiplier = np.zeros(problem.mechanical_constraint_count, dtype=np.float64)
    penalty = config.initial_penalty
    initial_evaluation = problem.objective_and_gradient(
        variables,
        multiplier=multiplier,
        penalty=penalty,
    )
    multiplier = _initial_feasible_multiplier(
        problem, initial_evaluation, config
    )
    previous_equilibrium_rms: float | None = None
    lower_bounds = np.full(problem.variable_count, -np.inf, dtype=np.float64)
    if not problem.reduced:
        lower_bounds[problem.displacement_unknown_count :] = 0.0
    bounds = list(zip(lower_bounds, np.full(problem.variable_count, np.inf), strict=True))
    constraints: tuple[LinearConstraint, ...] = ()
    if problem.reduced:
        assert problem.plastic_basis is not None
        constraint_matrix = np.zeros(
            (problem.plastic_basis.shape[0], problem.variable_count), dtype=np.float64
        )
        constraint_matrix[:, problem.displacement_unknown_count :] = (
            problem.plastic_basis * problem.plastic_increment_variable_scale
        )
        constraints = (
            LinearConstraint(
                constraint_matrix,
                -problem.ludwik_increment.ravel(),
                np.full(problem.plastic_basis.shape[0], np.inf),
            ),
        )
    history: list[ExperimentalOracleAugmentedIteration] = []
    final_evaluation: ExperimentalOracleObjectiveEvaluation | None = None
    message = "maximum augmented-Lagrangian iterations reached"
    converged = False

    try:
        for outer in range(1, config.maximum_augmented_iterations + 1):
            optimisation_kwargs: dict[str, object] = {
                "method": "SLSQP" if problem.reduced else "L-BFGS-B",
                "jac": True,
                "bounds": bounds,
                "options": {
                    "maxiter": config.maximum_inner_iterations,
                    "ftol": config.inner_function_tolerance,
                    **(
                        {}
                        if problem.reduced
                        else {"gtol": config.inner_gradient_tolerance}
                    ),
                },
            }
            if problem.reduced:
                optimisation_kwargs["constraints"] = constraints
            optimisation = minimize(
                partial(
                    _objective_value_gradient,
                    problem,
                    multiplier=multiplier.copy(),
                    penalty=penalty,
                ),
                variables,
                **optimisation_kwargs,
            )
            variables = np.asarray(optimisation.x, dtype=np.float64)
            final_evaluation = problem.objective_and_gradient(
                variables,
                multiplier=multiplier,
                penalty=penalty,
            )
            projected_gradient = np.asarray(optimisation.jac, dtype=np.float64).copy()
            plastic_variables = variables[problem.displacement_unknown_count :]
            plastic_gradient = projected_gradient[problem.displacement_unknown_count :]
            plastic_gradient[
                (plastic_variables <= 1.0e-14) & (plastic_gradient > 0.0)
            ] = 0.0
            projected_gradient_inf = float(
                np.max(np.abs(projected_gradient), initial=0.0)
            )
            history.append(
                ExperimentalOracleAugmentedIteration(
                    index=outer,
                    penalty=penalty,
                    objective=final_evaluation.value,
                    equilibrium_rms=final_evaluation.equilibrium_rms,
                    inner_iterations=int(optimisation.nit),
                    inner_success=bool(optimisation.success),
                    projected_gradient_inf=projected_gradient_inf,
                )
            )
            if (
                final_evaluation.equilibrium_rms
                <= config.equilibrium_rms_tolerance
                and projected_gradient_inf <= config.projected_gradient_tolerance
            ):
                converged = True
                message = str(optimisation.message)
                break
            residual = final_evaluation.linearisation.mechanical_residual[1:-1, 1:-1]
            multiplier += penalty * residual.ravel() / problem.equilibrium_scale
            if (
                previous_equilibrium_rms is not None
                and final_evaluation.equilibrium_rms
                > config.sufficient_constraint_reduction * previous_equilibrium_rms
            ):
                penalty *= config.penalty_growth
            previous_equilibrium_rms = final_evaluation.equilibrium_rms
        if final_evaluation is None:
            raise RuntimeError("oracle optimiser did not perform an evaluation")
        displacement, increment = problem.unpack_state(variables)
        final_evaluation = problem.objective_and_gradient(
            variables,
            multiplier=multiplier,
            penalty=penalty,
        )
        if converged and commit_on_success:
            material.commit()
        else:
            material.revert()
        return ExperimentalOracleIncrementResult(
            converged=converged,
            displacement=displacement,
            equivalent_plastic_increment=increment,
            linearisation=final_evaluation.linearisation,
            objective=final_evaluation.value,
            dic_misfit=final_evaluation.dic_misfit,
            ludwik_prior=final_evaluation.ludwik_prior,
            spatial_regularisation=final_evaluation.spatial_regularisation,
            temporal_regularisation=final_evaluation.temporal_regularisation,
            augmented_equilibrium=final_evaluation.augmented_equilibrium,
            equilibrium_rms=final_evaluation.equilibrium_rms,
            augmented_iterations=tuple(history),
            constitutive_rejections=problem.constitutive_rejections,
            message=message,
        )
    except Exception:
        material.revert()
        raise


def solve_experimental_mechanical_oracle_reduced_increment(
    *,
    material: DrivenJ2PlaneStressBatch,
    kinematics: DiscreteKinematics2D,
    measured_displacement: ArrayLike,
    whitener: DICSpectralWhitener,
    ludwik_increment: ArrayLike,
    initial_displacement: ArrayLike,
    initial_equivalent_plastic_increment: ArrayLike,
    previous_increment: ArrayLike | None = None,
    weights: ExperimentalOracleObjectiveWeights | None = None,
    config: ExperimentalOracleOptimizationConfig | None = None,
    time_increment: float = 1.0,
    commit_on_success: bool = True,
    plastic_basis: ArrayLike | None = None,
) -> ExperimentalOracleIncrementResult:
    """Minimise on the exact mechanical-equilibrium manifold using an adjoint."""

    if weights is None:
        weights = ExperimentalOracleObjectiveWeights()
    if config is None:
        config = ExperimentalOracleOptimizationConfig()
    measured = np.asarray(measured_displacement, dtype=np.float64)
    initial_u = np.asarray(initial_displacement, dtype=np.float64)
    initial_p = np.maximum(
        np.asarray(initial_equivalent_plastic_increment, dtype=np.float64), 0.0
    )
    problem = ExperimentalOracleIncrementProblem(
        material=material,
        kinematics=kinematics,
        measured_displacement=measured,
        whitener=whitener,
        ludwik_increment=ludwik_increment,
        previous_increment=previous_increment,
        weights=weights,
        time_increment=time_increment,
        displacement_variable_scale=config.displacement_variable_scale,
        plastic_increment_variable_scale=config.plastic_increment_variable_scale,
        equilibrium_scale=config.equilibrium_scale or 1.0,
        plastic_basis=plastic_basis,
    )
    if initial_p.shape != problem.plastic_increment_shape:
        raise ValueError(
            "initial_equivalent_plastic_increment has an incompatible shape"
        )
    initial_state = problem.pack_state(initial_u, initial_p)
    plastic_variables = initial_state[problem.displacement_unknown_count :]
    last_displacement = initial_u.copy()
    last_variables: FloatArray | None = None
    last_value: float | None = None
    last_gradient: FloatArray | None = None
    last_evaluation: ExperimentalOracleObjectiveEvaluation | None = None
    constitutive_rejections = 0

    def reduced_objective(candidate: FloatArray) -> tuple[float, FloatArray]:
        nonlocal last_displacement, last_variables, last_value, last_gradient
        nonlocal last_evaluation, constitutive_rejections
        candidate_values = np.asarray(candidate, dtype=np.float64)
        if problem.reduced:
            state_values = np.concatenate(
                (
                    np.zeros(problem.displacement_unknown_count, dtype=np.float64),
                    candidate_values,
                )
            )
            increment = problem.unpack_state(state_values)[1]
        else:
            increment = candidate_values.reshape(problem.plastic_increment_shape) * (
                config.plastic_increment_variable_scale
            )
        try:
            try:
                equilibrium = solve_fixed_plastic_increment_equilibrium(
                    material=material,
                    kinematics=kinematics,
                    boundary_displacement=measured,
                    equivalent_plastic_increment=increment,
                    initial_displacement=last_displacement,
                    time_increment=time_increment,
                    equilibrium_rms_tolerance=min(
                        0.1 * config.equilibrium_rms_tolerance, 1.0e-8
                    ),
                )
            except (ConstitutiveIntegrationError, RuntimeError):
                material.revert()
                equilibrium = solve_fixed_plastic_increment_equilibrium(
                    material=material,
                    kinematics=kinematics,
                    boundary_displacement=measured,
                    equivalent_plastic_increment=increment,
                    initial_displacement=measured,
                    time_increment=time_increment,
                    equilibrium_rms_tolerance=min(
                        0.1 * config.equilibrium_rms_tolerance, 1.0e-8
                    ),
                )
            last_displacement = equilibrium.displacement.copy()
            state = problem.pack_state(last_displacement, increment)
            evaluation = problem.objective_and_gradient(
                state,
                multiplier=np.zeros(problem.mechanical_constraint_count),
                penalty=1.0e-12,
            )
            displacement_gradient = evaluation.gradient[
                : problem.displacement_unknown_count
            ]
            linearisation = equilibrium.linearisation

            def transpose_displacement_action(
                vector: NDArray[np.float64],
            ) -> NDArray[np.float64]:
                dual = np.zeros(problem.displacement_shape, dtype=np.float64)
                dual[1:-1, 1:-1] = vector.reshape(problem.interior_shape)
                gradient_u, _ = linearisation.jacobian_transpose_action(dual)
                return (
                    gradient_u[1:-1, 1:-1].ravel()
                    * config.displacement_variable_scale
                )

            adjoint, info = gmres(
                LinearOperator(
                    (
                        problem.displacement_unknown_count,
                        problem.displacement_unknown_count,
                    ),
                    matvec=transpose_displacement_action,
                    dtype=np.float64,
                ),
                displacement_gradient,
                rtol=config.multiplier_krylov_relative_tolerance,
                atol=0.0,
                maxiter=config.maximum_multiplier_krylov_iterations,
            )
            if info != 0:
                raise RuntimeError(f"reduced-gradient adjoint failed with info={info}")
            dual = np.zeros(problem.displacement_shape, dtype=np.float64)
            dual[1:-1, 1:-1] = adjoint.reshape(problem.interior_shape)
            _, plastic_constraint_gradient = (
                linearisation.jacobian_transpose_action(dual)
            )
            if problem.reduced:
                assert problem.plastic_basis is not None
                constraint_gradient = (
                    problem.plastic_basis.T @ plastic_constraint_gradient.ravel()
                ) * config.plastic_increment_variable_scale
            else:
                constraint_gradient = (
                    plastic_constraint_gradient.ravel()
                    * config.plastic_increment_variable_scale
                )
            reduced_gradient = evaluation.gradient[
                problem.displacement_unknown_count :
            ] - constraint_gradient
            last_variables = np.asarray(candidate, dtype=np.float64).copy()
            last_value = evaluation.value
            last_gradient = reduced_gradient.copy()
            last_evaluation = evaluation
            return evaluation.value, reduced_gradient
        except (ConstitutiveIntegrationError, RuntimeError):
            material.revert()
            constitutive_rejections += 1
            reference = (
                np.zeros_like(candidate) if last_variables is None else last_variables
            )
            difference = np.asarray(candidate, dtype=np.float64) - reference
            scale = max(abs(last_value or 0.0), 1.0)
            value = (last_value or 0.0) + scale * (
                1.0 + 0.5 * float(np.vdot(difference, difference).real)
            )
            gradient = scale * difference if last_gradient is None else last_gradient
            return value, gradient.copy()

    bounds: list[tuple[float | None, float | None]] = [
        (None, None)
    ] * plastic_variables.size
    constraints: tuple[LinearConstraint, ...] = ()
    if problem.reduced:
        assert problem.plastic_basis is not None
        constraints = (
            LinearConstraint(
                problem.plastic_basis * config.plastic_increment_variable_scale,
                -problem.ludwik_increment.ravel(),
                np.full(problem.plastic_basis.shape[0], np.inf),
            ),
        )
    else:
        bounds = [(0.0, None)] * plastic_variables.size
    try:
        history: list[ExperimentalOracleAugmentedIteration] = []
        converged = False
        optimisation = None
        final_variables = plastic_variables
        projected_gradient_inf = float("inf")
        for restart in range(1, config.maximum_augmented_iterations + 1):
            optimisation_kwargs: dict[str, object] = {
                "method": "SLSQP" if problem.reduced else "L-BFGS-B",
                "jac": True,
                "bounds": bounds,
                "options": {
                    "maxiter": config.maximum_inner_iterations,
                    "ftol": config.inner_function_tolerance,
                    **(
                        {
                            "maxls": 100,
                            "gtol": config.inner_gradient_tolerance,
                        }
                        if not problem.reduced
                        else {}
                    ),
                },
            }
            if problem.reduced:
                optimisation_kwargs["constraints"] = constraints
            optimisation = minimize(
                reduced_objective,
                final_variables,
                **optimisation_kwargs,
            )
            final_variables = np.asarray(optimisation.x, dtype=np.float64)
            _, final_gradient = reduced_objective(final_variables)
            if last_evaluation is None:
                raise RuntimeError("reduced oracle did not produce an admissible state")
            projected_gradient = final_gradient.copy()
            if not problem.reduced:
                projected_gradient[
                    (final_variables <= 1.0e-14) & (projected_gradient > 0.0)
                ] = 0.0
            projected_gradient_inf = float(
                np.max(np.abs(projected_gradient), initial=0.0)
            )
            history.append(
                ExperimentalOracleAugmentedIteration(
                    index=restart,
                    penalty=0.0,
                    objective=last_evaluation.value,
                    equilibrium_rms=last_evaluation.equilibrium_rms,
                    inner_iterations=int(optimisation.nit),
                    inner_success=bool(optimisation.success),
                    projected_gradient_inf=projected_gradient_inf,
                )
            )
            converged = (
                last_evaluation.equilibrium_rms <= config.equilibrium_rms_tolerance
                and projected_gradient_inf <= config.projected_gradient_tolerance
            )
            if converged:
                break
        if optimisation is None or last_evaluation is None:
            raise RuntimeError("reduced oracle did not perform an optimisation")
        if problem.reduced:
            final_state = np.concatenate(
                (
                    np.zeros(problem.displacement_unknown_count, dtype=np.float64),
                    final_variables,
                )
            )
            increment = problem.unpack_state(final_state)[1]
        else:
            increment = final_variables.reshape(problem.plastic_increment_shape) * (
                config.plastic_increment_variable_scale
            )
        if converged and commit_on_success:
            material.commit()
        else:
            material.revert()
        return ExperimentalOracleIncrementResult(
            converged=converged,
            displacement=last_displacement,
            equivalent_plastic_increment=increment,
            linearisation=last_evaluation.linearisation,
            objective=last_evaluation.value,
            dic_misfit=last_evaluation.dic_misfit,
            ludwik_prior=last_evaluation.ludwik_prior,
            spatial_regularisation=last_evaluation.spatial_regularisation,
            temporal_regularisation=last_evaluation.temporal_regularisation,
            augmented_equilibrium=last_evaluation.augmented_equilibrium,
            equilibrium_rms=last_evaluation.equilibrium_rms,
            augmented_iterations=tuple(history),
            constitutive_rejections=constitutive_rejections,
            message=str(optimisation.message),
        )
    except Exception:
        material.revert()
        raise


def solve_experimental_mechanical_oracle_history(
    *,
    material: DrivenJ2PlaneStressBatch,
    kinematics: DiscreteKinematics2D,
    measured_displacement_history: ArrayLike,
    whitener: DICSpectralWhitener,
    ludwik_increment_history: ArrayLike,
    initial_displacement_history: ArrayLike | None = None,
    displacement_warm_start: ExperimentalOracleWarmStart | None = None,
    progress_callback: ExperimentalOracleProgress | None = None,
    solution_method: str = "augmented",
    weights: ExperimentalOracleObjectiveWeights | None = None,
    config: ExperimentalOracleOptimizationConfig | None = None,
    time_increments: ArrayLike | float = 1.0,
    plastic_basis: ArrayLike | None = None,
) -> ExperimentalOracleHistoryResult:
    """Solve a DIC history sequentially, committing only accepted increments."""

    measured = np.asarray(measured_displacement_history, dtype=np.float64)
    if measured.ndim != 4 or measured.shape[-1] != 2 or measured.shape[0] < 2:
        raise ValueError(
            "measured_displacement_history must have shape (states, nx+1, ny+1, 2)"
        )
    if tuple(measured.shape[1:]) != whitener.field_shape:
        raise ValueError("DIC history and whitener field shapes must match")
    sample_shape = kinematics.strain_samples(measured[0]).shape[:-1]
    if plastic_basis is not None:
        basis = np.asarray(plastic_basis, dtype=np.float64)
        if basis.ndim != 2 or basis.shape[0] != int(np.prod(sample_shape)):
            raise ValueError("plastic_basis has an incompatible plastic-point dimension")
        plastic_basis = basis
    ludwik = np.asarray(ludwik_increment_history, dtype=np.float64)
    expected_ludwik_shape = (measured.shape[0] - 1, *sample_shape)
    if ludwik.shape != expected_ludwik_shape:
        raise ValueError(
            f"ludwik_increment_history must have shape {expected_ludwik_shape}"
        )
    if np.any(ludwik < 0.0) or not np.isfinite(ludwik).all():
        raise ValueError("ludwik_increment_history must be finite and non-negative")
    initial_history = measured
    if initial_displacement_history is not None:
        initial_history = np.asarray(initial_displacement_history, dtype=np.float64)
        if initial_history.shape != measured.shape or not np.isfinite(initial_history).all():
            raise ValueError(
                "initial_displacement_history must be finite with the measured history shape"
            )
    dt = np.asarray(time_increments, dtype=np.float64)
    try:
        dt = np.broadcast_to(dt, (measured.shape[0] - 1,)).copy()
    except ValueError as error:
        raise ValueError("time_increments must broadcast to the increment count") from error
    if np.any(dt <= 0.0) or not np.isfinite(dt).all():
        raise ValueError("time_increments must be finite and positive")
    if solution_method not in {"augmented", "reduced"}:
        raise ValueError("solution_method must be 'augmented' or 'reduced'")

    accepted_displacements = [measured[0].copy()]
    accepted_increments: list[FloatArray] = []
    accepted_peeq = [np.zeros(sample_shape, dtype=np.float64)]
    increment_results: list[ExperimentalOracleIncrementResult] = []
    previous_increment = np.zeros(sample_shape, dtype=np.float64)
    failed_increment: int | None = None

    for index in range(measured.shape[0] - 1):
        initial_displacement = np.asarray(
            accepted_displacements[-1]
            if solution_method == "reduced"
            else initial_history[index + 1],
            dtype=np.float64,
        )
        if displacement_warm_start is not None:
            initial_displacement = np.asarray(
                displacement_warm_start(
                    ExperimentalOracleWarmStartRequest(
                        increment_index=index + 1,
                        material=material,
                        kinematics=kinematics,
                        measured_displacement=measured[index + 1],
                        ludwik_increment=ludwik[index],
                        initial_displacement=initial_displacement,
                        time_increment=float(dt[index]),
                    )
                ),
                dtype=np.float64,
            )
            if (
                initial_displacement.shape != measured[index + 1].shape
                or not np.isfinite(initial_displacement).all()
            ):
                raise ValueError(
                    "displacement_warm_start must return a finite field with "
                    "the measured displacement shape"
                )
        increment_solver = (
            solve_experimental_mechanical_oracle_increment
            if solution_method == "augmented"
            else solve_experimental_mechanical_oracle_reduced_increment
        )
        result = increment_solver(
            material=material,
            kinematics=kinematics,
            measured_displacement=measured[index + 1],
            whitener=whitener,
            ludwik_increment=ludwik[index],
            initial_displacement=initial_displacement,
            initial_equivalent_plastic_increment=ludwik[index],
            previous_increment=previous_increment,
            weights=weights,
            config=config,
            time_increment=float(dt[index]),
            commit_on_success=True,
            plastic_basis=plastic_basis,
        )
        increment_results.append(result)
        if progress_callback is not None:
            progress_callback(index + 1, result)
        if not result.converged:
            failed_increment = index + 1
            break
        accepted_displacements.append(result.displacement.copy())
        previous_increment = result.equivalent_plastic_increment.copy()
        accepted_increments.append(previous_increment)
        peeq = np.asarray(
            result.linearisation.trial.observables["equivalent_plastic_strain"],
            dtype=np.float64,
        ).reshape(sample_shape)
        accepted_peeq.append(peeq.copy())

    increment_shape = (0, *sample_shape)
    increment_history = (
        np.stack(accepted_increments)
        if accepted_increments
        else np.empty(increment_shape, dtype=np.float64)
    )
    return ExperimentalOracleHistoryResult(
        completed=failed_increment is None,
        increments=tuple(increment_results),
        displacement_history=np.stack(accepted_displacements),
        equivalent_plastic_increment_history=increment_history,
        equivalent_plastic_strain_history=np.stack(accepted_peeq),
        failed_increment=failed_increment,
    )


def ludwik_increment_history_from_measured_displacement(
    *,
    material: PlaneStressMaterialBatch,
    kinematics: DiscreteKinematics2D,
    measured_displacement_history: ArrayLike,
    time_increments: ArrayLike | float = 1.0,
) -> FloatArray:
    """Build the explicit Ludwik prior by replaying the measured DIC history.

    The supplied material is dedicated to this prior replay and is committed
    after every state.  No global equilibrium is implied by this operation.
    """

    measured = np.asarray(measured_displacement_history, dtype=np.float64)
    if measured.ndim != 4 or measured.shape[-1] != 2 or measured.shape[0] < 2:
        raise ValueError(
            "measured_displacement_history must have shape (states, nx+1, ny+1, 2)"
        )
    sample_shape = kinematics.strain_samples(measured[0]).shape[:-1]
    if material.point_count != int(np.prod(sample_shape)):
        raise ValueError("material point count does not match the DIC kinematic samples")
    dt = np.asarray(time_increments, dtype=np.float64)
    try:
        dt = np.broadcast_to(dt, (measured.shape[0] - 1,)).copy()
    except ValueError as error:
        raise ValueError("time_increments must broadcast to the increment count") from error
    if np.any(dt <= 0.0) or not np.isfinite(dt).all():
        raise ValueError("time_increments must be finite and positive")

    previous_peeq = np.zeros(material.point_count, dtype=np.float64)
    increments: list[FloatArray] = []
    try:
        for index in range(measured.shape[0] - 1):
            strain = kinematics.strain_samples(measured[index + 1]).reshape(-1, 3)
            trial = material.evaluate(
                strain,
                time_increment=float(dt[index]),
                consistent_tangent=False,
            )
            if "equivalent_plastic_strain" not in trial.observables:
                raise KeyError("Ludwik prior material does not expose equivalent_plastic_strain")
            peeq = np.asarray(
                trial.observables["equivalent_plastic_strain"],
                dtype=np.float64,
            ).reshape(-1)
            increment = peeq - previous_peeq
            if np.min(increment) < -1.0e-12:
                raise ValueError("Ludwik replay produced a decreasing plastic history")
            increments.append(np.maximum(increment, 0.0).reshape(sample_shape))
            material.commit()
            previous_peeq = peeq.copy()
    except Exception:
        material.revert()
        raise
    return np.stack(increments)
