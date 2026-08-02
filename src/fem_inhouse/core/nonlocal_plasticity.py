"""Staggered micromorphic coupling between MFront J2 plasticity and Helmholtz."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.nonlocal_criteria import (
    EquivalentPlasticStrainHelmholtzCriterion,
    NonlocalRegularisationContext,
    ScalarNonlocalCriterion,
)
from fem_inhouse.core.plane_stress_material import (
    ConstitutiveIntegrationError,
    InPlaneConstitutiveTrial,
)

FloatArray = NDArray[np.float64]


class NonlocalCouplingConvergenceError(ConstitutiveIntegrationError):
    """Raised when the staggered local/nonlocal constitutive solve fails."""

    def __init__(
        self,
        message: str,
        *,
        reason: str = "unknown",
        iteration_history: tuple[NonlocalFixedPointIteration, ...] = (),
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.iteration_history = iteration_history


class NonlocalPlaneStressMaterialBatch(Protocol):
    """Plane-stress batch exposing the external micromorphic field."""

    @property
    def point_count(self) -> int: ...

    def set_nonlocal_equivalent_plastic_strain(self, values: ArrayLike) -> None: ...

    def evaluate_equivalent_plastic_strain(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
    ) -> FloatArray: ...

    def evaluate_nonlocal_state(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
    ) -> tuple[FloatArray, FloatArray]: ...

    def evaluate_in_plane(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> InPlaneConstitutiveTrial: ...


@dataclass(frozen=True, slots=True)
class NonlocalFixedPointIteration:
    """Diagnostics for one uncommitted Picard/Aitken evaluation."""

    iteration: int
    absolute_residual: float
    relative_residual: float
    relaxation: float
    maximum_local_peeq_change: float | None
    maximum_nonlocal_peeq_change: float
    minimum_nonlocal_hardening_mpa: float
    maximum_nonlocal_hardening_mpa: float
    minimum_yield_surface_radius_mpa: float
    maximum_yield_surface_radius_mpa: float
    helmholtz_residual_relative: float
    residual_direction_cosine: float | None
    acceleration_accepted: bool
    acceleration_rejected_for_growth: bool


@dataclass(frozen=True, slots=True)
class NonlocalCouplingEvaluation:
    """Converged constitutive trial and element-centred coupling fields."""

    constitutive_trial: InPlaneConstitutiveTrial
    nonlocal_peeq: FloatArray
    local_element_peeq: FloatArray
    mismatch: FloatArray
    nonlocal_hardening_mpa: FloatArray
    yield_surface_radius_mpa: FloatArray
    residual_field: FloatArray
    iterations: int
    relative_residual: float
    helmholtz_residual_relative: float
    mean_drift: float
    mfront_seconds: float
    mfront_without_tangent_seconds: float
    mfront_with_tangent_seconds: float
    helmholtz_seconds: float
    relaxation_strategy: str
    iteration_history: tuple[NonlocalFixedPointIteration, ...]


@dataclass(slots=True)
class NonlocalFixedPointWorkspace:
    """Reusable arrays for the hot micromorphic fixed-point loop."""

    element_shape: tuple[int, int]
    gauss_points_per_element: int
    chi: FloatArray
    next_chi: FloatArray
    difference: FloatArray
    raw_residual: FloatArray
    previous_residual: FloatArray
    local_element_peeq: FloatArray
    previous_local_element_peeq: FloatArray
    gauss_nonlocal_peeq: FloatArray

    @classmethod
    def create(
        cls,
        element_shape: tuple[int, int],
        gauss_points_per_element: int,
    ) -> NonlocalFixedPointWorkspace:
        if min(element_shape) < 1 or gauss_points_per_element < 1:
            raise ValueError("workspace dimensions must be positive")
        element_count = element_shape[0] * element_shape[1]
        return cls(
            element_shape=element_shape,
            gauss_points_per_element=gauss_points_per_element,
            chi=np.empty(element_shape, dtype=np.float64, order="F"),
            next_chi=np.empty(element_shape, dtype=np.float64, order="F"),
            difference=np.empty(element_shape, dtype=np.float64, order="F"),
            raw_residual=np.empty(element_shape, dtype=np.float64, order="F"),
            previous_residual=np.empty(element_shape, dtype=np.float64, order="F"),
            local_element_peeq=np.empty(element_shape, dtype=np.float64, order="F"),
            previous_local_element_peeq=np.empty(
                element_shape,
                dtype=np.float64,
                order="F",
            ),
            gauss_nonlocal_peeq=np.empty(
                element_count * gauss_points_per_element,
                dtype=np.float64,
            ),
        )


def _element_average(
    point_values: ArrayLike,
    *,
    element_shape: tuple[int, int],
    gauss_points_per_element: int,
    name: str,
    out: FloatArray | None = None,
) -> FloatArray:
    values = np.asarray(point_values, dtype=np.float64)
    element_count = element_shape[0] * element_shape[1]
    expected_shape = (element_count * gauss_points_per_element,)
    if values.shape != expected_shape:
        raise NonlocalCouplingConvergenceError(
            f"{name} has shape {values.shape}, expected {expected_shape}"
        )
    if not np.isfinite(values).all():
        raise NonlocalCouplingConvergenceError(f"{name} contains non-finite values")
    destination = (
        np.empty(element_shape, dtype=np.float64, order="F") if out is None else out
    )
    if destination.shape != element_shape:
        raise ValueError(f"out must have shape {element_shape}")
    np.mean(
        values.reshape(element_count, gauss_points_per_element),
        axis=1,
        out=destination.ravel(order="F"),
    )
    return destination


def _gauss_values(
    element_values: FloatArray,
    gauss_points_per_element: int,
    *,
    out: FloatArray | None = None,
) -> FloatArray:
    flattened = element_values.ravel(order="F")
    expected_shape = (flattened.size * gauss_points_per_element,)
    destination = np.empty(expected_shape, dtype=np.float64) if out is None else out
    if destination.shape != expected_shape:
        raise ValueError(f"out must have shape {expected_shape}")
    destination.reshape(flattened.size, gauss_points_per_element)[:, :] = flattened[:, None]
    return destination


def _mixed_relative_maximum_norm(
    difference: ArrayLike,
    *states: ArrayLike,
) -> float:
    """Return a mesh-independent mixed relative maximum norm.

    PEEQ is dimensionless and remains far below one in the supported case.
    The unit reference therefore supplies the absolute branch of the mixed
    criterion, while the state maximum supplies its relative branch. Unlike a
    raw global L2 norm, this definition does not tighten when the ROI contains
    more elements.
    """

    delta = np.asarray(difference, dtype=np.float64)
    scale = 1.0
    for state in states:
        values = np.asarray(state, dtype=np.float64)
        if values.shape != delta.shape:
            raise ValueError("fixed-point states must have identical shapes")
        scale = max(scale, float(np.max(np.abs(values), initial=0.0)))
    return float(np.max(np.abs(delta), initial=0.0) / scale)


def classify_fixed_point_history(
    history: tuple[NonlocalFixedPointIteration, ...],
) -> Literal[
    "insufficient_data",
    "nonpositive_yield_surface",
    "oscillating",
    "diverging",
    "slow_or_stagnating",
]:
    """Classify a failed fixed point without changing its numerical treatment."""

    if not history:
        return "insufficient_data"
    if min(item.minimum_yield_surface_radius_mpa for item in history) <= 0.0:
        return "nonpositive_yield_surface"
    if any(
        item.residual_direction_cosine is not None
        and item.residual_direction_cosine < -0.2
        for item in history
    ):
        return "oscillating"
    if len(history) < 2:
        return "insufficient_data"
    if history[-1].relative_residual > (
        1.25 * history[0].relative_residual
    ):
        return "diverging"
    return "slow_or_stagnating"


def evaluate_nonlocal_fixed_point(
    material_batch: NonlocalPlaneStressMaterialBatch,
    in_plane_strain: ArrayLike,
    *,
    time_increment: float,
    element_shape: tuple[int, int],
    gauss_points_per_element: int,
    initial_nonlocal_peeq: ArrayLike,
    length_scale_mm: float,
    spacing_x_mm: float,
    spacing_y_mm: float,
    coupling_modulus_mpa: float,
    relaxation: float,
    relaxation_strategy: Literal["fixed", "aitken"] = "fixed",
    minimum_relaxation: float = 0.05,
    maximum_relaxation: float = 0.8,
    aitken_residual_growth_factor: float = 1.25,
    relative_tolerance: float,
    maximum_iterations: int,
    maximum_helmholtz_residual: float,
    workspace: NonlocalFixedPointWorkspace | None = None,
    criterion: ScalarNonlocalCriterion | None = None,
) -> NonlocalCouplingEvaluation:
    """Solve the staggered ``p``--``chi`` fixed point from one committed state.

    The criterion owns the constitutive source, the external field, the spatial
    operator and the sign constraint. Relaxation, Aitken, the MFront
    transactions, the diagnostics and the final tangent stay here. Passing
    `None` selects the historical PEEQ-Helmholtz coupling, so existing callers
    take exactly the same path as before.
    """

    active_criterion: ScalarNonlocalCriterion = (
        EquivalentPlasticStrainHelmholtzCriterion() if criterion is None else criterion
    )
    if not active_criterion.supports_material(material_batch):
        raise TypeError(
            f"material batch does not support nonlocal criterion "
            f"{active_criterion.identifier!r}"
        )
    strain = np.asarray(in_plane_strain, dtype=np.float64)
    expected_points = element_shape[0] * element_shape[1] * gauss_points_per_element
    if strain.shape != (expected_points, 3):
        raise ValueError(f"in_plane_strain must have shape {(expected_points, 3)}")
    initial_chi = np.asarray(initial_nonlocal_peeq, dtype=np.float64)
    if initial_chi.shape != element_shape:
        raise ValueError(f"initial_nonlocal_peeq must have shape {element_shape}")
    # A signed criterion must not have its field clipped at zero, which is a
    # PEEQ-specific constraint rather than a property of the fixed point.
    if not np.isfinite(initial_chi).all() or (
        active_criterion.requires_nonnegative_field and np.any(initial_chi < 0)
    ):
        constraint = (
            "finite and nonnegative"
            if active_criterion.requires_nonnegative_field
            else "finite"
        )
        raise ValueError(f"initial nonlocal field must be {constraint}")
    if relaxation_strategy not in {"fixed", "aitken"}:
        raise ValueError("relaxation_strategy must be 'fixed' or 'aitken'")
    if not 0 < minimum_relaxation <= maximum_relaxation <= 1:
        raise ValueError("invalid relaxation bounds")
    if relaxation_strategy == "aitken" and not (
        minimum_relaxation <= relaxation <= maximum_relaxation
    ):
        raise ValueError("Aitken relaxation must lie inside its bounds")
    if aitken_residual_growth_factor <= 1:
        raise ValueError("aitken_residual_growth_factor must be greater than one")
    buffers = (
        NonlocalFixedPointWorkspace.create(element_shape, gauss_points_per_element)
        if workspace is None
        else workspace
    )
    if (
        buffers.element_shape != element_shape
        or buffers.gauss_points_per_element != gauss_points_per_element
    ):
        raise ValueError("workspace does not match the fixed-point discretisation")
    np.copyto(buffers.chi, initial_chi)
    chi = buffers.chi

    mfront_without_tangent_seconds = 0.0
    mfront_with_tangent_seconds = 0.0
    helmholtz_seconds = 0.0
    relative_change = float("inf")
    filter_result = None
    iterations = 0
    history: list[NonlocalFixedPointIteration] = []
    current_relaxation = relaxation
    previous_absolute_residual: float | None = None
    have_previous_residual = False
    have_previous_local_peeq = False

    # At Hchi=0 the mechanical response is independent of chi. One source
    # evaluation and one final evaluation preserve exact local mechanics while
    # still producing a consistent nonlocal output field.
    regularisation_context = NonlocalRegularisationContext(
        length_scale_mm=length_scale_mm,
        spacing_x_mm=spacing_x_mm,
        spacing_y_mm=spacing_y_mm,
    )
    iteration_limit = 1 if coupling_modulus_mpa == 0.0 else maximum_iterations
    for iteration in range(1, iteration_limit + 1):
        iterations = iteration
        active_criterion.set_external_field(
            material_batch,
            _gauss_values(
                chi,
                gauss_points_per_element,
                out=buffers.gauss_nonlocal_peeq,
            ),
        )
        started = time.perf_counter()
        point_peeq, point_yield_radius = active_criterion.evaluate_source_and_safety(
            material_batch,
            strain,
            time_increment=time_increment,
        )
        mfront_without_tangent_seconds += time.perf_counter() - started
        local_peeq = _element_average(
            point_peeq,
            element_shape=element_shape,
            gauss_points_per_element=gauss_points_per_element,
            name=active_criterion.source_name,
            out=buffers.local_element_peeq,
        )
        if active_criterion.requires_nonnegative_field and np.any(local_peeq < -1e-14):
            raise NonlocalCouplingConvergenceError(
                f"constitutive model returned a negative {active_criterion.source_name}",
                reason="negative_nonlocal_source",
                iteration_history=tuple(history),
            )
        if not np.isfinite(point_yield_radius).all():
            raise NonlocalCouplingConvergenceError(
                "MFront returned a non-finite yield surface radius",
                reason="nonfinite_yield_surface",
                iteration_history=tuple(history),
            )
        minimum_yield_radius = float(np.min(point_yield_radius))
        maximum_yield_radius = float(np.max(point_yield_radius))
        maximum_local_peeq_change = (
            float(
                np.max(
                    np.abs(local_peeq - buffers.previous_local_element_peeq),
                    initial=0.0,
                )
            )
            if have_previous_local_peeq
            else None
        )
        started = time.perf_counter()
        filter_result = active_criterion.regularise(
            local_peeq,
            regularisation_context,
        )
        helmholtz_seconds += time.perf_counter() - started
        if filter_result.residual_relative > maximum_helmholtz_residual:
            raise NonlocalCouplingConvergenceError(
                "Helmholtz residual "
                f"{filter_result.residual_relative:.3e} exceeds "
                f"{maximum_helmholtz_residual:.3e}",
                reason="helmholtz_residual",
                iteration_history=tuple(history),
            )
        chi_star = filter_result.filtered_element_field
        # Both the rejection and the clamp belong to a nonnegative field. A
        # signed criterion keeps its sign rather than being flattened at zero.
        if active_criterion.requires_nonnegative_field:
            if np.min(chi_star) < -1e-12:
                raise NonlocalCouplingConvergenceError(
                    f"regularised solution is negative: minimum={np.min(chi_star):.3e}",
                    reason="negative_regularised_solution",
                    iteration_history=tuple(history),
                )
            np.maximum(chi_star, 0.0, out=chi_star)
        np.subtract(chi_star, chi, out=buffers.raw_residual)
        absolute_residual = float(
            np.max(np.abs(buffers.raw_residual), initial=0.0)
        )
        raw_relative_residual = _mixed_relative_maximum_norm(
            buffers.raw_residual,
            chi,
            chi_star,
        )
        np.subtract(local_peeq, chi, out=buffers.difference)
        np.multiply(
            buffers.difference,
            coupling_modulus_mpa,
            out=buffers.difference,
        )
        minimum_hardening = float(np.min(buffers.difference))
        maximum_hardening = float(np.max(buffers.difference))
        residual_direction_cosine: float | None = None
        acceleration_accepted = False
        acceleration_rejected_for_growth = False
        if have_previous_residual:
            residual_product = float(
                np.vdot(
                    buffers.previous_residual.ravel(),
                    buffers.raw_residual.ravel(),
                )
            )
            residual_norm_product = float(
                np.linalg.norm(buffers.previous_residual)
                * np.linalg.norm(buffers.raw_residual)
            )
            if residual_norm_product > 0.0:
                residual_direction_cosine = residual_product / residual_norm_product
        if relaxation_strategy == "aitken" and have_previous_residual:
            if (
                previous_absolute_residual is not None
                and absolute_residual
                > aitken_residual_growth_factor * previous_absolute_residual
            ):
                current_relaxation = max(
                    minimum_relaxation,
                    0.5 * current_relaxation,
                )
                acceleration_rejected_for_growth = True
            else:
                np.subtract(
                    buffers.raw_residual,
                    buffers.previous_residual,
                    out=buffers.difference,
                )
                denominator = float(np.vdot(buffers.difference, buffers.difference))
                if denominator > np.finfo(np.float64).tiny:
                    candidate = (
                        -current_relaxation
                        * float(
                            np.vdot(
                                buffers.previous_residual,
                                buffers.difference,
                            )
                        )
                        / denominator
                    )
                    if np.isfinite(candidate):
                        current_relaxation = float(
                            np.clip(
                                candidate,
                                minimum_relaxation,
                                maximum_relaxation,
                            )
                        )
                        acceleration_accepted = True
        if coupling_modulus_mpa == 0.0:
            np.copyto(buffers.next_chi, chi_star)
        elif relaxation_strategy == "fixed":
            # Preserve the historical arithmetic path exactly.
            np.multiply(chi, 1.0 - relaxation, out=buffers.next_chi)
            np.multiply(chi_star, relaxation, out=buffers.difference)
            np.add(buffers.next_chi, buffers.difference, out=buffers.next_chi)
        else:
            np.multiply(
                buffers.raw_residual,
                current_relaxation,
                out=buffers.difference,
            )
            np.add(chi, buffers.difference, out=buffers.next_chi)
        next_chi = buffers.next_chi
        np.subtract(next_chi, chi, out=buffers.difference)
        maximum_nonlocal_peeq_change = float(
            np.max(np.abs(buffers.difference), initial=0.0)
        )
        relative_change = _mixed_relative_maximum_norm(
            buffers.difference,
            next_chi,
            chi_star,
        )
        history.append(
            NonlocalFixedPointIteration(
                iteration=iteration,
                absolute_residual=absolute_residual,
                relative_residual=raw_relative_residual,
                relaxation=(
                    1.0 if coupling_modulus_mpa == 0.0 else current_relaxation
                ),
                maximum_local_peeq_change=maximum_local_peeq_change,
                maximum_nonlocal_peeq_change=maximum_nonlocal_peeq_change,
                minimum_nonlocal_hardening_mpa=minimum_hardening,
                maximum_nonlocal_hardening_mpa=maximum_hardening,
                minimum_yield_surface_radius_mpa=minimum_yield_radius,
                maximum_yield_surface_radius_mpa=maximum_yield_radius,
                helmholtz_residual_relative=filter_result.residual_relative,
                residual_direction_cosine=residual_direction_cosine,
                acceleration_accepted=acceleration_accepted,
                acceleration_rejected_for_growth=(
                    acceleration_rejected_for_growth
                ),
            )
        )
        if minimum_yield_radius <= 0.0:
            raise NonlocalCouplingConvergenceError(
                "yield surface radius became non-positive during the "
                f"fixed point; minimum={minimum_yield_radius:.3e} MPa",
                reason="nonpositive_yield_surface",
                iteration_history=tuple(history),
            )
        np.copyto(buffers.previous_residual, buffers.raw_residual)
        np.copyto(buffers.previous_local_element_peeq, local_peeq)
        previous_absolute_residual = absolute_residual
        have_previous_residual = True
        have_previous_local_peeq = True
        np.copyto(chi, next_chi)
        converged_fixed = relative_change <= relaxation * relative_tolerance
        converged_aitken = raw_relative_residual <= relative_tolerance
        if coupling_modulus_mpa == 0.0 or (
            converged_fixed
            if relaxation_strategy == "fixed"
            else converged_aitken
        ):
            break
    else:
        history_tuple = tuple(history)
        classification = classify_fixed_point_history(history_tuple)
        raise NonlocalCouplingConvergenceError(
            f"micromorphic fixed point did not converge in {maximum_iterations} iterations; "
            f"relative change={relative_change:.3e}; classification={classification}",
            reason=classification,
            iteration_history=history_tuple,
        )

    active_criterion.set_external_field(
        material_batch,
        _gauss_values(
            chi,
            gauss_points_per_element,
            out=buffers.gauss_nonlocal_peeq,
        ),
    )
    started = time.perf_counter()
    trial = material_batch.evaluate_in_plane(
        strain,
        time_increment=time_increment,
        consistent_tangent=True,
    )
    mfront_with_tangent_seconds += time.perf_counter() - started
    local_peeq = _element_average(
        active_criterion.source_from_trial(trial),
        element_shape=element_shape,
        gauss_points_per_element=gauss_points_per_element,
        name=active_criterion.source_name,
        out=buffers.local_element_peeq,
    )
    yield_radius = _element_average(
        active_criterion.safety_from_trial(trial),
        element_shape=element_shape,
        gauss_points_per_element=gauss_points_per_element,
        name="yield_surface_radius_mpa",
        out=buffers.next_chi,
    )
    if np.any(yield_radius <= 0):
        raise NonlocalCouplingConvergenceError(
            f"yield surface radius must remain positive; minimum={np.min(yield_radius):.3e}"
        )
    started = time.perf_counter()
    final_filter = active_criterion.regularise(local_peeq, regularisation_context)
    helmholtz_seconds += time.perf_counter() - started
    if final_filter.residual_relative > maximum_helmholtz_residual:
        raise NonlocalCouplingConvergenceError(
            "final Helmholtz residual "
            f"{final_filter.residual_relative:.3e} exceeds "
            f"{maximum_helmholtz_residual:.3e}"
        )
    residual_field = chi - final_filter.filtered_element_field
    coupling_residual = _mixed_relative_maximum_norm(
        residual_field,
        chi,
        final_filter.filtered_element_field,
    )
    if coupling_modulus_mpa > 0 and coupling_residual > relative_tolerance:
        raise NonlocalCouplingConvergenceError(
            f"final micromorphic residual {coupling_residual:.3e} exceeds "
            f"{relative_tolerance:.3e}"
        )
    nonlocal_peeq = chi.copy(order="F")
    local_element_peeq = local_peeq.copy(order="F")
    yield_surface_radius = yield_radius.copy(order="F")
    mismatch = local_element_peeq - nonlocal_peeq
    return NonlocalCouplingEvaluation(
        constitutive_trial=trial,
        nonlocal_peeq=nonlocal_peeq,
        local_element_peeq=local_element_peeq,
        mismatch=mismatch,
        nonlocal_hardening_mpa=coupling_modulus_mpa * mismatch,
        yield_surface_radius_mpa=yield_surface_radius,
        residual_field=residual_field,
        iterations=iterations,
        relative_residual=coupling_residual,
        helmholtz_residual_relative=final_filter.residual_relative,
        mean_drift=final_filter.mean_drift,
        mfront_seconds=(
            mfront_without_tangent_seconds + mfront_with_tangent_seconds
        ),
        mfront_without_tangent_seconds=mfront_without_tangent_seconds,
        mfront_with_tangent_seconds=mfront_with_tangent_seconds,
        helmholtz_seconds=helmholtz_seconds,
        relaxation_strategy=relaxation_strategy,
        iteration_history=tuple(history),
    )
