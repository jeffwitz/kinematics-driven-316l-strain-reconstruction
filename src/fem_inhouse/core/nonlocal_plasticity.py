"""Staggered micromorphic coupling between MFront J2 plasticity and Helmholtz."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.plane_stress_material import (
    ConstitutiveIntegrationError,
    InPlaneConstitutiveTrial,
)
from fem_inhouse.postprocessing.helmholtz import helmholtz_filter_element_field

FloatArray = NDArray[np.float64]


class NonlocalCouplingConvergenceError(ConstitutiveIntegrationError):
    """Raised when the staggered local/nonlocal constitutive solve fails."""


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

    def evaluate_in_plane(
        self,
        in_plane_strain: ArrayLike,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> InPlaneConstitutiveTrial: ...


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


@dataclass(slots=True)
class NonlocalFixedPointWorkspace:
    """Reusable arrays for the hot micromorphic fixed-point loop."""

    element_shape: tuple[int, int]
    gauss_points_per_element: int
    chi: FloatArray
    next_chi: FloatArray
    difference: FloatArray
    local_element_peeq: FloatArray
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
            local_element_peeq=np.empty(element_shape, dtype=np.float64, order="F"),
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
    relative_tolerance: float,
    maximum_iterations: int,
    maximum_helmholtz_residual: float,
    workspace: NonlocalFixedPointWorkspace | None = None,
) -> NonlocalCouplingEvaluation:
    """Solve the staggered ``p``--``chi`` fixed point from one committed state."""

    strain = np.asarray(in_plane_strain, dtype=np.float64)
    expected_points = element_shape[0] * element_shape[1] * gauss_points_per_element
    if strain.shape != (expected_points, 3):
        raise ValueError(f"in_plane_strain must have shape {(expected_points, 3)}")
    initial_chi = np.asarray(initial_nonlocal_peeq, dtype=np.float64)
    if initial_chi.shape != element_shape:
        raise ValueError(f"initial_nonlocal_peeq must have shape {element_shape}")
    if not np.isfinite(initial_chi).all() or np.any(initial_chi < 0):
        raise ValueError("initial_nonlocal_peeq must be finite and nonnegative")
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

    # At Hchi=0 the mechanical response is independent of chi. One source
    # evaluation and one final evaluation preserve exact local mechanics while
    # still producing a consistent nonlocal output field.
    iteration_limit = 1 if coupling_modulus_mpa == 0.0 else maximum_iterations
    for iteration in range(1, iteration_limit + 1):
        iterations = iteration
        material_batch.set_nonlocal_equivalent_plastic_strain(
            _gauss_values(
                chi,
                gauss_points_per_element,
                out=buffers.gauss_nonlocal_peeq,
            )
        )
        started = time.perf_counter()
        point_peeq = material_batch.evaluate_equivalent_plastic_strain(
            strain,
            time_increment=time_increment,
        )
        mfront_without_tangent_seconds += time.perf_counter() - started
        local_peeq = _element_average(
            point_peeq,
            element_shape=element_shape,
            gauss_points_per_element=gauss_points_per_element,
            name="equivalent_plastic_strain",
            out=buffers.local_element_peeq,
        )
        if np.any(local_peeq < -1e-14):
            raise NonlocalCouplingConvergenceError(
                "MFront returned a negative equivalent plastic strain"
            )
        started = time.perf_counter()
        filter_result = helmholtz_filter_element_field(
            local_peeq,
            length_scale_mm=length_scale_mm,
            spacing_x_mm=spacing_x_mm,
            spacing_y_mm=spacing_y_mm,
        )
        helmholtz_seconds += time.perf_counter() - started
        if filter_result.residual_relative > maximum_helmholtz_residual:
            raise NonlocalCouplingConvergenceError(
                "Helmholtz residual "
                f"{filter_result.residual_relative:.3e} exceeds "
                f"{maximum_helmholtz_residual:.3e}"
            )
        chi_star = filter_result.filtered_element_field
        if np.min(chi_star) < -1e-12:
            raise NonlocalCouplingConvergenceError(
                f"Helmholtz solution is negative: minimum={np.min(chi_star):.3e}"
            )
        np.maximum(chi_star, 0.0, out=chi_star)
        if coupling_modulus_mpa == 0.0:
            np.copyto(buffers.next_chi, chi_star)
        else:
            np.multiply(chi, 1.0 - relaxation, out=buffers.next_chi)
            np.multiply(chi_star, relaxation, out=buffers.difference)
            np.add(buffers.next_chi, buffers.difference, out=buffers.next_chi)
        next_chi = buffers.next_chi
        np.subtract(next_chi, chi, out=buffers.difference)
        relative_change = _mixed_relative_maximum_norm(
            buffers.difference,
            next_chi,
            chi_star,
        )
        np.copyto(chi, next_chi)
        if coupling_modulus_mpa == 0.0 or relative_change <= (
            relaxation * relative_tolerance
        ):
            break
    else:
        raise NonlocalCouplingConvergenceError(
            f"micromorphic fixed point did not converge in {maximum_iterations} iterations; "
            f"relative change={relative_change:.3e}"
        )

    material_batch.set_nonlocal_equivalent_plastic_strain(
        _gauss_values(
            chi,
            gauss_points_per_element,
            out=buffers.gauss_nonlocal_peeq,
        )
    )
    started = time.perf_counter()
    trial = material_batch.evaluate_in_plane(
        strain,
        time_increment=time_increment,
        consistent_tangent=True,
    )
    mfront_with_tangent_seconds += time.perf_counter() - started
    local_peeq = _element_average(
        trial.observables["equivalent_plastic_strain"],
        element_shape=element_shape,
        gauss_points_per_element=gauss_points_per_element,
        name="equivalent_plastic_strain",
        out=buffers.local_element_peeq,
    )
    yield_radius = _element_average(
        trial.observables["yield_surface_radius_mpa"],
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
    final_filter = helmholtz_filter_element_field(
        local_peeq,
        length_scale_mm=length_scale_mm,
        spacing_x_mm=spacing_x_mm,
        spacing_y_mm=spacing_y_mm,
    )
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
    )
