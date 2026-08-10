"""Generic constitutive sensitivity adapters.

The finite-difference adapter is an oracle and compatibility path while an
MFront behaviour does not export its local implicit residual/Jacobian blocks.
It only assumes that a material response can be evaluated from a strain and a
parameter field.  Consequently the same code can validate J2, SRIX, Méric, or
another MFront law without knowing its internal variables.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Response = Callable[[FloatArray, FloatArray], tuple[FloatArray, FloatArray]]


@dataclass(frozen=True, slots=True)
class ConstitutiveFiniteDifferenceSensitivity:
    """Pointwise derivatives of stress and an arbitrary scalar/vector observable."""

    stress_parameter: FloatArray
    observable_strain: FloatArray
    observable_parameter: FloatArray


def finite_difference_sensitivities(
    response: Response,
    strain: ArrayLike,
    parameter: ArrayLike,
    *,
    base_stress: ArrayLike | None = None,
    base_observable: ArrayLike | None = None,
    strain_step: float,
    parameter_step: ArrayLike | float,
    central_parameter: bool = True,
    forward_strain: bool = True,
) -> ConstitutiveFiniteDifferenceSensitivity:
    """Differentiate a batched constitutive response without law knowledge.

    ``response(strain, parameter)`` must return ``(stress, observable)`` with
    matching leading batch dimensions.  The parameter may be scalar per point
    or have any trailing parameter dimension.  The observable may likewise be
    scalar or vector-valued.  The strain derivative is forward by default so
    that the already-computed base observable can be reused; parameter
    derivatives are central by default because the base state is not assumed
    to be a safe one-sided stencil.

    This function is intentionally not the production constitutive tangent:
    it is a generic fallback and validation oracle until the local implicit
    blocks described in ``implicit_sensitivities`` are exported by MFront.
    """

    strain_values = np.asarray(strain, dtype=np.float64)
    parameter_values = np.asarray(parameter, dtype=np.float64)
    if strain_values.ndim != 2 or strain_values.shape[1] == 0:
        raise ValueError("strain must have shape (points, components)")
    if parameter_values.shape[0] != strain_values.shape[0]:
        raise ValueError("parameter and strain must have the same point count")
    if not np.isfinite(strain_values).all() or not np.isfinite(parameter_values).all():
        raise ValueError("strain and parameter must be finite")
    if not np.isfinite(strain_step) or strain_step <= 0:
        raise ValueError("strain_step must be finite and positive")

    if base_stress is None or base_observable is None:
        stress_0, observable_0 = response(strain_values, parameter_values)
        stress_0 = np.asarray(stress_0, dtype=np.float64)
        observable_0 = np.asarray(observable_0, dtype=np.float64)
        if base_stress is not None:
            supplied_stress = np.asarray(base_stress, dtype=np.float64)
            if supplied_stress.shape != stress_0.shape:
                raise ValueError("base_stress has an incompatible shape")
            stress_0 = supplied_stress
        if base_observable is not None:
            supplied_observable = np.asarray(base_observable, dtype=np.float64)
            if supplied_observable.shape != observable_0.shape:
                raise ValueError("base_observable has an incompatible shape")
            observable_0 = supplied_observable
    else:
        stress_0 = np.asarray(base_stress, dtype=np.float64)
        observable_0 = np.asarray(base_observable, dtype=np.float64)

    scalar_parameter = parameter_values.ndim == 1
    parameter_values = parameter_values[:, None] if scalar_parameter else parameter_values
    parameter_step_values = np.asarray(parameter_step, dtype=np.float64)
    if np.any(parameter_step_values <= 0) or not np.isfinite(parameter_step_values).all():
        raise ValueError("parameter_step must be finite and positive")
    if scalar_parameter and parameter_step_values.ndim == 1:
        if parameter_step_values.size not in (1, parameter_values.shape[0]):
            raise ValueError("parameter_step has an incompatible point count")
        parameter_step_values = parameter_step_values[:, None]
    parameter_step_values = np.broadcast_to(parameter_step_values, parameter_values.shape)
    parameter_count = parameter_values.shape[-1]

    def parameter_argument(values: FloatArray) -> FloatArray:
        return values[:, 0] if scalar_parameter else values

    stress_parameter = np.empty((*stress_0.shape, parameter_count))
    observable_parameter = np.empty((*observable_0.shape, parameter_count))
    for component in range(parameter_count):
        plus = parameter_values.copy()
        plus[:, component] += parameter_step_values[:, component]
        stress_plus, observable_plus = response(
            strain_values, parameter_argument(plus)
        )
        if central_parameter:
            minus = parameter_values.copy()
            minus[:, component] -= parameter_step_values[:, component]
            stress_minus, observable_minus = response(
                strain_values, parameter_argument(minus)
            )
            denominator = 2.0 * parameter_step_values[:, component]
            stress_parameter[..., component] = (
                np.asarray(stress_plus) - np.asarray(stress_minus)
            ) / denominator.reshape((-1,) + (1,) * (stress_0.ndim - 1))
            observable_parameter[..., component] = (
                np.asarray(observable_plus) - np.asarray(observable_minus)
            ) / denominator.reshape((-1,) + (1,) * (observable_0.ndim - 1))
        else:
            denominator = parameter_step_values[:, component]
            stress_parameter[..., component] = (
                np.asarray(stress_plus) - stress_0
            ) / denominator.reshape((-1,) + (1,) * (stress_0.ndim - 1))
            observable_parameter[..., component] = (
                np.asarray(observable_plus) - observable_0
            ) / denominator.reshape((-1,) + (1,) * (observable_0.ndim - 1))

    observable_strain = np.empty((*observable_0.shape, strain_values.shape[1]))
    for component in range(strain_values.shape[1]):
        plus_strain = strain_values.copy()
        plus_strain[:, component] += strain_step
        _, observable_plus = response(plus_strain, parameter_argument(parameter_values))
        if forward_strain:
            observable_strain[..., component] = (
                np.asarray(observable_plus) - observable_0
            ) / strain_step
        else:
            minus_strain = strain_values.copy()
            minus_strain[:, component] -= strain_step
            _, observable_minus = response(
                minus_strain, parameter_argument(parameter_values)
            )
            observable_strain[..., component] = (
                np.asarray(observable_plus) - np.asarray(observable_minus)
            ) / (2.0 * strain_step)

    return ConstitutiveFiniteDifferenceSensitivity(
        stress_parameter=stress_parameter[..., 0]
        if scalar_parameter
        else stress_parameter,
        observable_strain=observable_strain,
        observable_parameter=observable_parameter[..., 0]
        if scalar_parameter
        else observable_parameter,
    )
