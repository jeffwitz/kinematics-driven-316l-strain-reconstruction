"""Plane-stress J2 plasticity for the supported Ludwik case study."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

HardeningMode = Literal["ludwik", "tabular"]
HardeningFunction = Callable[[NDArray], NDArray]
PLANE_STRESS_VON_MISES_METRIC = np.array([[1.0, -0.5, 0.0], [-0.5, 1.0, 0.0], [0.0, 0.0, 3.0]])


def von_mises(stress: NDArray) -> NDArray:
    """Return the plane-stress von Mises value for ``[..., S11, S22, S12]``."""

    values = np.asarray(stress)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError("stress final axis must contain S11, S22 and S12")
    return np.sqrt(
        np.maximum(
            values[..., 0] ** 2
            - values[..., 0] * values[..., 1]
            + values[..., 1] ** 2
            + 3 * values[..., 2] ** 2,
            0.0,
        )
    )


def make_hardening(
    exponent: float,
    mode: HardeningMode = "ludwik",
    plastic_strain_max: float = 0.2,
    point_count: int = 1_000,
    first_positive_strain: float = 1e-6,
) -> tuple[HardeningFunction, HardeningFunction]:
    """Create the hardening value and derivative used by return mapping."""

    if exponent <= 0:
        raise ValueError("exponent must be positive")
    if plastic_strain_max <= 0:
        raise ValueError("plastic_strain_max must be positive")
    if point_count < 3:
        raise ValueError("point_count must be at least 3")
    if not 0 < first_positive_strain < plastic_strain_max:
        raise ValueError("first_positive_strain must lie inside the tabulated range")

    if mode == "ludwik":

        def hardening(strain: NDArray) -> NDArray:
            return np.where(strain > 0, np.maximum(strain, 0.0) ** exponent, 0.0)

        def derivative(strain: NDArray) -> NDArray:
            values = np.asarray(strain, dtype=float)
            result = np.zeros_like(values)
            np.power(
                values,
                exponent - 1.0,
                out=result,
                where=values > 1e-15,
            )
            return exponent * result

        return hardening, derivative
    if mode == "tabular":
        knots = np.concatenate(
            (
                np.array([0.0]),
                np.linspace(
                    first_positive_strain,
                    plastic_strain_max,
                    point_count - 1,
                ),
            )
        )
        values = knots**exponent
        slopes = np.diff(values) / np.diff(knots)

        def hardening(strain: NDArray) -> NDArray:
            return np.interp(
                np.clip(strain, 0.0, knots[-1]),
                knots,
                values,
            )

        def derivative(strain: NDArray) -> NDArray:
            indices = np.clip(
                np.searchsorted(knots, strain, side="right") - 1,
                0,
                len(slopes) - 1,
            )
            return np.where(strain < knots[-1], slopes[indices], 0.0)

        return hardening, derivative
    raise ValueError(f"unknown hardening mode {mode!r}")


def return_mapping(
    trial_stress: NDArray,
    accumulated_plastic_strain: NDArray,
    initial_yield_stress: NDArray,
    hardening_coefficient: NDArray,
    hardening: HardeningFunction,
    cm11: float,
    cm12: float,
    cm33: float,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-10,
) -> tuple[NDArray, NDArray, NDArray]:
    """Apply vectorized guarded return mapping to independent Gauss points."""

    def yield_stress(strain: NDArray) -> NDArray:
        return initial_yield_stress + hardening_coefficient * hardening(strain)

    plastic = von_mises(trial_stress) - yield_stress(accumulated_plastic_strain) > 0
    stress = trial_stress.copy()
    plastic_increment = np.zeros_like(trial_stress)
    equivalent_increment = np.zeros(len(accumulated_plastic_strain))
    if not plastic.any():
        return stress, plastic_increment, equivalent_increment

    indices = np.where(plastic)[0]
    selected_trial = trial_stress[indices]
    selected_accumulated = accumulated_plastic_strain[indices]
    selected_yield = initial_yield_stress[indices]
    selected_coefficient = hardening_coefficient[indices]

    def selected_yield_stress(increment: NDArray) -> NDArray:
        return selected_yield + selected_coefficient * hardening(selected_accumulated + increment)

    def selected_stress(increment: NDArray) -> NDArray:
        current_yield = selected_yield_stress(increment)
        ratio = increment / np.where(current_yield > 1e-30, current_yield, 1e-30)
        a = 1 + ratio * cm11
        b = ratio * cm12
        c = 1 + ratio * cm33
        determinant = a * a - b * b
        return np.stack(
            (
                (a * selected_trial[:, 0] - b * selected_trial[:, 1]) / determinant,
                (a * selected_trial[:, 1] - b * selected_trial[:, 0]) / determinant,
                selected_trial[:, 2] / c,
            ),
            axis=1,
        )

    def residual(increment: NDArray) -> NDArray:
        return von_mises(selected_stress(increment)) - selected_yield_stress(increment)

    point_count = len(indices)
    lower = np.zeros(point_count)
    upper = np.full(point_count, 1e-4)
    for _ in range(80):
        open_bracket = residual(upper) > 0
        if not open_bracket.any():
            break
        upper = np.where(open_bracket, upper * 2.0, upper)

    increment = 0.5 * (lower + upper)
    for _ in range(max(max_iterations, 100)):
        current_residual = residual(increment)
        relative_residual = (
            np.abs(current_residual) / (selected_yield_stress(increment) + 1e-30)
        ).max()
        if relative_residual < tolerance:
            break
        lower = np.where(current_residual > 0, increment, lower)
        upper = np.where(current_residual <= 0, increment, upper)
        step = np.maximum(np.abs(increment), 1e-8) * 1e-7
        residual_derivative = (residual(increment + step) - current_residual) / step
        residual_derivative = np.where(
            np.abs(residual_derivative) < 1e-30,
            -1.0,
            residual_derivative,
        )
        candidate = increment - current_residual / residual_derivative
        outside_bracket = ~np.isfinite(candidate) | (candidate <= lower) | (candidate >= upper)
        increment = np.where(
            outside_bracket,
            0.5 * (lower + upper),
            candidate,
        )

    returned_stress = selected_stress(increment)
    equivalent_stress = von_mises(returned_stress)
    flow_direction = (returned_stress @ PLANE_STRESS_VON_MISES_METRIC.T) / np.maximum(
        equivalent_stress, 1e-30
    )[:, None]
    stress[indices] = returned_stress
    plastic_increment[indices] = increment[:, None] * flow_direction
    equivalent_increment[indices] = increment
    return stress, plastic_increment, equivalent_increment


def consistent_tangent(
    stress: NDArray,
    equivalent_increment: NDArray,
    accumulated_plastic_strain: NDArray,
    initial_yield_stress: NDArray,
    hardening_coefficient: NDArray,
    hardening: HardeningFunction,
    hardening_derivative: HardeningFunction,
    elasticity: NDArray,
    cm11: float,
    cm12: float,
    cm33: float,
) -> NDArray:
    """Return the analytical consistent tangent for plastic Gauss points."""

    current_yield = initial_yield_stress + hardening_coefficient * hardening(
        accumulated_plastic_strain + equivalent_increment
    )
    hardening_modulus = hardening_coefficient * hardening_derivative(
        accumulated_plastic_strain + equivalent_increment
    )
    ratio = equivalent_increment / np.where(
        current_yield > 1e-30,
        current_yield,
        1e-30,
    )
    a = 1 + ratio * cm11
    b = ratio * cm12
    c = 1 + ratio * cm33
    determinant = a * a - b * b
    safe_modulus = np.where(
        np.abs(hardening_modulus) > 1e-30,
        hardening_modulus,
        1e-30,
    )
    alpha = (1.0 - ratio * hardening_modulus) / (current_yield * safe_modulus)

    stress_11, stress_22, stress_12 = stress[:, 0], stress[:, 1], stress[:, 2]
    q0 = cm11 * stress_11 + cm12 * stress_22
    q1 = cm12 * stress_11 + cm11 * stress_22
    q2 = cm33 * stress_12
    alpha_q0, alpha_q1, alpha_q2 = alpha * q0, alpha * q1, alpha * q2
    w0 = (a * alpha_q0 - b * alpha_q1) / determinant
    w1 = (-b * alpha_q0 + a * alpha_q1) / determinant
    w2 = alpha_q2 / c

    equivalent_stress = von_mises(stress)
    safe_stress = np.where(equivalent_stress > 1e-30, equivalent_stress, 1e-30)
    n0 = (stress_11 - 0.5 * stress_22) / safe_stress
    n1 = (stress_22 - 0.5 * stress_11) / safe_stress
    n2 = 3 * stress_12 / safe_stress
    p0 = (a * n0 - b * n1) / determinant
    p1 = (-b * n0 + a * n1) / determinant
    p2 = n2 / c
    beta = n0 * w0 + n1 * w1 + n2 * w2

    zeros = np.zeros_like(a)
    inverse_a = np.stack(
        (
            np.stack((a / determinant, -b / determinant, zeros), axis=1),
            np.stack((-b / determinant, a / determinant, zeros), axis=1),
            np.stack((zeros, zeros, 1 / c), axis=1),
        ),
        axis=1,
    )
    vector_w = np.stack((w0, w1, w2), axis=1)
    vector_p = np.stack((p0, p1, p2), axis=1)
    correction = np.einsum("ni,nj->nij", vector_w, vector_p) / (1.0 + beta)[:, None, None]
    return np.einsum("nij,jk->nik", inverse_a - correction, elasticity)
