"""Mathematical checks for the compact SRIX ``abs(dg)`` regularisation."""

from __future__ import annotations

import numpy as np
import pytest


def q_delta(g: float, delta: float) -> float:
    if delta <= 0.0 or abs(g) >= delta:
        return abs(g)
    t = abs(g) / delta
    return delta * (3.0 * t**2 - 3.0 * t**3 + t**4)


def q_prime(g: float, delta: float) -> float:
    if delta <= 0.0 or abs(g) >= delta:
        return 1.0 if g > 0.0 else -1.0
    u = abs(g)
    return 6.0 * g / delta - 9.0 * g * u / delta**2 + 4.0 * g**3 / delta**3


def backstrain(g: float, a: float, d: float, theta: float, delta: float) -> float:
    q = q_delta(g, delta)
    return (g - d * a * q) / (1.0 + theta * d * q)


def backstrain_derivative(
    g: float, a: float, d: float, theta: float, delta: float,
) -> float:
    q = q_delta(g, delta)
    qp = q_prime(g, delta)
    numerator = g - d * a * q
    denominator = 1.0 + theta * d * q
    d_numerator = 1.0 - d * a * qp
    d_denominator = theta * d * qp
    return (d_numerator * denominator - numerator * d_denominator) / denominator**2


def test_compact_regularisation_is_even_and_zero_at_origin() -> None:
    for value in np.linspace(0.0, 1.0, 21):
        assert q_delta(value, 0.1) == pytest.approx(q_delta(-value, 0.1))
    assert q_delta(0.0, 0.1) == 0.0
    assert q_prime(0.0, 0.1) == 0.0


def test_compact_regularisation_is_monotone_and_exact_outside_band() -> None:
    values = np.linspace(0.0, 0.1, 101)
    regularised = np.array([q_delta(value, 0.1) for value in values])
    assert np.all(np.diff(regularised) >= 0.0)
    assert q_delta(0.1, 0.1) == pytest.approx(0.1)
    assert q_prime(0.1, 0.1) == pytest.approx(1.0)
    assert q_delta(0.2, 0.1) == pytest.approx(0.2)
    assert q_prime(0.2, 0.1) == pytest.approx(1.0)


@pytest.mark.parametrize("value", [-0.099, -0.05, -0.001, 0.001, 0.05, 0.099])
def test_compact_derivative_matches_central_difference(value: float) -> None:
    delta = 0.1
    step = 1.0e-8
    finite_difference = (q_delta(value + step, delta) - q_delta(value - step, delta)) / (
        2.0 * step
    )
    assert q_prime(value, delta) == pytest.approx(finite_difference, rel=1.0e-7)


def test_zero_delta_preserves_the_historical_abs_branch() -> None:
    for value in (-0.3, -1.0e-12, 0.0, 1.0e-12, 0.3):
        assert q_delta(value, 0.0) == abs(value)
    assert q_prime(-0.3, 0.0) == -1.0
    assert q_prime(0.3, 0.0) == 1.0


@pytest.mark.parametrize("value", [0.0, 1.0e-5, 0.005, 0.099, 0.1, 0.2])
def test_backstrain_derivative_matches_finite_difference(value: float) -> None:
    a, d, theta, delta = 0.02, 1500.0, 1.0, 0.1
    step = 1.0e-8
    finite_difference = (
        backstrain(value + step, a, d, theta, delta)
        - backstrain(value - step, a, d, theta, delta)
    ) / (2.0 * step)
    assert backstrain_derivative(value, a, d, theta, delta) == pytest.approx(
        finite_difference, rel=2.0e-6, abs=2.0e-6,
    )


def test_backstrain_derivative_is_not_the_unsmoothed_shortcut() -> None:
    value, a, d, theta, delta = 0.005, 0.02, 1500.0, 1.0, 0.1
    shortcut = (1.0 - d * a * q_prime(value, delta)) / (
        1.0 + theta * d * q_delta(value, delta)
    )**2
    assert abs(backstrain_derivative(value, a, d, theta, delta) - shortcut) > 1.0e-3
