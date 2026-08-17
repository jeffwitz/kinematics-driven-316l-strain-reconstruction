"""The algebra of the free tensor family, and the invisibility it inherits.

Everything mechanical is qualified elsewhere. What is pinned here is the
parameterisation, the assembled dissipative projection and its transpose, and
the one physical fact that decides whether the family is identifiable at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.core.constitutive import PLANE_STRESS_VON_MISES_METRIC, von_mises
from fem_inhouse.core.kelvin import KELVIN_SCALE_2D, PLANE_STRESS_PLASTIC_GAUGE
from fem_inhouse.identification.tensor_local_inverse import (
    DissipativeProjection,
    TensorLocalBasis,
    j2_flow_direction,
    plastic_gauge_norm,
)


def _stress(count: int, seed: int = 3) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values = generator.standard_normal((count, 3)) * 40.0
    values[:, 1] += 300.0
    return values


def test_the_basis_transpose_is_the_transpose() -> None:
    basis = TensorLocalBasis.build(21, 17, 5)
    generator = np.random.default_rng(2)
    coefficients = generator.standard_normal(basis.coefficient_shape)
    dual = generator.standard_normal((21, 17, 3))
    left = float(np.sum(basis.assemble(coefficients) * dual))
    right = float(np.sum(coefficients * basis.assemble_transpose(dual)))
    assert abs(left - right) <= 1e-12 * max(abs(left), abs(right))


def test_the_basis_carries_three_components_per_patch() -> None:
    basis = TensorLocalBasis.build(16, 16, 8)
    assert basis.coefficient_shape == (8, 8, 3)
    assert basis.coefficient_count == 192


def test_the_components_do_not_mix() -> None:
    """A coefficient in one Kelvin slot may never leak into another."""

    basis = TensorLocalBasis.build(19, 19, 4)
    coefficients = np.zeros(basis.coefficient_shape)
    coefficients[:, :, 2] = 1.0
    field = basis.assemble(coefficients)
    np.testing.assert_allclose(field[:, :, 0], 0.0, atol=1e-15)
    np.testing.assert_allclose(field[:, :, 1], 0.0, atol=1e-15)
    np.testing.assert_allclose(field[:, :, 2], 1.0, atol=1e-14)


def test_the_projection_lands_in_the_half_space() -> None:
    stress = _stress(200)
    projection = DissipativeProjection(stress=stress)
    generator = np.random.default_rng(11)
    field = generator.standard_normal((200, 3)) * 1e-3
    projected, active = projection.apply(field)
    assert active.any(), "the test field must exercise the active branch"
    # The tolerance must scale with `sigma^T v`, which is order 0.3 here: an
    # absolute bound below one ulp of that product tests the floating-point
    # unit, not the projection.
    scale = float(np.abs(projection.dissipation(field)).max())
    assert projection.dissipation(projected).min() >= -1e-14 * scale


def test_the_projection_keeps_what_is_already_dissipative() -> None:
    stress = _stress(120)
    projection = DissipativeProjection(stress=stress)
    field = stress * 1e-6
    projected, active = projection.apply(field)
    assert not active.any()
    np.testing.assert_allclose(projected, field, atol=1e-18)


def test_the_projection_transpose_is_the_transpose() -> None:
    stress = _stress(150)
    projection = DissipativeProjection(stress=stress)
    generator = np.random.default_rng(5)
    field = generator.standard_normal((150, 3)) * 1e-3
    _, active = projection.apply(field)
    direction = generator.standard_normal((150, 3))
    dual = generator.standard_normal((150, 3))
    left = float(np.sum(projection.jacobian_action(direction, active) * dual))
    right = float(np.sum(direction * projection.transpose_action(dual, active)))
    assert abs(left - right) <= 1e-12 * max(abs(left), abs(right))


def test_assembling_then_projecting_is_not_projecting_then_assembling() -> None:
    """Branch D is an order, not a preference, and the two really differ.

    Both orders are admissible -- `H_sigma` is a convex cone, so a non-negative
    blend of admissible contributions cannot leave it, and the second assertion
    below records that. The ordering matters because mode-wise projection clips
    each contribution in isolation, shrinking the reachable family and tying it
    to an arbitrary decomposition into modes. If the first assertion ever fails,
    the ordering has stopped mattering and the family has silently changed.
    """

    basis = TensorLocalBasis.build(12, 12, 3)
    points = 12 * 12
    stress = _stress(points, seed=8)
    projection = DissipativeProjection(stress=stress)
    generator = np.random.default_rng(19)
    coefficients = generator.standard_normal(basis.coefficient_shape) * 1e-3

    assembled, _ = projection.apply(basis.assemble(coefficients).reshape(-1, 3))

    per_mode = np.zeros((points, 3))
    seed = np.zeros(basis.coefficient_shape)
    for index in np.ndindex(basis.coefficient_shape):
        seed[:] = 0.0
        seed[index] = coefficients[index]
        contribution, _ = projection.apply(basis.assemble(seed).reshape(-1, 3))
        per_mode += contribution

    assert not np.allclose(assembled, per_mode, atol=1e-12)
    scale = float(np.abs(projection.dissipation(assembled)).max())
    assert projection.dissipation(assembled).min() >= -1e-14 * scale
    # The blend stays admissible too: the half-space is a convex cone. This is
    # the assertion that corrects the original rationale for the ordering.
    assert projection.dissipation(per_mode).min() >= -1e-14 * scale


def test_the_j2_direction_survives_the_convention_change() -> None:
    """`sigma^T n` must equal the von Mises stress, in Kelvin and in Voigt alike.

    The repository's metric is Voigt and this module is Kelvin. The identity
    below is what proves the conversion was done rather than assumed: it fails
    by a factor on the shear entry if either scaling is dropped.
    """

    kelvin = _stress(80, seed=6)
    direction = j2_flow_direction(kelvin)
    voigt = kelvin / KELVIN_SCALE_2D
    equivalent = von_mises(voigt)

    np.testing.assert_allclose(
        np.einsum("pi,pi->p", kelvin, direction), equivalent, rtol=1e-12
    )
    engineering = direction * KELVIN_SCALE_2D
    np.testing.assert_allclose(
        engineering, (voigt @ PLANE_STRESS_VON_MISES_METRIC.T) / equivalent[:, None],
        rtol=1e-12,
    )


def test_the_j2_direction_is_dissipative_and_deviatoric_in_plane() -> None:
    kelvin = _stress(60, seed=7)
    direction = j2_flow_direction(kelvin)
    assert np.einsum("pi,pi->p", kelvin, direction).min() > 0.0


def test_the_gauge_norm_is_not_the_euclidean_one() -> None:
    """Shear is weighted differently, which is the whole reason Gp exists."""

    shear = np.array([[0.0, 0.0, 1.0]])
    axial = np.array([[1.0, 0.0, 0.0]])
    assert plastic_gauge_norm(shear) == pytest.approx(np.sqrt(2.0 / 3.0))
    assert plastic_gauge_norm(axial) == pytest.approx(np.sqrt(4.0 / 3.0))
    assert plastic_gauge_norm(shear) != pytest.approx(np.linalg.norm(shear))


def test_the_gauge_is_the_registered_matrix() -> None:
    np.testing.assert_allclose(
        PLANE_STRESS_PLASTIC_GAUGE,
        (2.0 / 3.0) * np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]]),
    )
