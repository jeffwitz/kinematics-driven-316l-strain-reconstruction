"""The algebra of the local-coefficient parameterisation.

The mechanics is qualified elsewhere; what needs pinning here is the
parameterisation and its transpose, because a wrong transpose produces a
plausible descent that converges to the wrong answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.identification.local_coefficient_inverse import (
    AdmissibleProjection,
    SeparableLocalBasis,
    _axis_operator,
    _legendre,
)


def test_legendre_matches_the_closed_forms() -> None:
    coordinate = np.linspace(-1.0, 1.0, 11)
    np.testing.assert_allclose(_legendre(0, coordinate), 1.0)
    np.testing.assert_allclose(_legendre(1, coordinate), coordinate)
    np.testing.assert_allclose(
        _legendre(2, coordinate), 0.5 * (3.0 * coordinate**2 - 1.0)
    )


def test_the_partition_of_unity_sums_to_one() -> None:
    operator = _axis_operator(37, 5, 0)
    np.testing.assert_allclose(operator[:, :, 0].sum(axis=1), 1.0, atol=1e-14)


def test_degree_zero_reproduces_the_bilinear_partition_of_unity() -> None:
    """The bench's field, exactly, so richer modes did not change old meanings."""

    pixels, patches = 23, 4
    basis = SeparableLocalBasis.build(pixels, pixels, patches, 0)
    generator = np.random.default_rng(3)
    coefficients = generator.standard_normal((patches, patches))

    x = np.linspace(0.0, patches - 1.0, pixels)
    nodes = np.arange(patches, dtype=np.float64)
    weight = np.clip(1.0 - np.abs(x[:, None] - nodes[None, :]), 0.0, None)
    weight /= weight.sum(axis=1, keepdims=True)
    expected = np.einsum("xi,ij,yj->xy", weight, coefficients, weight)

    np.testing.assert_allclose(
        basis.assemble(coefficients.reshape(patches, patches, 1, 1)),
        expected,
        atol=1e-15,
    )


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_the_basis_transpose_is_the_transpose(degree: int) -> None:
    basis = SeparableLocalBasis.build(19, 23, 5, degree)
    generator = np.random.default_rng(degree + 17)
    coefficients = generator.standard_normal(basis.coefficient_shape)
    dual = generator.standard_normal((19, 23))
    left = float(np.sum(basis.assemble(coefficients) * dual))
    right = float(np.sum(coefficients * basis.assemble_transpose(dual)))
    assert abs(left - right) <= 1e-12 * max(abs(left), abs(right))


def test_the_projection_clips_both_ways_and_reports_the_mask() -> None:
    projection = AdmissibleProjection(safety=0.5)
    field = np.array([[-1.0, 0.5, 4.0]])
    bound = np.array([[2.0, 2.0, 2.0]])
    projected, mask = projection.apply(field, bound)
    np.testing.assert_allclose(projected, [[0.0, 0.5, 1.0]])
    # The mask is the pass-through of the chain rule: zero wherever a bound is
    # active, because there the field no longer responds to the coefficients.
    np.testing.assert_allclose(mask, [[0.0, 1.0, 0.0]])


def test_the_projection_rejects_a_degenerate_safety_factor() -> None:
    with pytest.raises(ValueError, match="safety"):
        AdmissibleProjection(safety=1.0)
    with pytest.raises(ValueError, match="safety"):
        AdmissibleProjection(safety=0.0)


def test_the_basis_rejects_a_degenerate_patch_count() -> None:
    with pytest.raises(ValueError, match="patches"):
        _axis_operator(16, 1, 0)
    with pytest.raises(ValueError, match="degree"):
        _axis_operator(16, 4, -1)


def test_the_partition_of_unity_reproduces_linear_functions_exactly() -> None:
    """`sum_j w_j(x) (x - x_j) = 0`, which is why enrichment is rank-deficient.

    Linear reproduction is a virtue of the partition of unity and a defect of
    any basis that enriches it with linear modes: the enriched modes are then
    linearly dependent on the constants by construction. This identity is the
    mechanism, kept here so the next reader does not rediscover it through a
    singular Gauss-Newton step.
    """

    operator = _axis_operator(48, 6, 1)
    np.testing.assert_allclose(operator[:, :, 0].sum(axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(operator[:, :, 1].sum(axis=1), 0.0, atol=1e-15)


@pytest.mark.parametrize(
    ("degree", "expected_null"), [(0, 0), (1, 23), (2, 68)]
)
def test_enriched_local_modes_are_rank_deficient(degree: int, expected_null: int) -> None:
    """Measured, not assumed: degree zero is clean, enrichment is not.

    The mechanics adds no degeneracy of its own -- the 23 null directions found
    in the parameter-to-observable spectrum at degree one are exactly these.
    Until the enriched basis is orthogonalised, `q > 1` cannot be identified.
    """

    basis = SeparableLocalBasis.build(48, 48, 6, degree)
    count = basis.coefficient_count
    matrix = np.empty((48 * 48, count))
    seed = np.zeros(count)
    for index in range(count):
        seed[:] = 0.0
        seed[index] = 1.0
        matrix[:, index] = basis.assemble(seed.reshape(basis.coefficient_shape)).ravel()
    singular = np.linalg.svd(matrix, compute_uv=False)
    assert int(np.sum(singular <= singular[0] * 1e-12)) == expected_null


def test_the_coefficient_count_is_the_tensor_product() -> None:
    basis = SeparableLocalBasis.build(16, 16, 6, 1)
    # Degree one in each axis is four modes per patch, not three: the basis is a
    # tensor product, so `q = (degree + 1)^2`.
    assert basis.coefficient_shape == (6, 6, 2, 2)
    assert basis.coefficient_count == 144
