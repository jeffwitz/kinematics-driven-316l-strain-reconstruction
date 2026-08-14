from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.core.constitutive import PLANE_STRESS_VON_MISES_METRIC
from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
    inverse_gauge_square_root,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior

PIXEL_SIZE_MM = 0.00184
YOUNG = 205_000.0
POISSON = 0.30


class _Identity:
    """Stand-in for the DIC chain: the mechanics is what is under test here."""

    def apply(self, values):
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values):
        return np.asarray(values, dtype=np.float64)


def _grid(pixels: int) -> StructuredGrid2D:
    return StructuredGrid2D(
        pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels
    )


def _operator(pixels: int) -> TensorPlasticObservabilityOperator:
    return TensorPlasticObservabilityOperator.build(
        _grid(pixels),
        young_modulus_mpa=YOUNG,
        poisson_ratio=POISSON,
        transfer=_Identity(),
        whitener=_Identity(),
    )


def test_the_gauge_root_squares_back_to_the_inverse_von_mises_metric() -> None:
    """`H_loc = M^-1 / N` is the metric in which the norm of `Delta p n` is `Delta p`."""

    points = 7
    root = inverse_gauge_square_root(points)
    np.testing.assert_allclose(
        root @ root, PLANE_STRESS_VON_MISES_METRIC * points, rtol=1e-13, atol=1e-13
    )

    # The property the gauge exists for, checked on an actual J2 increment.
    stress = np.array([120.0, -45.0, 30.0])
    equivalent = float(np.sqrt(stress @ PLANE_STRESS_VON_MISES_METRIC @ stress))
    direction = PLANE_STRESS_VON_MISES_METRIC @ stress / equivalent
    increment = 3.5e-4 * direction
    gauge = np.linalg.inv(PLANE_STRESS_VON_MISES_METRIC)
    assert float(increment @ gauge @ increment) == pytest.approx(3.5e-4**2, rel=1e-12)


def test_the_coloured_sparse_stiffness_reproduces_the_assembled_residual() -> None:
    """The stencil colouring must recover `K = B^T C B` exactly, not approximately.

    Eighteen probes are enough only if the nodal coupling really is confined to
    a `3 x 3` neighbourhood. If the strain operator ever reached further, the
    recovered matrix would silently lose the outer entries, so the identity is
    checked against the kinematics rather than assumed from the stencil.
    """

    pixels = 6
    grid = _grid(pixels)
    kinematics = TwoSubcellDiagnostic2D(grid)
    operator = _operator(pixels)
    elasticity = plane_stress_elasticity(YOUNG, POISSON)
    weight = kinematics.sample_quadrature_weight

    generator = np.random.default_rng(4)
    displacement = generator.normal(size=operator.free_size) * 1e-6
    strain = np.asarray(kinematics.strain(unpack_interior(displacement, grid))).reshape(-1, 3)
    stress = strain @ elasticity
    expected = -pack_interior(
        kinematics.divergence_from_sample_stress(stress.reshape(pixels, pixels, 2, 3))
    ) / weight

    # solve_stiffness inverts the recovered sparse operator, so applying it to
    # the assembled product must return the displacement it came from.
    recovered = operator.solve_stiffness(expected)
    np.testing.assert_allclose(recovered, displacement, rtol=1e-8, atol=1e-16)


def test_the_operator_and_its_adjoint_are_a_transpose_pair() -> None:
    """Without this, a partial SVD converges to a well-formed wrong answer."""

    operator = _operator(6)
    generator = np.random.default_rng(5)
    plastic = generator.normal(size=operator.plastic_size)
    observed = generator.normal(size=operator.observation_size)
    forward = float(operator.matvec(plastic) @ observed)
    backward = float(plastic @ operator.rmatvec(observed))
    assert abs(forward - backward) <= 1e-10 * abs(forward)


def test_a_self_equilibrated_plastic_field_is_invisible() -> None:
    """The kernel that motivates the whole quotient, exhibited rather than argued.

    `G = B^T C` annihilates every plastic field whose induced stress is
    self-equilibrated, and those fields make up about seventy per cent of the
    space at M20. Such a field must produce exactly no observation.
    """

    pixels = 6
    grid = _grid(pixels)
    kinematics = TwoSubcellDiagnostic2D(grid)
    operator = _operator(pixels)
    elasticity = plane_stress_elasticity(YOUNG, POISSON)

    # C^-1 times a divergence-free stress field is such a plastic field; take one
    # from the null space of the assembled forcing operator on this small mesh.
    forcing = np.empty((operator.free_size, operator.plastic_size))
    unit = np.zeros(operator.plastic_size)
    for column in range(operator.plastic_size):
        unit[:] = 0.0
        unit[column] = 1.0
        stress = (unit.reshape(-1, 3) @ elasticity).reshape(pixels, pixels, 2, 3)
        forcing[:, column] = -pack_interior(
            kinematics.divergence_from_sample_stress(stress)
        ) / kinematics.sample_quadrature_weight
    _, _, right = np.linalg.svd(forcing, full_matrices=True)
    kernel_vector = right[-1]
    assert np.linalg.norm(forcing @ kernel_vector) < 1e-6 * np.linalg.norm(forcing)

    gauge_root = operator.inverse_gauge_root
    coordinates = (kernel_vector.reshape(-1, 3) @ np.linalg.inv(gauge_root)).reshape(-1)
    response = operator.matvec(coordinates)
    assert float(np.abs(response).max()) < 1e-9 * float(np.abs(coordinates).max())
