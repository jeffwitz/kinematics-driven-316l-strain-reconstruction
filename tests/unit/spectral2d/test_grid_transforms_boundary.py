import numpy as np
import pytest

from fem_inhouse.spectral2d import (
    FullDirichletDSTIPlan2D,
    HarmonicDirichletExtension2D,
    StructuredGrid2D,
    TransfiniteBoundaryInterpolation2D,
)


@pytest.mark.parametrize("nx, ny", [(4, 4), (5, 4), (4, 5), (7, 6), (12, 12)])
def test_dst_i_round_trip_and_zero_boundary_embedding(nx: int, ny: int) -> None:
    grid = StructuredGrid2D(nx, ny, 2.0, 3.0)
    plan = FullDirichletDSTIPlan2D(grid)
    rng = np.random.default_rng(nx * 100 + ny)
    field = rng.normal(size=(*grid.interior_shape, 2))

    transformed = plan.forward_displacement(field)
    recovered = plan.inverse_displacement(transformed)

    np.testing.assert_allclose(recovered, field, rtol=0.0, atol=1.0e-13)
    embedded = plan.embed_interior(recovered)
    np.testing.assert_allclose(plan.extract_interior(embedded), field, rtol=0.0, atol=1.0e-13)
    np.testing.assert_array_equal(embedded[0], 0.0)
    np.testing.assert_array_equal(embedded[-1], 0.0)
    np.testing.assert_array_equal(embedded[:, 0], 0.0)
    np.testing.assert_array_equal(embedded[:, -1], 0.0)


def test_dst_i_measured_round_trip_and_inner_product_contract() -> None:
    grid = StructuredGrid2D(7, 6, 2.0, 3.0)
    plan = FullDirichletDSTIPlan2D(grid)
    rng = np.random.default_rng(706)
    u = rng.normal(size=(*grid.interior_shape, 2))
    v = rng.normal(size=(*grid.interior_shape, 2))
    u_hat = plan.forward_displacement(u)
    v_hat = plan.forward_displacement(v)

    round_trip_error = np.linalg.norm(plan.inverse_displacement(u_hat) - u) / np.linalg.norm(u)
    inner_product_error = abs(np.vdot(u_hat, v_hat) - np.vdot(u, v)) / max(
        abs(np.vdot(u, v)), np.linalg.norm(u) * np.linalg.norm(v), 1.0e-30
    )

    assert round_trip_error < 1.0e-13
    assert inner_product_error < 1.0e-13


def test_dst_i_frequencies_match_full_dirichlet_grid() -> None:
    plan = FullDirichletDSTIPlan2D(StructuredGrid2D(4, 5, 2.0, 3.0))
    np.testing.assert_allclose(plan.frequencies_x, np.pi * np.arange(1, 4) / 4)
    np.testing.assert_allclose(plan.frequencies_y, np.pi * np.arange(1, 5) / 5)


def test_harmonic_extension_preserves_boundary_and_is_harmonic() -> None:
    grid = StructuredGrid2D(7, 6, 2.0, 3.0)
    x, y = grid.coordinates
    boundary = np.zeros((*grid.node_shape, 2))
    boundary[..., 0] = x[:, None] ** 2 + y[None, :]
    boundary[..., 1] = x[:, None] - 2.0 * y[None, :]

    extended = HarmonicDirichletExtension2D().extend(boundary, grid)
    np.testing.assert_allclose(extended[0], boundary[0])
    np.testing.assert_allclose(extended[-1], boundary[-1])
    np.testing.assert_allclose(extended[:, 0], boundary[:, 0])
    np.testing.assert_allclose(extended[:, -1], boundary[:, -1])

    laplacian_x = (
        extended[2:, 1:-1, 0] - 2.0 * extended[1:-1, 1:-1, 0] + extended[:-2, 1:-1, 0]
    ) / grid.spacing_x**2 + (
        extended[1:-1, 2:, 0] - 2.0 * extended[1:-1, 1:-1, 0] + extended[1:-1, :-2, 0]
    ) / grid.spacing_y**2
    np.testing.assert_allclose(laplacian_x, 0.0, atol=1.0e-12)


def test_transfinite_extension_preserves_boundary() -> None:
    grid = StructuredGrid2D(4, 5, 1.0, 1.0)
    boundary = np.zeros((*grid.node_shape, 2))
    boundary[0, :, 0] = 1.0
    boundary[-1, :, 1] = 2.0

    extended = TransfiniteBoundaryInterpolation2D().extend(boundary, grid)

    np.testing.assert_allclose(extended[0], boundary[0])
    np.testing.assert_allclose(extended[-1], boundary[-1])
    np.testing.assert_allclose(extended[:, 0], boundary[:, 0])
    np.testing.assert_allclose(extended[:, -1], boundary[:, -1])
