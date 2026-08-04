import numpy as np
import pytest

from fem_inhouse.spectral2d import (
    CellCenteredOnePoint2D,
    FullDirichletDSTIPlan2D,
    StructuredGrid2D,
    TwoSubcellDiagnostic2D,
)
from fem_inhouse.spectral2d.kinematics import _modal_symbols


@pytest.mark.parametrize("kinematics", [CellCenteredOnePoint2D, TwoSubcellDiagnostic2D])
def test_affine_displacement_has_exact_constant_strain(kinematics) -> None:
    grid = StructuredGrid2D(5, 4, 2.0, 3.0)
    x, y = grid.coordinates
    u = np.zeros((*grid.node_shape, 2))
    u[..., 0] = 0.3 * x[:, None] + 0.2 * y[None, :] + 1.0
    u[..., 1] = -0.4 * x[:, None] + 0.5 * y[None, :] - 2.0

    strain = kinematics(grid).strain(u)
    expected = np.array([0.3, 0.5, -0.2])
    np.testing.assert_allclose(
        strain,
        np.broadcast_to(expected, strain.shape),
        rtol=0.0,
        atol=1.0e-13,
    )


@pytest.mark.parametrize("kinematics", [CellCenteredOnePoint2D, TwoSubcellDiagnostic2D])
def test_divergence_is_the_negative_adjoint_of_strain(kinematics) -> None:
    grid = StructuredGrid2D(5, 4, 2.0, 3.0)
    operator = kinematics(grid)
    rng = np.random.default_rng(42 + operator.points_per_pixel)
    displacement = rng.normal(size=(*grid.node_shape, 2))
    stress_shape = (*grid.pixel_shape, 3)
    if operator.points_per_pixel == 2:
        stress_shape = (*grid.pixel_shape, 2, 3)
    stress = rng.normal(size=stress_shape)

    strain = operator.strain(displacement)
    divergence = operator.divergence(stress)
    area = grid.spacing_x * grid.spacing_y
    lhs = float(np.sum(stress * strain) * area / operator.points_per_pixel)
    rhs = float(np.sum(divergence * displacement))
    np.testing.assert_allclose(lhs + rhs, 0.0, rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize("kinematics", [CellCenteredOnePoint2D, TwoSubcellDiagnostic2D])
def test_full_dirichlet_kinematics_has_no_nonzero_constant_kernel(kinematics) -> None:
    grid = StructuredGrid2D(4, 4, 1.0, 1.0)
    operator = kinematics(grid)
    displacement = np.zeros((*grid.node_shape, 2))
    displacement[1:-1, 1:-1] = 1.0
    assert np.linalg.norm(operator.strain(displacement)) > 0.0


@pytest.mark.parametrize("kinematics", [CellCenteredOnePoint2D, TwoSubcellDiagnostic2D])
def test_reference_symbols_are_positive_and_consistent(kinematics) -> None:
    grid = StructuredGrid2D(5, 4, 2.0, 1.0)
    plan = FullDirichletDSTIPlan2D(grid)
    symbols = kinematics(grid).reference_operator_symbols(plan)
    assert symbols.laplacian.shape == (grid.nx - 1, grid.ny - 1)
    assert np.all(symbols.directional_x > 0.0)
    assert np.all(symbols.directional_y > 0.0)
    assert np.all(symbols.laplacian > 0.0)


@pytest.mark.parametrize("kinematics", [CellCenteredOnePoint2D, TwoSubcellDiagnostic2D])
@pytest.mark.parametrize("nx,ny", [(4, 4), (5, 4), (4, 5), (7, 6)])
def test_closed_form_symbols_match_modal_oracle(kinematics, nx, ny) -> None:
    grid = StructuredGrid2D(nx, ny, 2.0, 1.0)
    plan = FullDirichletDSTIPlan2D(grid)
    analytical = kinematics(grid).reference_operator_symbols(plan)
    modal = _modal_symbols(kinematics(grid), plan)
    for actual, expected in zip(analytical.as_tuple(), modal, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize("kinematics", [CellCenteredOnePoint2D, TwoSubcellDiagnostic2D])
@pytest.mark.parametrize("nx,ny", [(4, 4), (5, 4), (4, 5), (7, 6)])
def test_full_dirichlet_kinematic_matrix_has_full_column_rank(kinematics, nx, ny) -> None:
    grid = StructuredGrid2D(nx, ny, 2.0, 1.0)
    operator = kinematics(grid)
    columns = []
    for component in range(2):
        for i in range(1, nx):
            for j in range(1, ny):
                displacement = np.zeros((*grid.node_shape, 2))
                displacement[i, j, component] = 1.0
                columns.append(operator.strain(displacement).reshape(-1))
    matrix = np.column_stack(columns)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = np.count_nonzero(singular_values > 1.0e-11 * singular_values[0])
    assert rank == matrix.shape[1]


@pytest.mark.parametrize("kinematics", [CellCenteredOnePoint2D, TwoSubcellDiagnostic2D])
def test_reference_symbols_diagonalize_the_real_scalar_operator(kinematics) -> None:
    grid = StructuredGrid2D(5, 4, 2.0, 1.0)
    plan = FullDirichletDSTIPlan2D(grid)
    operator = kinematics(grid)
    symbols = operator.reference_operator_symbols(plan)
    rng = np.random.default_rng(123)
    transformed = rng.normal(size=plan.interior_shape)
    interior = plan.inverse_displacement(transformed)
    displacement = np.zeros((*grid.node_shape, 2))
    displacement[1:-1, 1:-1, 0] = interior
    strain = operator.strain(displacement)
    stress = np.zeros_like(strain)
    stress[..., 0] = strain[..., 0]
    stress[..., 2] = strain[..., 2]
    response = plan.forward_displacement(-operator.divergence(stress)[1:-1, 1:-1, 0])
    np.testing.assert_allclose(response, symbols.laplacian * transformed, rtol=0.0, atol=1.0e-12)
