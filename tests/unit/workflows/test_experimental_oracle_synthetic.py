from __future__ import annotations

import numpy as np

from fem_inhouse.core.driven_j2 import DrivenJ2PlaneStressBatch
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.workflows.experimental_oracle_synthetic import (
    diagonal_localised_plastic_increment,
    solve_fixed_plastic_increment_equilibrium,
)


def _tensile_boundary(grid: StructuredGrid2D) -> np.ndarray:
    x = np.linspace(0.0, grid.length_x, grid.node_shape[0])[:, None]
    y = np.linspace(0.0, grid.length_y, grid.node_shape[1])[None, :]
    displacement = np.empty((*grid.node_shape, 2), dtype=np.float64)
    displacement[..., 0] = -0.001 * x
    displacement[..., 1] = 0.008 * y
    return displacement


def test_diagonal_increment_is_positive_localised_and_subcell_consistent() -> None:
    grid = StructuredGrid2D(9, 7, 0.9, 0.7)
    increment = diagonal_localised_plastic_increment(
        grid,
        points_per_pixel=2,
        background=1.0e-4,
        amplitude=1.0e-3,
        width_pixels=1.0,
    )

    assert increment.shape == (9, 7, 2)
    assert float(increment.min()) >= 1.0e-4
    assert float(increment.max()) > 8.0e-4
    np.testing.assert_array_equal(increment[..., 0], increment[..., 1])


def test_fixed_increment_newton_equilibrates_a_localised_band() -> None:
    grid = StructuredGrid2D(5, 5, 0.5, 0.5)
    kinematics = TwoSubcellDiagnostic2D(grid)
    material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    boundary = _tensile_boundary(grid)
    increment = diagonal_localised_plastic_increment(
        grid,
        points_per_pixel=kinematics.points_per_pixel,
        background=1.0e-4,
        amplitude=8.0e-4,
        width_pixels=1.0,
    )

    result = solve_fixed_plastic_increment_equilibrium(
        material=material,
        kinematics=kinematics,
        boundary_displacement=boundary,
        equivalent_plastic_increment=increment,
        equilibrium_rms_tolerance=1.0e-8,
    )

    assert result.newton_iterations > 0
    assert result.equilibrium_rms <= 1.0e-8
    assert np.max(np.abs(result.displacement - boundary)) > 1.0e-8
    np.testing.assert_array_equal(
        material.committed_equivalent_plastic_strain,
        np.zeros(material.point_count),
    )
