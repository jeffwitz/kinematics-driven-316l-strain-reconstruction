"""Coupling of the causal TANN-FCC batch to the two-state spectral solver.

The solver is reused untouched: the TANN implements the
`PlaneStressMaterialBatch` contract and the solver supplies equilibrium.
These tests mirror `test_newton_two_state.py` at the smallest scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.constitutive.tann_fcc import TannFCCBatch, TannFCCConfig
from fem_inhouse.core.fcc_interaction_matrix import SLIP_SYSTEMS
from fem_inhouse.spectral2d import EBISpectralSolverConfig, StructuredGrid2D
from fem_inhouse.spectral2d.newton_two_state import (
    solve_two_state_dirichlet_plane_stress,
)
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig


def _identity_systems(points: int) -> np.ndarray:
    tensors = np.empty((12, 3, 3))
    for index, (burgers, normal) in enumerate(SLIP_SYSTEMS):
        s = np.asarray(burgers, dtype=np.float64)
        m = np.asarray(normal, dtype=np.float64)
        s /= np.linalg.norm(s)
        m /= np.linalg.norm(m)
        tensors[index] = 0.5 * (np.outer(s, m) + np.outer(m, s))
    return np.tile(tensors, (points, 1, 1, 1))


def _boundary_history(grid: StructuredGrid2D, states: int) -> np.ndarray:
    x, y = grid.coordinates
    boundary = np.zeros((states, *grid.node_shape, 2))
    boundary[1:, ..., 0] = 0.01 * x[:, None]
    boundary[1:, ..., 1] = 0.005 * y[None, :]
    return boundary


def test_tann_small_dirichlet_converges() -> None:
    pytest.importorskip("pyfftw")
    grid = StructuredGrid2D(4, 4, 2.0, 2.0)
    material = TannFCCBatch(
        TannFCCConfig(),
        point_count=2 * 4 * 4,
        systems_global=_identity_systems(32),
    )
    result = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=material,
        boundary_displacement_history=_boundary_history(grid, 3),
        config=EBISpectralSolverConfig(
            relative_equilibrium_tolerance=1.0e-10,
            transform=SpectralTransformConfig(
                backend="scipy", fftw_planner_effort="estimate"
            ),
        ),
    )
    assert result.diagnostics.dimensionless_equilibrium_history[-1] < 1.0e-9
    assert np.isfinite(result.stress_in_plane_mpa).all()


def test_tann_state_advances_only_on_commit() -> None:
    pytest.importorskip("pyfftw")
    grid = StructuredGrid2D(4, 4, 2.0, 2.0)
    material = TannFCCBatch(
        TannFCCConfig(),
        point_count=2 * 4 * 4,
        systems_global=_identity_systems(32),
    )
    assert np.max(np.abs(material.committed_state)) == 0.0
    solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=material,
        boundary_displacement_history=_boundary_history(grid, 3),
        config=EBISpectralSolverConfig(
            relative_equilibrium_tolerance=1.0e-10,
            transform=SpectralTransformConfig(backend="scipy"),
        ),
    )
    # Two accepted increments: the committed state has advanced, and a
    # bare evaluate never mutates it.
    assert np.max(np.abs(material.committed_state)) > 0.0
    before = material.committed_state.copy()
    material.evaluate(np.zeros((32, 3)), compute_tangent=False)
    assert np.array_equal(material.committed_state, before)
