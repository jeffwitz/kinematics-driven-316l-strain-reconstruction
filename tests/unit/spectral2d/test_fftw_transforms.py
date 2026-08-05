from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.transform_factory import create_full_dirichlet_dsti_plan
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig
from fem_inhouse.spectral2d.transforms_scipy import FullDirichletDSTIPlan2D

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pyfftw") is None,
    reason="pyFFTW optional dependency is not installed",
)


def test_fftw_matches_scipy_round_trip_and_forward() -> None:
    grid = StructuredGrid2D(12, 8, 12.0, 8.0)
    config = SpectralTransformConfig(
        backend="fftw",
        workers=1,
        fftw_planner_effort="estimate",
        fftw_use_wisdom=False,
    )
    fftw = create_full_dirichlet_dsti_plan(grid, config)
    scipy = FullDirichletDSTIPlan2D(grid)
    field = np.random.default_rng(42).normal(size=(*grid.interior_shape, 2))

    np.testing.assert_allclose(
        fftw.forward_displacement(field), scipy.forward_displacement(field), atol=1.0e-13
    )
    np.testing.assert_allclose(
        fftw.inverse_displacement(fftw.forward_displacement(field)), field, atol=1.0e-13
    )
    assert fftw.diagnostics.backend == "fftw"
