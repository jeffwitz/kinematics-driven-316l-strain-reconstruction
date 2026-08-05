from __future__ import annotations

import importlib.util

import pytest

from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.transform_factory import create_full_dirichlet_dsti_plan
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig


def test_factory_rejects_unknown_backend() -> None:
    config = object.__new__(SpectralTransformConfig)
    object.__setattr__(config, "backend", "invalid")
    object.__setattr__(config, "workers", 1)
    object.__setattr__(config, "fftw_planner_effort", "estimate")
    object.__setattr__(config, "fftw_planning_time_limit_s", None)
    object.__setattr__(config, "fftw_wisdom_directory", None)
    object.__setattr__(config, "fftw_use_wisdom", False)
    with pytest.raises(ValueError, match="unsupported transform backend"):
        create_full_dirichlet_dsti_plan(StructuredGrid2D(4, 4, 1.0, 1.0), config)


def test_factory_has_no_silent_fftw_fallback() -> None:
    config = SpectralTransformConfig(backend="fftw", fftw_planner_effort="estimate")
    if importlib.util.find_spec("pyfftw") is not None:
        pytest.skip("pyFFTW is installed")
    with pytest.raises(ImportError, match="pyFFTW is not installed"):
        create_full_dirichlet_dsti_plan(StructuredGrid2D(4, 4, 1.0, 1.0), config)
