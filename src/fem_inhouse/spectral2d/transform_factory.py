"""Factory for explicit full-Dirichlet transform backends."""

from __future__ import annotations

from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.transforms import (
    SpectralTransformConfig,
    TransformPlan2D,
)
from fem_inhouse.spectral2d.transforms_scipy import FullDirichletDSTIPlan2D as SciPyPlan


def create_full_dirichlet_dsti_plan(
    grid: StructuredGrid2D,
    config: SpectralTransformConfig,
) -> TransformPlan2D:
    if config.backend == "scipy":
        return SciPyPlan(grid, workers=config.workers)
    if config.backend == "fftw":
        from fem_inhouse.spectral2d.transforms_fftw import FullDirichletDSTIFFTWPlan2D

        return FullDirichletDSTIFFTWPlan2D(grid, config)
    raise ValueError(f"unsupported transform backend: {config.backend}")
