"""Common contracts and configuration for full-Dirichlet transform plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from fem_inhouse.spectral2d.transforms_scipy import FullDirichletDSTIPlan2D

FloatArray = NDArray[np.float64]
TransformBackend = Literal["scipy", "fftw"]
FFTWPlannerEffort = Literal["estimate", "measure", "patient"]


@dataclass(frozen=True, slots=True)
class SpectralTransformConfig:
    backend: TransformBackend = "scipy"
    workers: int = 1
    fftw_planner_effort: FFTWPlannerEffort = "measure"
    fftw_planning_time_limit_s: float | None = 2.0
    fftw_wisdom_directory: Path | None = None
    fftw_use_wisdom: bool = True

    def __post_init__(self) -> None:
        if self.backend not in {"scipy", "fftw"}:
            raise ValueError("transform backend must be 'scipy' or 'fftw'")
        if self.workers < 1:
            raise ValueError("transform workers must be at least one")
        if self.fftw_planner_effort not in {"estimate", "measure", "patient"}:
            raise ValueError("unsupported FFTW planner effort")
        if self.fftw_planning_time_limit_s is not None and self.fftw_planning_time_limit_s <= 0.0:
            raise ValueError("FFTW planning time limit must be positive or None")


@dataclass(frozen=True, slots=True)
class TransformDiagnostics:
    backend: str
    implementation: str
    interior_shape: tuple[int, int]
    batch_components: int
    dtype: str
    workers: int
    planner_effort: str | None
    wisdom_loaded: bool
    planning_seconds: float


class TransformPlan2D(Protocol):
    @property
    def frequencies_x(self) -> FloatArray: ...

    @property
    def frequencies_y(self) -> FloatArray: ...

    @property
    def backend_name(self) -> str: ...

    @property
    def diagnostics(self) -> TransformDiagnostics: ...

    def forward_displacement(self, interior_field: ArrayLike) -> FloatArray: ...

    def inverse_displacement(self, transformed_field: ArrayLike) -> FloatArray: ...

    def extract_interior(self, nodal_field: ArrayLike) -> FloatArray: ...

    def embed_interior(self, interior_field: ArrayLike) -> FloatArray: ...


class BufferedTransformPlan2D(TransformPlan2D, Protocol):
    def forward_into(self, source: ArrayLike, destination: FloatArray) -> None: ...

    def inverse_into(self, source: ArrayLike, destination: FloatArray) -> None: ...


def __getattr__(name: str) -> object:
    if name == "FullDirichletDSTIPlan2D":
        from fem_inhouse.spectral2d.transforms_scipy import FullDirichletDSTIPlan2D

        return FullDirichletDSTIPlan2D
    raise AttributeError(name)

__all__ = [
    "BufferedTransformPlan2D",
    "FFTWPlannerEffort",
    "FloatArray",
    "FullDirichletDSTIPlan2D",
    "SpectralTransformConfig",
    "TransformBackend",
    "TransformDiagnostics",
    "TransformPlan2D",
]
