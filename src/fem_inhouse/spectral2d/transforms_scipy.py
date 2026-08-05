"""SciPy reference implementation of the full-Dirichlet DST-I plan."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import dstn, idstn

from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.transforms import FloatArray, TransformDiagnostics


@dataclass(slots=True)
class FullDirichletDSTIPlan2D:
    grid: StructuredGrid2D
    workers: int = 1
    _diagnostics: TransformDiagnostics = field(init=False)

    def __post_init__(self) -> None:
        if self.grid.nx < 2 or self.grid.ny < 2:
            raise ValueError("DST-I full-Dirichlet plans need at least 2 pixels per axis")
        if self.workers < 1:
            raise ValueError("transform workers must be at least one")
        self._diagnostics = TransformDiagnostics(
            backend="scipy",
            implementation="scipy.fft.dstn",
            interior_shape=self.interior_shape,
            batch_components=2,
            dtype="float64",
            workers=self.workers,
            planner_effort=None,
            wisdom_loaded=False,
            planning_seconds=0.0,
        )

    @property
    def frequencies_x(self) -> FloatArray:
        return np.pi * np.arange(1, self.grid.nx, dtype=np.float64) / self.grid.nx

    @property
    def frequencies_y(self) -> FloatArray:
        return np.pi * np.arange(1, self.grid.ny, dtype=np.float64) / self.grid.ny

    @property
    def interior_shape(self) -> tuple[int, int]:
        return self.grid.interior_shape

    @property
    def backend_name(self) -> str:
        return "scipy"

    @property
    def diagnostics(self) -> TransformDiagnostics:
        return self._diagnostics

    def _validate(self, values: ArrayLike) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[:2] != self.interior_shape or array.ndim not in {2, 3}:
            raise ValueError(f"expected leading shape {self.interior_shape}, got {array.shape}")
        if array.ndim == 3 and array.shape[2] != 2:
            raise ValueError("displacement transforms require two batch components")
        return array

    def forward_into(self, source: ArrayLike, destination: FloatArray) -> None:
        values = self._validate(source)
        if destination.shape != values.shape:
            raise ValueError(f"destination shape {destination.shape} does not match {values.shape}")
        destination[...] = dstn(
            values, type=1, norm="ortho", axes=(0, 1), workers=self.workers
        )

    def inverse_into(self, source: ArrayLike, destination: FloatArray) -> None:
        values = self._validate(source)
        if destination.shape != values.shape:
            raise ValueError(f"destination shape {destination.shape} does not match {values.shape}")
        destination[...] = idstn(
            values, type=1, norm="ortho", axes=(0, 1), workers=self.workers
        )

    def forward_displacement(self, interior_field: ArrayLike) -> FloatArray:
        values = self._validate(interior_field)
        transformed = np.empty_like(values)
        self.forward_into(values, transformed)
        return transformed

    def inverse_displacement(self, transformed_field: ArrayLike) -> FloatArray:
        values = self._validate(transformed_field)
        recovered = np.empty_like(values)
        self.inverse_into(values, recovered)
        return recovered

    def extract_interior(self, nodal_field: ArrayLike) -> FloatArray:
        values = np.asarray(nodal_field, dtype=np.float64)
        if values.shape[:2] != self.grid.node_shape:
            raise ValueError(f"expected nodal shape {self.grid.node_shape}, got {values.shape}")
        return values[1:-1, 1:-1, ...].copy()

    def embed_interior(self, interior_field: ArrayLike) -> FloatArray:
        values = self._validate(interior_field)
        embedded = np.zeros(self.grid.node_shape + values.shape[2:], dtype=np.float64)
        embedded[1:-1, 1:-1, ...] = values
        return embedded
