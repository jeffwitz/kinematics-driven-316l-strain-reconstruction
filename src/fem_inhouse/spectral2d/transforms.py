"""Discrete transforms for full-Dirichlet fluctuation fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import dstn, idstn

from fem_inhouse.spectral2d.grid import StructuredGrid2D

FloatArray = NDArray[np.float64]


class TransformPlan2D(Protocol):
    """Frequency interface consumed by discrete operator symbol builders."""

    @property
    def frequencies_x(self) -> FloatArray: ...

    @property
    def frequencies_y(self) -> FloatArray: ...

    def forward_displacement(self, interior_field: ArrayLike) -> FloatArray: ...

    def inverse_displacement(self, transformed_field: ArrayLike) -> FloatArray: ...


@dataclass(frozen=True, slots=True)
class FullDirichletDSTIPlan2D:
    """Orthonormal DST-I plan on the interior nodes of a full-Dirichlet grid."""

    grid: StructuredGrid2D

    def __post_init__(self) -> None:
        if self.grid.nx < 2 or self.grid.ny < 2:
            raise ValueError("DST-I full-Dirichlet plans need at least 2 pixels per axis")

    @property
    def frequencies_x(self) -> FloatArray:
        return np.pi * np.arange(1, self.grid.nx, dtype=np.float64) / self.grid.nx

    @property
    def frequencies_y(self) -> FloatArray:
        return np.pi * np.arange(1, self.grid.ny, dtype=np.float64) / self.grid.ny

    @property
    def interior_shape(self) -> tuple[int, int]:
        return self.grid.interior_shape

    def forward_displacement(self, interior_field: ArrayLike) -> FloatArray:
        """Transform an interior scalar or component field with DST-I."""

        values = np.asarray(interior_field, dtype=np.float64)
        if values.shape[:2] != self.interior_shape:
            raise ValueError(f"expected leading shape {self.interior_shape}, got {values.shape}")
        if values.ndim == 2:
            return np.asarray(dstn(values, type=1, norm="ortho"), dtype=np.float64)
        transformed = np.empty_like(values)
        for component in range(values.shape[2]):
            transformed[..., component] = dstn(values[..., component], type=1, norm="ortho")
        return transformed

    def inverse_displacement(self, transformed_field: ArrayLike) -> FloatArray:
        """Invert a scalar or component field transformed by DST-I."""

        values = np.asarray(transformed_field, dtype=np.float64)
        if values.shape[:2] != self.interior_shape:
            raise ValueError(f"expected leading shape {self.interior_shape}, got {values.shape}")
        if values.ndim == 2:
            return np.asarray(idstn(values, type=1, norm="ortho"), dtype=np.float64)
        recovered = np.empty_like(values)
        for component in range(values.shape[2]):
            recovered[..., component] = idstn(values[..., component], type=1, norm="ortho")
        return recovered

    def extract_interior(self, nodal_field: ArrayLike) -> FloatArray:
        """Extract the zero-boundary fluctuation field from nodal values."""

        values = np.asarray(nodal_field, dtype=np.float64)
        if values.shape[:2] != self.grid.node_shape:
            raise ValueError(f"expected nodal shape {self.grid.node_shape}, got {values.shape}")
        return values[1:-1, 1:-1, ...].copy()

    def embed_interior(self, interior_field: ArrayLike) -> FloatArray:
        """Embed an interior field into a zero-boundary nodal array."""

        values = np.asarray(interior_field, dtype=np.float64)
        if values.shape[:2] != self.interior_shape:
            raise ValueError(f"expected leading shape {self.interior_shape}, got {values.shape}")
        embedded = np.zeros(self.grid.node_shape + values.shape[2:], dtype=np.float64)
        embedded[1:-1, 1:-1, ...] = values
        return embedded
