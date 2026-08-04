"""Rectangular grids used by the spectral mechanics discretisation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StructuredGrid2D:
    """Pixel grid with nodal displacement locations.

    Array axes follow the repository convention: ``(x, y)``.  A field with
    two displacement components therefore has shape ``(nx + 1, ny + 1, 2)``.
    """

    nx: int
    ny: int
    length_x: float
    length_y: float

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx and ny must be positive")
        if self.length_x <= 0.0 or self.length_y <= 0.0:
            raise ValueError("domain lengths must be positive")

    @property
    def node_shape(self) -> tuple[int, int]:
        return self.nx + 1, self.ny + 1

    @property
    def pixel_shape(self) -> tuple[int, int]:
        return self.nx, self.ny

    @property
    def interior_shape(self) -> tuple[int, int]:
        return max(self.nx - 1, 0), max(self.ny - 1, 0)

    @property
    def spacing_x(self) -> float:
        return self.length_x / self.nx

    @property
    def spacing_y(self) -> float:
        return self.length_y / self.ny

    @property
    def coordinates(self) -> tuple[object, object]:
        """Return nodal coordinates without imposing a numerical backend."""

        import numpy as np

        return (
            np.arange(self.nx + 1, dtype=np.float64) * self.spacing_x,
            np.arange(self.ny + 1, dtype=np.float64) * self.spacing_y,
        )
