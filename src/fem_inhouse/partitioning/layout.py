"""Partition a structured element grid into padded rectangular subdomains."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Bounds2D = tuple[int, int, int, int]
Slice2D = tuple[slice, slice]


@dataclass(frozen=True, slots=True)
class Partition:
    """One padded solve region and its non-overlapping core."""

    partition_id: int
    index_x: int
    index_y: int
    core_bounds: Bounds2D
    solve_bounds: Bounds2D

    @staticmethod
    def _shape(bounds: Bounds2D) -> tuple[int, int]:
        x0, x1, y0, y1 = bounds
        return x1 - x0, y1 - y0

    @staticmethod
    def _slice(bounds: Bounds2D) -> Slice2D:
        x0, x1, y0, y1 = bounds
        return slice(x0, x1), slice(y0, y1)

    @property
    def core_shape(self) -> tuple[int, int]:
        return self._shape(self.core_bounds)

    @property
    def solve_shape(self) -> tuple[int, int]:
        return self._shape(self.solve_bounds)

    @property
    def core_element_slice_global(self) -> Slice2D:
        return self._slice(self.core_bounds)

    @property
    def solve_element_slice_global(self) -> Slice2D:
        return self._slice(self.solve_bounds)

    @property
    def core_element_slice_local(self) -> Slice2D:
        core_x0, core_x1, core_y0, core_y1 = self.core_bounds
        solve_x0, _, solve_y0, _ = self.solve_bounds
        return (
            slice(core_x0 - solve_x0, core_x1 - solve_x0),
            slice(core_y0 - solve_y0, core_y1 - solve_y0),
        )

    @property
    def solve_node_slice_global(self) -> Slice2D:
        x0, x1, y0, y1 = self.solve_bounds
        return slice(x0, x1 + 1), slice(y0, y1 + 1)

    @property
    def owned_node_bounds_global(self) -> Bounds2D:
        """Return node bounds owned uniquely by this core.

        A partition owns its right/top core boundary nodes. Shared left/bottom
        nodes belong to the preceding partition, except at the global boundary.
        """

        core_x0, core_x1, core_y0, core_y1 = self.core_bounds
        owned_x0 = core_x0 if self.index_x == 0 else core_x0 + 1
        owned_y0 = core_y0 if self.index_y == 0 else core_y0 + 1
        return owned_x0, core_x1 + 1, owned_y0, core_y1 + 1

    @property
    def owned_node_slice_global(self) -> Slice2D:
        return self._slice(self.owned_node_bounds_global)

    @property
    def owned_node_slice_local(self) -> Slice2D:
        owned_x0, owned_x1, owned_y0, owned_y1 = self.owned_node_bounds_global
        solve_x0, _, solve_y0, _ = self.solve_bounds
        return (
            slice(owned_x0 - solve_x0, owned_x1 - solve_x0),
            slice(owned_y0 - solve_y0, owned_y1 - solve_y0),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "index": [self.index_x, self.index_y],
            "core_bounds": list(self.core_bounds),
            "solve_bounds": list(self.solve_bounds),
            "core_shape": list(self.core_shape),
            "solve_shape": list(self.solve_shape),
        }


@dataclass(frozen=True, slots=True)
class PartitionLayout:
    """Balanced rectangular partition layout for a global element grid."""

    global_shape: tuple[int, int]
    partition_shape: tuple[int, int]
    padding: int = 0

    def __post_init__(self) -> None:
        nx, ny = self.global_shape
        parts_x, parts_y = self.partition_shape
        if nx < 1 or ny < 1:
            raise ValueError("global_shape entries must be positive")
        if parts_x < 1 or parts_y < 1:
            raise ValueError("partition_shape entries must be positive")
        if parts_x > nx or parts_y > ny:
            raise ValueError("cannot create more partitions than elements along an axis")
        if self.padding < 0:
            raise ValueError("padding cannot be negative")

    @staticmethod
    def _boundaries(size: int, parts: int) -> tuple[int, ...]:
        base, remainder = divmod(size, parts)
        boundaries = [0]
        for index in range(parts):
            boundaries.append(boundaries[-1] + base + (index < remainder))
        return tuple(boundaries)

    @property
    def count(self) -> int:
        return self.partition_shape[0] * self.partition_shape[1]

    def __iter__(self) -> Iterator[Partition]:
        nx, ny = self.global_shape
        parts_x, parts_y = self.partition_shape
        bounds_x = self._boundaries(nx, parts_x)
        bounds_y = self._boundaries(ny, parts_y)
        for index_x in range(parts_x):
            for index_y in range(parts_y):
                core_x0, core_x1 = bounds_x[index_x], bounds_x[index_x + 1]
                core_y0, core_y1 = bounds_y[index_y], bounds_y[index_y + 1]
                yield Partition(
                    partition_id=index_x * parts_y + index_y,
                    index_x=index_x,
                    index_y=index_y,
                    core_bounds=(core_x0, core_x1, core_y0, core_y1),
                    solve_bounds=(
                        max(0, core_x0 - self.padding),
                        min(nx, core_x1 + self.padding),
                        max(0, core_y0 - self.padding),
                        min(ny, core_y1 + self.padding),
                    ),
                )

    def get(self, partition_id: int) -> Partition:
        if not 0 <= partition_id < self.count:
            raise KeyError(f"unknown partition_id {partition_id}")
        return next(partition for partition in self if partition.partition_id == partition_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_shape": list(self.global_shape),
            "partition_shape": list(self.partition_shape),
            "padding": self.padding,
            "partitions": [partition.as_dict() for partition in self],
        }

    def write_manifest(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
