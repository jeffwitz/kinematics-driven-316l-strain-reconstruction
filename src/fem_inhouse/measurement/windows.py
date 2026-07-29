"""Predeclared image windows for synthetic metrology."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class MeasurementWindow:
    """One result-independent rectangular image window."""

    identifier: str
    row_start: int
    row_stop: int
    column_start: int
    column_stop: int
    justification: str

    def __post_init__(self) -> None:
        if not self.identifier or not self.justification:
            raise ValueError("window identifier and justification must be non-empty")
        if self.row_start < 0 or self.column_start < 0:
            raise ValueError("window starts must be nonnegative")
        if self.row_stop <= self.row_start or self.column_stop <= self.column_start:
            raise ValueError("window stops must exceed starts")

    def extract(self, image: NDArray[np.generic]) -> NDArray[np.generic]:
        """Extract a contiguous window, rejecting out-of-bounds declarations."""

        values = np.asarray(image)
        if values.ndim != 2:
            raise ValueError("window source must be two-dimensional")
        if self.row_stop > values.shape[0] or self.column_stop > values.shape[1]:
            raise ValueError(f"window {self.identifier!r} exceeds image support")
        return np.ascontiguousarray(
            values[
                self.row_start : self.row_stop,
                self.column_start : self.column_stop,
            ]
        )

    def manifest(self, image: NDArray[np.generic]) -> dict[str, Any]:
        extracted = self.extract(image)
        return {
            "identifier": self.identifier,
            "bounds": [
                self.row_start,
                self.row_stop,
                self.column_start,
                self.column_stop,
            ],
            "shape": list(extracted.shape),
            "justification": self.justification,
            "sha256": hashlib.sha256(extracted.tobytes(order="C")).hexdigest(),
        }


def measurement_windows(rows: Iterable[dict[str, Any]]) -> tuple[MeasurementWindow, ...]:
    """Parse a YAML/JSON-compatible list of window declarations."""

    windows = tuple(
        MeasurementWindow(
            identifier=str(row["id"]),
            row_start=int(row["bounds"][0]),
            row_stop=int(row["bounds"][1]),
            column_start=int(row["bounds"][2]),
            column_stop=int(row["bounds"][3]),
            justification=str(row["justification"]),
        )
        for row in rows
    )
    identifiers = [window.identifier for window in windows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("measurement window identifiers must be unique")
    return windows
