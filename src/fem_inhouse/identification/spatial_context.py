"""Spatial context interface for the TANN-FCC family (T0/T1/T2/T3).

The TANN is the only temporal transition: `Y_n -> Y_{n+1}`. A future
spatial operator (crystallographic convolution, intergrain transport) may
only provide *context* to the TANN, never produce slips, plastic strain or
latent states directly. T0 ships `ZeroSpatialContext`: the context is
exactly zero, and the TANN accepts it in its signature so the extension
point is real rather than documentation.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class SpatialContextProvider(Protocol):
    """Context per material point and slip system, fed to the TANN.

    The context may carry static EBSD structure (T1), intragranular
    convolution features (T2) or intergrain transport features (T3). It is
    read-only conditioning: the constitutive state still advances only
    through the TANN integrator.
    """

    def forward(
        self,
        committed_or_predicted_state: FloatArray,
        crystal_geometry: FloatArray,
        grain_ids: FloatArray,
    ) -> FloatArray:
        """Return `(points, 12, context_dim)` conditioning."""
        ...


class ZeroSpatialContext:
    """T0 context: exactly zero, of the configured width."""

    def __init__(self, context_dim: int) -> None:
        self.context_dim = context_dim

    def forward(
        self,
        committed_or_predicted_state: FloatArray,
        crystal_geometry: FloatArray,
        grain_ids: FloatArray,
    ) -> FloatArray:
        points = int(np.asarray(committed_or_predicted_state).shape[0])
        return np.zeros((points, 12, self.context_dim), dtype=np.float64)
