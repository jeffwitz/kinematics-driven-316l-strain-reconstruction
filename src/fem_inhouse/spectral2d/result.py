"""Result container for the spectral plane-stress solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.spectral2d.diagnostics import Spectral2DDiagnostics

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Spectral2DResult:
    displacement: FloatArray
    applied_displacement: FloatArray
    fluctuation_displacement: FloatArray
    strain_in_plane: FloatArray
    stress_in_plane_mpa: FloatArray
    full_stress_tensor_mpa: FloatArray | None
    full_strain_tensor: FloatArray | None
    elastic_strain_tensor: FloatArray | None
    plastic_strain_tensor: FloatArray | None
    observables: dict[str, FloatArray]
    reaction_forces: FloatArray
    diagnostics: Spectral2DDiagnostics

    def pixel_average(self, field: FloatArray) -> FloatArray:
        """Average only the two TRI2 subcells of a supplied raw field."""

        values = np.asarray(field, dtype=np.float64)
        if values.ndim < 3 or values.shape[2] != 2:
            raise ValueError("pixel_average expects a TRI2 field with axis 2 of length 2")
        return values.mean(axis=2)
