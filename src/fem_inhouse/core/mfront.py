"""Compatibility facade for the split MFront/MGIS bridges.

The implementation lives in the focused modules below.  This module keeps
the historical import surface stable for applications, tests, and diagnostics.
"""

from __future__ import annotations

import numpy as np

from fem_inhouse.core import mfront_runtime as _mfront_runtime
from fem_inhouse.core.mfront_3d import (
    MFront3DMaterialPointBatch,
)
from fem_inhouse.core.mfront_condensation import (
    MFront3DCondensedPlaneStressBatch,
    MFront3DCondensedPlaneStressBlockBatch,
    condense_kelvin_tangent_blocks,
    condense_kelvin_tangent_to_engineering,
)
from fem_inhouse.core.mfront_gps import MFrontNativeGeneralisedPlaneStressBatch
from fem_inhouse.core.mfront_native import (
    MFrontMaterialPointBatch,
    MFrontNativePlaneStressBatch,
)
from fem_inhouse.core.mfront_runtime import (
    MFrontIntegrationError,
    MFrontIntegrationResult,
    MFrontUnavailableError,
    engineering_strain_to_kelvin,
    kelvin_strain_to_engineering,
    kelvin_stress_to_engineering,
    kelvin_tangent_to_engineering,
)
from fem_inhouse.core.mfront_state import (
    MFrontCondensedBlocksStateSnapshot,
    MFrontCondensedStateSnapshot,
    MFrontMaterialStateSnapshot,
    MFrontTimingStatistics,
)

_SQRT_TWO = np.sqrt(2.0)
_ENGINEERING_TO_KELVIN_STRAIN_SCALE = _mfront_runtime._ENGINEERING_TO_KELVIN_STRAIN_SCALE
_KELVIN_TO_ENGINEERING_STRESS_SCALE = _mfront_runtime._KELVIN_TO_ENGINEERING_STRESS_SCALE
_PLANE_STRESS_COMPONENTS = np.array([0, 1, 3])
_TRANSVERSE_COMPONENTS_3D = np.array([2, 4, 5])
_declared_internal_slices = _mfront_runtime._declared_internal_slices
_SYMMETRIC_POSITIVE_DEFINITE_J2_BEHAVIOURS = frozenset(
    {
        "PixelLudwikJ2Plasticity",
        "PixelMicromorphicLudwikJ2Plasticity",
        "PixelLudwikJ2Plasticity3D",
        "PixelMicromorphicLudwikJ2Plasticity3D",
    }
)

__all__ = [
    "MFront3DCondensedPlaneStressBatch",
    "MFront3DCondensedPlaneStressBlockBatch",
    "MFront3DMaterialPointBatch",
    "MFrontCondensedBlocksStateSnapshot",
    "MFrontCondensedStateSnapshot",
    "MFrontIntegrationError",
    "MFrontIntegrationResult",
    "MFrontMaterialPointBatch",
    "MFrontMaterialStateSnapshot",
    "MFrontNativeGeneralisedPlaneStressBatch",
    "MFrontNativePlaneStressBatch",
    "MFrontTimingStatistics",
    "MFrontUnavailableError",
    "condense_kelvin_tangent_blocks",
    "condense_kelvin_tangent_to_engineering",
    "engineering_strain_to_kelvin",
    "kelvin_strain_to_engineering",
    "kelvin_stress_to_engineering",
    "kelvin_tangent_to_engineering",
]
