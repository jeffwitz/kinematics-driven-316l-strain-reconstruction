from __future__ import annotations

import numpy as np

from fem_inhouse.spectral2d.grid import StructuredGrid2D
from scripts.qualify_srix_svd_shadow import (
    _material_stress_to_spectral_samples,
    _spectral_samples_to_material,
)


def test_spectral_shadow_material_layout_is_c_order_and_round_trips_sentinel() -> None:
    """EBSD F ordering must not permute the spectral TRI2 material batch."""

    grid = StructuredGrid2D(3, 5, 3.0, 5.0)
    samples = np.empty((3, 5, 2, 3), dtype=float)
    for x, y, q in np.ndindex(3, 5, 2):
        samples[x, y, q] = (1000 * x + 100 * y + 10 * q) + np.arange(3)

    flattened = _spectral_samples_to_material(samples)
    restored = _material_stress_to_spectral_samples(flattened, grid)

    np.testing.assert_array_equal(restored, samples)
    np.testing.assert_array_equal(flattened[2 * (5 * 1 + 3) + 1], samples[1, 3, 1])
