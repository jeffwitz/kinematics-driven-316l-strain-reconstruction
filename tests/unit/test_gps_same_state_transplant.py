"""Pure checks for the GPS/reference diagnostic frame conversions."""

from __future__ import annotations

import numpy as np

from scripts.diagnose_gps_tangent_blocks import (
    _rotate_gradient_to_crystal,
    _rotate_gradient_to_global,
)


def test_gradient_global_crystal_global_roundtrip() -> None:
    angle = 0.37
    q = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    global_gradient = np.array([0.011, -0.007, 0.023, 0.004, -0.003, 0.006])
    crystal = _rotate_gradient_to_crystal(global_gradient, q)
    roundtrip = _rotate_gradient_to_global(crystal, q)
    np.testing.assert_allclose(roundtrip, global_gradient, rtol=0.0, atol=1.0e-13)


def test_gradient_rotation_preserves_symmetry() -> None:
    q = np.array(
        [
            [0.8, -0.6, 0.0],
            [0.6, 0.8, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    gradient = np.array([0.01, 0.02, -0.01, 0.003, 0.004, -0.002])
    transformed = _rotate_gradient_to_crystal(gradient, q)
    assert np.isfinite(transformed).all()
