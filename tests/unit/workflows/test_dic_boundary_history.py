from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.workflows.dic_boundary_history import (
    affine_boundary_decomposition,
    element_gauss_engineering_strain,
)


def test_affine_boundary_decomposition_removes_rigid_and_affine_motion() -> None:
    spacing = 0.2
    x = np.arange(7, dtype=float)[:, None] * spacing
    y = np.arange(6, dtype=float)[None, :] * spacing
    displacement = np.empty((7, 6, 2))
    displacement[..., 0] = 0.4 + 0.03 * x - 0.02 * y
    displacement[..., 1] = -0.2 + 0.01 * x + 0.04 * y

    result = affine_boundary_decomposition(
        displacement,
        spacing_x_mm=spacing,
        spacing_y_mm=spacing,
    )

    assert result.residual_rms_mm == pytest.approx(0.0, abs=5.0e-16)
    assert result.residual_maximum_mm == pytest.approx(0.0, abs=1.0e-15)
    np.testing.assert_allclose(
        result.coefficients,
        np.array([[0.4, -0.2], [0.03, 0.01], [-0.02, 0.04]]),
        rtol=0.0,
        atol=2.0e-15,
    )


def test_affine_boundary_decomposition_detects_boundary_spike() -> None:
    displacement = np.zeros((8, 7, 2))
    displacement[3, -1, 1] = 0.1

    result = affine_boundary_decomposition(
        displacement,
        spacing_x_mm=1.0,
        spacing_y_mm=1.0,
    )

    assert result.residual_rms_mm > 0.0
    assert result.residual_maximum_mm > 0.08


def test_element_gauss_engineering_strain_recovers_affine_field() -> None:
    spacing = 0.25
    x = np.arange(5, dtype=float)[:, None] * spacing
    y = np.arange(4, dtype=float)[None, :] * spacing
    displacement = np.empty((5, 4, 2))
    displacement[..., 0] = 0.01 * x + 0.03 * y
    displacement[..., 1] = -0.02 * x + 0.04 * y

    strain = element_gauss_engineering_strain(
        displacement,
        element_indices=(0, 5, 11),
        spacing_mm=spacing,
    )

    expected = np.array([0.01, 0.04, 0.01])
    np.testing.assert_allclose(
        strain,
        np.broadcast_to(expected, strain.shape),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_boundary_diagnostics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="finite shape"):
        affine_boundary_decomposition(
            np.zeros((4, 5)),
            spacing_x_mm=1.0,
            spacing_y_mm=1.0,
        )
    with pytest.raises(ValueError, match="outside"):
        element_gauss_engineering_strain(
            np.zeros((3, 3, 2)),
            element_indices=(4,),
            spacing_mm=1.0,
        )
