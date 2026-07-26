from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.identification.observation import (
    DICObservationOperator,
    DICObservationOperatorConfig,
)


def _affine_displacement(nx: int, ny: int, spacing: float) -> np.ndarray:
    x = np.arange(nx + 1, dtype=float)[:, None] * spacing
    y = np.arange(ny + 1, dtype=float)[None, :] * spacing
    displacement = np.empty((nx + 1, ny + 1, 2), dtype=float)
    displacement[..., 0] = 0.01 * x + 0.003 * y
    displacement[..., 1] = -0.002 * x + 0.004 * y
    return displacement


def test_identity_and_coincident_stride_observe_same_affine_evm() -> None:
    displacement = _affine_displacement(8, 6, 0.002)
    identity = DICObservationOperator(
        DICObservationOperatorConfig(use_core_only=False),
        poisson_ratio=0.3,
    )
    reduced = DICObservationOperator(
        DICObservationOperatorConfig(
            grid_mapping="coincident-node-stride",
            grid_reduction=2,
            use_core_only=False,
        ),
        poisson_ratio=0.3,
    )

    fine = identity.observe_displacement(
        displacement,
        spacing_x_mm=0.002,
        spacing_y_mm=0.002,
    )
    coarse = reduced.observe_displacement(
        displacement,
        spacing_x_mm=0.002,
        spacing_y_mm=0.002,
    )

    assert coarse.element_field.shape == (4, 3)
    np.testing.assert_allclose(coarse.element_field, fine.element_field[::2, ::2])
    assert coarse.spacing_x_mm == 0.004
    assert coarse.spacing_y_mm == 0.004


def test_observation_operator_reduces_core_and_mask_deterministically() -> None:
    displacement = _affine_displacement(8, 8, 0.001)
    mask = np.ones((8, 8), dtype=bool)
    mask[2:4, 2:4] = False
    operator = DICObservationOperator(
        DICObservationOperatorConfig(
            grid_mapping="coincident-node-stride",
            grid_reduction=2,
        ),
        poisson_ratio=0.3,
    )

    result = operator.observe_displacement(
        displacement,
        spacing_x_mm=0.001,
        spacing_y_mm=0.001,
        core_slice=(slice(2, 6), slice(2, 6)),
        mask=mask,
    )

    assert result.element_field.shape == (2, 2)
    assert result.valid_mask.shape == (2, 2)
    assert not result.valid_mask[0, 0]
    assert len(result.operator_sha256) == 64


def test_observation_operator_rejects_non_divisible_grid() -> None:
    operator = DICObservationOperator(
        DICObservationOperatorConfig(
            grid_mapping="coincident-node-stride",
            grid_reduction=2,
            use_core_only=False,
        ),
        poisson_ratio=0.3,
    )
    with pytest.raises(ValueError, match="divisible"):
        operator.observe_displacement(
            np.zeros((8, 8, 2)),
            spacing_x_mm=0.001,
            spacing_y_mm=0.001,
        )
