from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.measurement import DISFlowConfig, run_disflow, warp_image
from fem_inhouse.workflows.dic_measurement_chain import (
    image_flow_to_historical_evm,
    radial_autocorrelation,
)


def test_disflow_configuration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="preset"):
        DISFlowConfig(preset="unknown")
    with pytest.raises(ValueError, match="positive"):
        DISFlowConfig(patch_size=0)
    with pytest.raises(ValueError, match="nonnegative"):
        DISFlowConfig(variational_refinement_epsilon=-1.0)


def test_image_flow_evm_is_zero_for_rigid_translation() -> None:
    flow = np.zeros((17, 19, 2), dtype=np.float32)
    flow[..., 0] = 1.25
    flow[..., 1] = -0.5

    evm = image_flow_to_historical_evm(flow)

    assert evm.shape == (18, 16)
    np.testing.assert_allclose(evm, 0.0, rtol=0.0, atol=1e-14)


def test_radial_autocorrelation_constant_field_has_no_length() -> None:
    profile, length = radial_autocorrelation(np.ones((16, 12)))

    assert profile == [{"radius_pixels": 0.0, "correlation": 1.0}]
    assert length is None


def test_warp_and_disflow_recover_smooth_translation() -> None:
    pytest.importorskip("cv2")
    rows, columns = np.indices((96, 96))
    reference = (
        127.0
        + 50.0 * np.sin(0.31 * columns)
        + 40.0 * np.cos(0.27 * rows)
        + 20.0 * np.sin(0.13 * (rows + columns))
    )
    reference = np.clip(reference, 0, 255).astype(np.uint8)
    imposed = np.zeros((96, 96, 2), dtype=np.float32)
    imposed[..., 0] = 0.75
    imposed[..., 1] = -0.5
    warped = warp_image(reference, imposed)

    recovered = run_disflow(
        reference,
        warped,
        config=DISFlowConfig(
            finest_scale=0,
            gradient_descent_iterations=15,
            variational_refinement_iterations=5,
        ),
    )

    core = recovered[16:-16, 16:-16]
    np.testing.assert_allclose(np.mean(core, axis=(0, 1)), (0.75, -0.5), atol=0.2)
