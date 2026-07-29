from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.measurement import direct_photometric_residual


@pytest.mark.measurement
def test_direct_photometric_residual_is_zero_for_identical_images() -> None:
    pytest.importorskip("cv2")
    image = np.arange(7 * 9, dtype=np.uint8).reshape(7, 9)
    flow = np.zeros((*image.shape, 2), dtype=np.float64)

    result = direct_photometric_residual(image, image.copy(), flow)

    np.testing.assert_array_equal(result.absolute_residual_grey_levels, 0.0)
    assert np.all(result.valid_mask)


@pytest.mark.measurement
def test_direct_photometric_residual_uses_forward_flow_ordering() -> None:
    pytest.importorskip("cv2")
    reference = np.arange(8 * 10, dtype=np.uint8).reshape(8, 10)
    current = np.zeros_like(reference)
    current[:, 1:] = reference[:, :-1]
    flow = np.zeros((*reference.shape, 2), dtype=np.float64)
    flow[..., 0] = 1.0

    result = direct_photometric_residual(reference, current, flow)

    np.testing.assert_array_equal(
        result.absolute_residual_grey_levels[:, :-1],
        0.0,
    )
    assert np.all(result.valid_mask[:, :-1])
    assert not np.any(result.valid_mask[:, -1])


@pytest.mark.measurement
def test_direct_photometric_residual_rejects_invalid_inputs() -> None:
    pytest.importorskip("cv2")
    image = np.zeros((4, 5), dtype=np.uint8)
    with pytest.raises(TypeError, match="uint8"):
        direct_photometric_residual(image.astype(float), image, np.zeros((4, 5, 2)))
    with pytest.raises(ValueError, match="shape"):
        direct_photometric_residual(image, image, np.zeros((4, 5)))
    flow = np.zeros((4, 5, 2))
    flow[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        direct_photometric_residual(image, image, flow)
