from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.measurement import (
    apply_image_mask,
    binary_mask,
    canonical_to_image_flow,
    declared_all_valid_mask,
    historical_uv_to_canonical,
    image_flow_to_canonical,
)


def test_image_and_canonical_conversions_are_exact_inverses() -> None:
    flow = np.arange(5 * 7 * 2, dtype=np.float32).reshape(5, 7, 2) / 10.0

    canonical = image_flow_to_canonical(flow, pixel_size_mm=0.00184)
    recovered = canonical_to_image_flow(canonical, pixel_size_mm=0.00184)

    assert canonical.shape == (7, 5, 2)
    np.testing.assert_allclose(recovered, flow, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(canonical[..., 0], flow[..., 0].T * 0.00184)
    np.testing.assert_allclose(canonical[..., 1], flow[..., 1].T * 0.00184)


def test_historical_experiment_maps_v_to_ux_and_u_to_uy() -> None:
    historical_u = np.full((3, 4), 20.0)
    historical_v = np.full((3, 4), -2.0)

    canonical = historical_uv_to_canonical(
        historical_u,
        historical_v,
        pixel_size_mm=0.002,
    )

    np.testing.assert_array_equal(canonical[..., 0], -0.004)
    np.testing.assert_array_equal(canonical[..., 1], 0.04)


@pytest.mark.parametrize(
    "mask",
    [
        np.array([[False, True], [True, False]]),
        np.array([[0, 1], [1, 0]], dtype=np.uint8),
        np.array([[0, 255], [255, 0]], dtype=np.uint8),
    ],
)
def test_binary_mask_accepts_documented_encodings(mask: np.ndarray) -> None:
    expected = np.array([[False, True], [True, False]])
    np.testing.assert_array_equal(binary_mask(mask, shape=(2, 2)), expected)


def test_binary_mask_rejects_ambiguous_values_and_dtype() -> None:
    with pytest.raises(ValueError, match="0/1 or 0/255"):
        binary_mask(np.array([[0, 2]], dtype=np.uint8), shape=(1, 2))
    with pytest.raises(TypeError, match="boolean or integer"):
        binary_mask(np.array([[0.0, 1.0]]), shape=(1, 2))


def test_legacy_uint8_multiply_is_distinct_from_binary_mask() -> None:
    image = np.array([[2, 3]], dtype=np.uint8)
    mask = np.array([[255, 255]], dtype=np.uint8)

    binary = apply_image_mask(image, mask, mode="binary_mask")
    historical = apply_image_mask(image, mask, mode="legacy_uint8_multiply")

    np.testing.assert_array_equal(binary, image)
    np.testing.assert_array_equal(historical, np.array([[254, 253]], dtype=np.uint8))


def test_declared_all_valid_mask_is_boolean_and_deterministic() -> None:
    mask = declared_all_valid_mask((4, 3))
    assert mask.dtype == np.bool_
    assert mask.shape == (4, 3)
    assert np.all(mask)
