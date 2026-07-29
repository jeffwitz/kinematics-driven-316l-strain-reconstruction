from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.measurement import (
    gaussian_gradient_band,
    profile_metrology,
    warp_forward_displacement,
)


def _coordinate_image(shape: tuple[int, int]) -> np.ndarray:
    rows, columns = np.indices(shape)
    return np.asarray(20 + columns + 2 * rows, dtype=np.uint8)


def test_iterative_warp_is_exact_for_rigid_translation() -> None:
    pytest.importorskip("cv2")
    image = _coordinate_image((48, 52))
    displacement = np.zeros((*image.shape, 2), dtype=np.float32)
    displacement[..., 0] = 1.0
    displacement[..., 1] = -2.0

    result = warp_forward_displacement(image, displacement)

    assert result.converged
    assert result.residual_pixels == 0.0
    np.testing.assert_array_equal(result.image[4:-4, 4:-4], image[6:-2, 3:-5])


def test_iterative_affine_inverse_differs_from_legacy_approximation() -> None:
    pytest.importorskip("cv2")
    image = _coordinate_image((64, 64))
    rows, columns = np.indices(image.shape, dtype=np.float32)
    displacement = np.zeros((*image.shape, 2), dtype=np.float32)
    displacement[..., 0] = 0.08 * columns
    displacement[..., 1] = -0.04 * rows

    corrected = warp_forward_displacement(image, displacement)
    legacy = warp_forward_displacement(
        image,
        displacement,
        mode="legacy_approximate_inverse",
    )

    assert corrected.iterations > 1
    assert corrected.residual_pixels <= 1.0e-5
    assert np.mean(np.abs(corrected.image.astype(float) - legacy.image.astype(float))) > 0.05


def test_noninvertible_forward_map_is_rejected() -> None:
    pytest.importorskip("cv2")
    image = _coordinate_image((32, 32))
    _, columns = np.indices(image.shape, dtype=np.float32)
    displacement = np.zeros((*image.shape, 2), dtype=np.float32)
    displacement[..., 0] = -1.1 * columns

    with pytest.raises(ValueError, match="non-invertible"):
        warp_forward_displacement(image, displacement)


def test_subpixel_fwhm_and_centroid_are_distinct_metrics() -> None:
    coordinate = np.arange(101, dtype=float)
    centre = 47.3
    sigma = 6.2
    profile = 0.2 + np.exp(-0.5 * np.square((coordinate - centre) / sigma))

    result = profile_metrology(profile)
    expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma

    assert result.fwhm_status == "ok"
    assert result.subpixel_fwhm_pixels == pytest.approx(expected, abs=0.03)
    assert result.centroid_index_pixels == pytest.approx(centre, abs=0.05)
    assert result.peak_index_pixels == 47.0
    assert result.legacy_integer_fwhm_pixels == 15.0


def test_missing_fwhm_crossing_has_explicit_status() -> None:
    result = profile_metrology(np.array([1.0, 2.0, 3.0, 4.0]))
    assert result.subpixel_fwhm_pixels is None
    assert result.fwhm_status == "missing_right_crossing"


@pytest.mark.parametrize("width", [4.0, 8.0, 16.0, 32.0])
@pytest.mark.parametrize("peak", [0.01, 0.025, 0.05])
def test_constant_peak_gradient_bands_respect_declared_constraints(
    width: float,
    peak: float,
) -> None:
    flow, gradient = gaussian_gradient_band(
        (128, 128),
        fwhm_pixels=width,
        orientation="horizontal",
        peak_gradient=peak,
    )

    assert np.max(gradient) == pytest.approx(peak, rel=0.01)
    assert np.max(flow) < 2.0
    assert profile_metrology(gradient).subpixel_fwhm_pixels == pytest.approx(width, abs=0.2)
