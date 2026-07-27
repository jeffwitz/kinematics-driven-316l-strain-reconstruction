from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.postprocessing.spatial_correlation import (
    CorrelationProfile,
    fit_exponential_decay,
    mask_corrected_autocorrelation,
    structural_correlation,
)


def test_mask_corrected_autocorrelation_is_normalized_and_symmetric() -> None:
    rng = np.random.default_rng(42)
    field = rng.normal(size=(48, 40))
    mask = np.ones_like(field, dtype=bool)
    mask[3, 5] = False

    correlation, pairs = mask_corrected_autocorrelation(field, valid_mask=mask)

    assert correlation[24, 20] == pytest.approx(1.0, abs=1e-14)
    assert pairs[24, 20] == pytest.approx(float(mask.sum()))
    assert np.all(np.isfinite(correlation))


def test_exponential_fit_recovers_known_profile() -> None:
    distance = np.arange(100, dtype=np.float64)
    length = 18.0
    profile = CorrelationProfile(
        distance_pixels=distance,
        correlation=np.exp(-distance / length),
        pair_weight=np.linspace(10_000.0, 9_000.0, distance.size),
    )

    fit = fit_exponential_decay(profile)

    assert fit.length_pixels == pytest.approx(length, rel=1e-12)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-14)
    assert fit.point_count >= 5


def test_structural_correlation_detects_anisotropic_gaussian_field() -> None:
    rng = np.random.default_rng(7)
    white = rng.normal(size=(128, 128))
    frequencies_x = np.fft.fftfreq(128)[:, None]
    frequencies_y = np.fft.rfftfreq(128)[None, :]
    spectral_filter = np.exp(
        -0.5 * ((frequencies_x / 0.018) ** 2 + (frequencies_y / 0.035) ** 2)
    )
    field = np.fft.irfftn(
        np.fft.rfftn(white) * spectral_filter,
        s=white.shape,
        axes=(0, 1),
    )

    result = structural_correlation(field, maximum_lag_pixels=31)

    assert result.x_decay.length_pixels > result.y_decay.length_pixels
    assert result.radial_decay.length_pixels > 0.0
    assert result.rms_control_length_pixels > 0.0


@pytest.mark.parametrize(
    "field,match",
    [
        (np.ones(8), "two-dimensional"),
        (np.ones((8, 8)), "variance"),
    ],
)
def test_invalid_fields_are_rejected(field: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        structural_correlation(field)
