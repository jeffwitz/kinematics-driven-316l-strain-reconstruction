"""One-dimensional profile metrology with explicit legacy metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ProfileMetrology:
    """Width and location measures for one positive profile."""

    legacy_integer_fwhm_pixels: float
    subpixel_fwhm_pixels: float | None
    fwhm_status: str
    peak_index_pixels: float
    centroid_index_pixels: float | None
    background: float
    peak_above_background: float


def _crossing(x0: float, y0: float, x1: float, y1: float, threshold: float) -> float:
    if y1 == y0:
        return 0.5 * (x0 + x1)
    return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)


def profile_metrology(values: NDArray[np.generic]) -> ProfileMetrology:
    """Measure FWHM, peak and positive-background centroid."""

    profile = np.asarray(values, dtype=np.float64)
    if profile.ndim != 1 or profile.size < 3 or not np.isfinite(profile).all():
        raise ValueError("profile must be a finite one-dimensional array of length >= 3")
    background = float(np.min(profile))
    positive = np.clip(profile - background, 0.0, None)
    peak_index = int(np.argmax(positive))
    peak = float(positive[peak_index])
    if peak <= 0.0:
        return ProfileMetrology(
            0.0,
            None,
            "nonpositive_peak",
            float(peak_index),
            None,
            background,
            peak,
        )

    centroid = float(np.sum(np.arange(profile.size) * positive) / np.sum(positive))
    threshold = 0.5 * peak
    selected = np.flatnonzero(positive >= threshold)
    legacy = float(selected[-1] - selected[0] + 1) if selected.size >= 2 else 0.0

    left_candidates = np.flatnonzero(positive[:peak_index] < threshold)
    right_candidates = np.flatnonzero(positive[peak_index + 1 :] < threshold)
    if not left_candidates.size:
        return ProfileMetrology(
            legacy, None, "missing_left_crossing", float(peak_index), centroid, background, peak
        )
    if not right_candidates.size:
        return ProfileMetrology(
            legacy, None, "missing_right_crossing", float(peak_index), centroid, background, peak
        )
    left0 = int(left_candidates[-1])
    left1 = left0 + 1
    right1 = peak_index + 1 + int(right_candidates[0])
    right0 = right1 - 1
    left = _crossing(
        float(left0), positive[left0], float(left1), positive[left1], threshold
    )
    right = _crossing(
        float(right0), positive[right0], float(right1), positive[right1], threshold
    )
    return ProfileMetrology(
        legacy_integer_fwhm_pixels=legacy,
        subpixel_fwhm_pixels=float(right - left),
        fwhm_status="ok",
        peak_index_pixels=float(peak_index),
        centroid_index_pixels=centroid,
        background=background,
        peak_above_background=peak,
    )
