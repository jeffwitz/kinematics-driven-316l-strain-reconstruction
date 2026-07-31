"""Normal-section profiles across a band, and the metrics read from them.

Position, width and amplitude are measured separately and never combined into a
single number here: a band can be right in position and wrong in width, and a
scalar score would hide exactly that.

Every width definition is reported alongside the others. None is declared
superior — they disagree on multimodal and heavy-tailed profiles, and that
disagreement is information.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

FloatArray = NDArray[np.float64]


class WidthStatus(StrEnum):
    """Why a width measurement is or is not usable."""

    OK = "ok"
    NO_CROSSING = "no_crossing"
    TOO_WEAK = "too_weak"
    MULTIMODAL = "multimodal"
    PEAK_AT_EDGE = "peak_at_edge"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class NormalProfile:
    """One sampled section across a band."""

    section_id: int
    offsets_pixels: FloatArray
    values: FloatArray
    valid: bool
    exclusion_reason: str


@dataclass(frozen=True, slots=True)
class BackgroundEstimate:
    """Local background taken from the tails of a normal profile."""

    level: float
    spread: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class WidthMeasurement:
    """The three declared width definitions and their status."""

    fwhm_pixels: float
    integral_pixels: float
    second_moment_pixels: float
    status: WidthStatus
    peak_count: int


def sample_normal_profile(
    field: NDArray[np.generic],
    *,
    origin: tuple[float, float],
    normal: tuple[float, float],
    half_length_pixels: float,
    spacing_pixels: float = 1.0,
    section_id: int = 0,
    valid_mask: NDArray[np.bool_] | None = None,
    border_margin_pixels: float = 2.0,
) -> NormalProfile:
    """Sample a field along a normal, with an explicit reason when unusable.

    Bilinear interpolation; a section leaving the support or crossing an
    invalid pixel is returned marked invalid rather than silently clipped.
    """

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("field must be two-dimensional")
    if not np.isfinite(half_length_pixels) or half_length_pixels <= 0.0:
        raise ValueError("half_length_pixels must be finite and positive")
    if not np.isfinite(spacing_pixels) or spacing_pixels <= 0.0:
        raise ValueError("spacing_pixels must be finite and positive")

    offsets = np.arange(-half_length_pixels, half_length_pixels + 0.5 * spacing_pixels,
                        spacing_pixels)
    rows = origin[0] + offsets * normal[0]
    columns = origin[1] + offsets * normal[1]

    margin = border_margin_pixels
    inside = (
        (rows >= margin)
        & (rows <= values.shape[0] - 1 - margin)
        & (columns >= margin)
        & (columns <= values.shape[1] - 1 - margin)
    )
    empty = np.zeros(0, dtype=np.float64)
    if not inside.all():
        return NormalProfile(section_id, offsets, empty, False, "leaves_support")

    sampled = ndimage.map_coordinates(
        values, np.vstack((rows, columns)), order=1, mode="nearest"
    )
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must match the field shape")
        covered = ndimage.map_coordinates(
            mask.astype(np.float64), np.vstack((rows, columns)), order=1, mode="nearest"
        )
        if float(np.min(covered)) < 1.0:
            return NormalProfile(section_id, offsets, empty, False, "crosses_invalid")
    if not np.isfinite(sampled).all():
        return NormalProfile(section_id, offsets, empty, False, "nonfinite_sample")
    return NormalProfile(
        section_id, offsets, np.ascontiguousarray(sampled), True, ""
    )


def estimate_background(
    profile: NormalProfile,
    *,
    corridor_half_width_pixels: float,
) -> BackgroundEstimate:
    """Estimate the local background from the profile tails.

    The background is not assumed to be zero: a band sits on a spatially varying
    field, and subtracting nothing would inflate every width and mass.
    """

    if not profile.valid or profile.values.size == 0:
        raise ValueError("cannot estimate a background from an invalid profile")
    tails = np.abs(profile.offsets_pixels) > corridor_half_width_pixels
    if np.count_nonzero(tails) < 2:
        raise ValueError("the profile is too short to expose a background")
    samples = profile.values[tails]
    return BackgroundEstimate(
        level=float(np.median(samples)),
        spread=float(np.subtract(*np.percentile(samples, [75, 25]))),
        sample_count=int(samples.size),
    )


def excess_profile(profile: NormalProfile, background: BackgroundEstimate) -> FloatArray:
    """Return the non-negative excess of the profile over its background."""

    return np.clip(profile.values - background.level, 0.0, None)


def _count_peaks(excess: FloatArray, *, relative_prominence: float = 0.25) -> int:
    peak = float(np.max(excess)) if excess.size else 0.0
    if peak <= 0.0:
        return 0
    floor = relative_prominence * peak
    above = excess >= floor
    transitions = int(np.count_nonzero(above[1:] & ~above[:-1]))
    return transitions + int(bool(above[0]))


def measure_width(
    profile: NormalProfile,
    background: BackgroundEstimate,
    *,
    minimum_peak: float = 0.0,
) -> WidthMeasurement:
    """Return the three width definitions, with an explicit status.

    The status is what makes a missing width usable downstream: a section that
    has no half-maximum crossing is different from one whose band is simply
    absent, and averaging over both would be wrong.
    """

    nan = float("nan")
    if not profile.valid or profile.values.size == 0:
        return WidthMeasurement(nan, nan, nan, WidthStatus.EMPTY, 0)

    excess = excess_profile(profile, background)
    offsets = profile.offsets_pixels
    peak = float(np.max(excess))
    peaks = _count_peaks(excess)
    if peak <= minimum_peak or peak <= 0.0:
        return WidthMeasurement(nan, nan, nan, WidthStatus.TOO_WEAK, peaks)

    index = int(np.argmax(excess))
    if index == 0 or index == len(excess) - 1:
        return WidthMeasurement(nan, nan, nan, WidthStatus.PEAK_AT_EDGE, peaks)

    mass = float(np.trapezoid(excess, offsets))
    integral = mass / peak if peak > 0.0 else nan
    if mass > 0.0:
        centre = float(np.trapezoid(excess * offsets, offsets) / mass)
        variance = float(np.trapezoid(excess * (offsets - centre) ** 2, offsets) / mass)
        second_moment = 2.0 * float(np.sqrt(max(variance, 0.0)))
    else:
        second_moment = nan

    half = 0.5 * peak
    left = np.where(excess[:index] <= half)[0]
    right = np.where(excess[index:] <= half)[0]
    if left.size == 0 or right.size == 0:
        return WidthMeasurement(nan, integral, second_moment, WidthStatus.NO_CROSSING, peaks)

    def _cross(i0: int, i1: int) -> float:
        y0, y1 = excess[i0], excess[i1]
        if y1 == y0:
            return float(offsets[i0])
        weight = (half - y0) / (y1 - y0)
        return float(offsets[i0] + weight * (offsets[i1] - offsets[i0]))

    left_index = int(left[-1])
    right_index = int(index + right[0])
    fwhm = _cross(right_index, right_index - 1) - _cross(left_index, left_index + 1)
    status = WidthStatus.MULTIMODAL if peaks > 1 else WidthStatus.OK
    return WidthMeasurement(float(fwhm), integral, second_moment, status, peaks)


def measure_position(
    profile: NormalProfile,
    background: BackgroundEstimate,
) -> dict[str, float]:
    """Return where the band sits on its own section, in pixels from centre."""

    nan = float("nan")
    if not profile.valid or profile.values.size == 0:
        return {"peak_offset": nan, "centroid_offset": nan, "detected": 0.0}
    excess = excess_profile(profile, background)
    peak = float(np.max(excess))
    if peak <= 0.0:
        return {"peak_offset": nan, "centroid_offset": nan, "detected": 0.0}
    offsets = profile.offsets_pixels
    mass = float(np.trapezoid(excess, offsets))
    centroid = float(np.trapezoid(excess * offsets, offsets) / mass) if mass > 0 else nan
    return {
        "peak_offset": float(offsets[int(np.argmax(excess))]),
        "centroid_offset": centroid,
        "detected": 1.0,
    }


def measure_amplitude(
    profile: NormalProfile,
    background: BackgroundEstimate,
    *,
    corridor_half_width_pixels: float,
) -> dict[str, float]:
    """Return peak, upper quantile, integrated mass and corridor mean."""

    nan = float("nan")
    if not profile.valid or profile.values.size == 0:
        return {"peak": nan, "q95": nan, "mass": nan, "corridor_mean": nan}
    excess = excess_profile(profile, background)
    inside = np.abs(profile.offsets_pixels) <= corridor_half_width_pixels
    return {
        "peak": float(np.max(excess)),
        "q95": float(np.quantile(excess, 0.95)),
        "mass": float(np.trapezoid(excess, profile.offsets_pixels)),
        "corridor_mean": float(np.mean(excess[inside])) if inside.any() else nan,
    }


def compare_profiles(
    reference: NormalProfile,
    candidate: NormalProfile,
    reference_background: BackgroundEstimate,
    candidate_background: BackgroundEstimate,
) -> dict[str, float]:
    """Compare the shapes of two sections sampled on the same geometry."""

    nan = float("nan")
    if not (reference.valid and candidate.valid):
        return {"correlation": nan, "l1": nan, "l2": nan, "asymmetry_difference": nan}
    a = excess_profile(reference, reference_background)
    b = excess_profile(candidate, candidate_background)
    if a.shape != b.shape:
        raise ValueError("profiles must be sampled on the same offsets")
    scale = float(np.max(a))
    if scale <= 0.0:
        return {"correlation": nan, "l1": nan, "l2": nan, "asymmetry_difference": nan}
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        correlation = nan
    else:
        correlation = float(np.corrcoef(a, b)[0, 1])

    def _asymmetry(values: FloatArray, offsets: FloatArray) -> float:
        mass = float(np.trapezoid(values, offsets))
        if mass <= 0.0:
            return nan
        centre = float(np.trapezoid(values * offsets, offsets) / mass)
        third = float(np.trapezoid(values * (offsets - centre) ** 3, offsets) / mass)
        second = float(np.trapezoid(values * (offsets - centre) ** 2, offsets) / mass)
        return third / second**1.5 if second > 0.0 else nan

    offsets = reference.offsets_pixels
    return {
        "correlation": correlation,
        "l1": float(np.mean(np.abs(a - b)) / scale),
        "l2": float(np.sqrt(np.mean((a - b) ** 2)) / scale),
        "asymmetry_difference": _asymmetry(b, offsets) - _asymmetry(a, offsets),
    }


def continuity_metrics(
    detected: NDArray[np.generic],
    *,
    spacing_pixels: float,
) -> dict[str, float]:
    """Summarise how continuously a band was detected along its centreline.

    A band that disappears over part of its length is a different failure from
    one that is uniformly too weak, and the longest gap is what separates them.
    """

    flags = np.asarray(detected, dtype=bool)
    if flags.size == 0:
        raise ValueError("detected must be non-empty")
    if not np.isfinite(spacing_pixels) or spacing_pixels <= 0.0:
        raise ValueError("spacing_pixels must be finite and positive")
    gaps: list[int] = []
    run = 0
    for flag in flags:
        if flag:
            if run:
                gaps.append(run)
            run = 0
        else:
            run += 1
    if run:
        gaps.append(run)
    return {
        "detected_fraction": float(np.count_nonzero(flags) / flags.size),
        "detected_length_pixels": float(np.count_nonzero(flags) * spacing_pixels),
        "gap_count": float(len(gaps)),
        "longest_gap_pixels": float(max(gaps) * spacing_pixels) if gaps else 0.0,
    }


def summarise(values: NDArray[np.generic]) -> dict[str, float]:
    """Return the distributional summary the specification requires.

    Never a mean alone: the worst decile is what catches a band that is well
    reproduced along most of its length and lost at one end.
    """

    samples = np.asarray(values, dtype=np.float64).ravel()
    finite = samples[np.isfinite(samples)]
    nan = float("nan")
    if finite.size == 0:
        return {
            "median": nan, "mean": nan, "iqr": nan, "p90": nan,
            "worst_decile": nan, "valid_fraction": 0.0,
            "missing_fraction": 1.0 if samples.size else nan,
        }
    q25, q75 = np.percentile(finite, [25, 75])
    return {
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "iqr": float(q75 - q25),
        "p90": float(np.percentile(finite, 90)),
        "worst_decile": float(np.mean(finite[finite >= np.percentile(finite, 90)])),
        "valid_fraction": float(finite.size / samples.size),
        "missing_fraction": float(1.0 - finite.size / samples.size),
    }
