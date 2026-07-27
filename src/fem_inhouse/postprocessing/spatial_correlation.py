"""Mask-aware spatial-correlation estimators for structural scalar fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class CorrelationProfile:
    """One-dimensional normalized autocorrelation profile."""

    distance_pixels: FloatArray
    correlation: FloatArray
    pair_weight: FloatArray


@dataclass(frozen=True, slots=True)
class DecayFit:
    """Weighted exponential fit on a preregistered correlation interval."""

    length_pixels: float
    slope_per_pixel: float
    intercept: float
    r_squared: float
    first_index: int
    last_index: int
    point_count: int


@dataclass(frozen=True, slots=True)
class StructuralCorrelationResult:
    """Radial and directional structural-correlation summary."""

    radial: CorrelationProfile
    x_direction: CorrelationProfile
    y_direction: CorrelationProfile
    radial_decay: DecayFit
    x_decay: DecayFit
    y_decay: DecayFit
    rms_radius_pixels: float
    rms_control_length_pixels: float
    valid_fraction: float


def _validated_field_and_mask(
    field: np.ndarray,
    valid_mask: np.ndarray | None,
) -> tuple[FloatArray, BoolArray]:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"field must be two-dimensional, got shape {values.shape}")
    mask = (
        np.isfinite(values)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=np.bool_) & np.isfinite(values)
    )
    if mask.shape != values.shape:
        raise ValueError(f"mask shape {mask.shape} does not match field shape {values.shape}")
    if np.count_nonzero(mask) < 2:
        raise ValueError("at least two valid field values are required")
    return values, mask


def mask_corrected_autocorrelation(
    field: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return centered circular covariance and valid-pair count, normalized at zero lag."""

    values, mask = _validated_field_and_mask(field, valid_mask)
    mean = float(np.mean(values[mask], dtype=np.float64))
    centered = np.where(mask, values - mean, 0.0)
    spectrum = np.fft.rfftn(centered)
    mask_spectrum = np.fft.rfftn(mask.astype(np.float64))
    covariance = np.fft.irfftn(
        spectrum * spectrum.conj(),
        s=values.shape,
        axes=(0, 1),
    ).real
    pairs = np.fft.irfftn(
        mask_spectrum * mask_spectrum.conj(),
        s=values.shape,
        axes=(0, 1),
    ).real
    covariance = np.fft.fftshift(covariance)
    pairs = np.fft.fftshift(pairs)
    centre = tuple(size // 2 for size in values.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = np.divide(
            covariance,
            pairs,
            out=np.full_like(covariance, np.nan),
            where=pairs > 0.5,
        )
    variance = float(corrected[centre])
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("valid field variance must be strictly positive")
    return corrected / variance, np.maximum(pairs, 0.0)


def correlation_profiles(
    correlation: np.ndarray,
    pair_weight: np.ndarray,
    *,
    maximum_lag_pixels: int | None = None,
) -> tuple[CorrelationProfile, CorrelationProfile, CorrelationProfile]:
    """Build pair-weighted radial and positive-axis profiles."""

    corr = np.asarray(correlation, dtype=np.float64)
    weights = np.asarray(pair_weight, dtype=np.float64)
    if corr.ndim != 2 or corr.shape != weights.shape:
        raise ValueError("correlation and pair_weight must be matching 2-D arrays")
    cx, cy = (size // 2 for size in corr.shape)
    maximum = min(corr.shape) // 4 if maximum_lag_pixels is None else maximum_lag_pixels
    if maximum < 1 or maximum >= min(cx + 1, cy + 1):
        raise ValueError("maximum_lag_pixels is outside the supported centered domain")

    window_corr = corr[cx - maximum : cx + maximum + 1, cy - maximum : cy + maximum + 1]
    window_weight = weights[
        cx - maximum : cx + maximum + 1,
        cy - maximum : cy + maximum + 1,
    ]
    dx, dy = np.ogrid[-maximum : maximum + 1, -maximum : maximum + 1]
    bins = np.rint(np.sqrt(dx * dx + dy * dy)).astype(np.int32)
    keep = (bins <= maximum) & np.isfinite(window_corr) & (window_weight > 0.5)
    denominator = np.bincount(
        bins[keep],
        weights=window_weight[keep],
        minlength=maximum + 1,
    )
    numerator = np.bincount(
        bins[keep],
        weights=window_corr[keep] * window_weight[keep],
        minlength=maximum + 1,
    )
    radial_corr = np.divide(
        numerator,
        denominator,
        out=np.full(maximum + 1, np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )
    distance = np.arange(maximum + 1, dtype=np.float64)
    radial = CorrelationProfile(distance, radial_corr, denominator.astype(np.float64))
    x_profile = CorrelationProfile(
        distance.copy(),
        corr[cx : cx + maximum + 1, cy].copy(),
        weights[cx : cx + maximum + 1, cy].copy(),
    )
    y_profile = CorrelationProfile(
        distance.copy(),
        corr[cx, cy : cy + maximum + 1].copy(),
        weights[cx, cy : cy + maximum + 1].copy(),
    )
    return radial, x_profile, y_profile


def fit_exponential_decay(
    profile: CorrelationProfile,
    *,
    lower_correlation: float = 0.15,
    upper_correlation: float = 0.60,
    minimum_points: int = 5,
) -> DecayFit:
    """Fit the first contiguous correlation branch inside the frozen interval."""

    corr = profile.correlation
    eligible = np.flatnonzero(
        np.isfinite(corr)
        & (corr >= lower_correlation)
        & (corr <= upper_correlation)
        & (profile.pair_weight > 0.0)
    )
    if eligible.size == 0:
        raise ValueError("correlation never enters the preregistered fit interval")
    first = int(eligible[0])
    last = first
    while (
        last + 1 < corr.size
        and np.isfinite(corr[last + 1])
        and lower_correlation <= corr[last + 1] <= upper_correlation
        and corr[last + 1] <= corr[last]
    ):
        last += 1
    count = last - first + 1
    if count < minimum_points:
        raise ValueError(
            f"decay branch has {count} points; preregistration requires {minimum_points}"
        )
    x = profile.distance_pixels[first : last + 1]
    y = np.log(corr[first : last + 1])
    weights = profile.pair_weight[first : last + 1]
    coefficients = np.polyfit(x, y, deg=1, w=np.sqrt(weights))
    slope, intercept = (float(value) for value in coefficients)
    if slope >= 0.0:
        raise ValueError("fitted exponential slope is not negative")
    predicted = slope * x + intercept
    weighted_mean = float(np.average(y, weights=weights))
    residual = float(np.sum(weights * (y - predicted) ** 2))
    total = float(np.sum(weights * (y - weighted_mean) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return DecayFit(
        length_pixels=-1.0 / slope,
        slope_per_pixel=slope,
        intercept=intercept,
        r_squared=r_squared,
        first_index=first,
        last_index=last,
        point_count=count,
    )


def rms_positive_correlation_radius(profile: CorrelationProfile) -> float:
    """Return the radial-area-weighted RMS radius before the first zero crossing."""

    corr = profile.correlation
    nonpositive = np.flatnonzero(corr[1:] <= 0.0)
    stop = int(nonpositive[0] + 1) if nonpositive.size else corr.size
    radius = profile.distance_pixels[1:stop]
    positive = corr[1:stop]
    weights = profile.pair_weight[1:stop] * radius
    denominator = float(np.sum(positive * weights))
    if denominator <= 0.0:
        raise ValueError("positive correlation branch has no positive radial moment")
    numerator = float(np.sum(radius * radius * positive * weights))
    return float(np.sqrt(numerator / denominator))


def structural_correlation(
    field: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    maximum_lag_pixels: int | None = None,
) -> StructuralCorrelationResult:
    """Evaluate the preregistered radial and directional structural lengths."""

    values, mask = _validated_field_and_mask(field, valid_mask)
    correlation, pairs = mask_corrected_autocorrelation(values, valid_mask=mask)
    radial, x_profile, y_profile = correlation_profiles(
        correlation,
        pairs,
        maximum_lag_pixels=maximum_lag_pixels,
    )
    rms_radius = rms_positive_correlation_radius(radial)
    return StructuralCorrelationResult(
        radial=radial,
        x_direction=x_profile,
        y_direction=y_profile,
        radial_decay=fit_exponential_decay(radial),
        x_decay=fit_exponential_decay(x_profile),
        y_decay=fit_exponential_decay(y_profile),
        rms_radius_pixels=rms_radius,
        rms_control_length_pixels=rms_radius / 2.0,
        valid_fraction=float(np.mean(mask)),
    )
