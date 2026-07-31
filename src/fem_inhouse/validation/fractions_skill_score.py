"""Multiscale agreement of active area between two fields.

A pixel-to-pixel overlap score punishes a band twice when it is slightly
displaced: once for being absent where it should be, once for being present
where it should not. The fractions skill score asks a different question — at
what spatial scale does the active area of the candidate become compatible with
the reference — and so separates "wrong place" from "wrong amount".

Thresholds are computed on the reference and applied to the candidate without
recalibration. Recomputing a quantile per candidate would let each field define
its own notion of "active" and make the comparison meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

#: Neighbourhood sizes in pixels. Registered before any candidate is analysed.
DEFAULT_SCALES_PIXELS: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96)

#: Skill levels whose first attaining scale is reported.
DEFAULT_SKILL_LEVELS: tuple[float, ...] = (0.5, 0.7, 0.9)


@dataclass(frozen=True, slots=True)
class SkillCurve:
    """Skill against neighbourhood size at one threshold."""

    threshold_quantile: float
    threshold_value: float
    scales_pixels: tuple[int, ...]
    values: FloatArray
    reference_active_fraction: float
    candidate_active_fraction: float


def active_fraction_field(
    active: NDArray[np.bool_],
    *,
    scale_pixels: int,
    valid_mask: NDArray[np.bool_] | None = None,
) -> FloatArray:
    """Fraction of valid pixels that are active, in a square neighbourhood.

    Normalising by the count of valid pixels in each window, rather than by the
    window area, keeps the fraction meaningful at the support edge and around
    an invalid region: a half-covered window is not reported as half empty.
    """

    flags = np.asarray(active, dtype=bool)
    if flags.ndim != 2:
        raise ValueError("active must be two-dimensional")
    if scale_pixels < 1:
        raise ValueError("scale_pixels must be positive")
    valid = (
        np.ones_like(flags)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != flags.shape:
        raise ValueError("valid_mask must match the field shape")
    size = int(scale_pixels)
    numerator = ndimage.uniform_filter(
        (flags & valid).astype(np.float64), size=size, mode="constant", cval=0.0
    )
    denominator = ndimage.uniform_filter(
        valid.astype(np.float64), size=size, mode="constant", cval=0.0
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


def fractions_skill_score(
    reference_active: NDArray[np.bool_],
    candidate_active: NDArray[np.bool_],
    *,
    scale_pixels: int,
    valid_mask: NDArray[np.bool_] | None = None,
) -> float:
    """Return ``1 - <(f_c - f_r)^2> / <f_c^2 + f_r^2>`` at one scale.

    Two fields with no active pixel anywhere give ``0/0``; the score is then
    undefined and ``nan`` is returned rather than a flattering ``1.0``.
    """

    reference = np.asarray(reference_active, dtype=bool)
    candidate = np.asarray(candidate_active, dtype=bool)
    if reference.shape != candidate.shape:
        raise ValueError("the two fields must share a shape")
    valid = (
        np.ones_like(reference)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != reference.shape:
        raise ValueError("valid_mask must match the field shape")
    if not valid.any():
        return float("nan")

    reference_fraction = active_fraction_field(
        reference, scale_pixels=scale_pixels, valid_mask=valid
    )[valid]
    candidate_fraction = active_fraction_field(
        candidate, scale_pixels=scale_pixels, valid_mask=valid
    )[valid]
    numerator = float(np.mean((candidate_fraction - reference_fraction) ** 2))
    denominator = float(np.mean(candidate_fraction**2 + reference_fraction**2))
    if denominator <= 0.0:
        return float("nan")
    return 1.0 - numerator / denominator


def skill_curve(
    reference: NDArray[np.generic],
    candidate: NDArray[np.generic],
    *,
    threshold_value: float,
    threshold_quantile: float,
    scales_pixels: tuple[int, ...] = DEFAULT_SCALES_PIXELS,
    valid_mask: NDArray[np.bool_] | None = None,
) -> SkillCurve:
    """Evaluate the skill of one candidate across the registered scales."""

    reference_values = np.asarray(reference, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    if reference_values.shape != candidate_values.shape:
        raise ValueError("the two fields must share a shape")
    if not scales_pixels:
        raise ValueError("at least one neighbourhood size is required")

    # One threshold, from the reference, applied unchanged to both fields.
    reference_active = reference_values >= threshold_value
    candidate_active = candidate_values >= threshold_value
    valid = (
        np.ones_like(reference_active)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    values = np.asarray(
        [
            fractions_skill_score(
                reference_active,
                candidate_active,
                scale_pixels=scale,
                valid_mask=valid,
            )
            for scale in scales_pixels
        ]
    )
    total = float(np.count_nonzero(valid))
    return SkillCurve(
        threshold_quantile=float(threshold_quantile),
        threshold_value=float(threshold_value),
        scales_pixels=tuple(int(s) for s in scales_pixels),
        values=values,
        reference_active_fraction=float(np.count_nonzero(reference_active & valid) / total),
        candidate_active_fraction=float(np.count_nonzero(candidate_active & valid) / total),
    )


def minimum_skilful_scale(curve: SkillCurve, *, level: float) -> float:
    """Smallest neighbourhood size at which the skill first reaches ``level``.

    Returns ``nan`` when the level is never reached, which is a result rather
    than a failure: it means the candidate is incompatible with the reference
    at every scale examined.
    """

    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    for scale, value in zip(curve.scales_pixels, curve.values, strict=True):
        if np.isfinite(value) and value >= level:
            return float(scale)
    return float("nan")


def skill_table(
    reference: NDArray[np.generic],
    candidate: NDArray[np.generic],
    *,
    thresholds: dict[float, float],
    scales_pixels: tuple[int, ...] = DEFAULT_SCALES_PIXELS,
    valid_mask: NDArray[np.bool_] | None = None,
    levels: tuple[float, ...] = DEFAULT_SKILL_LEVELS,
) -> dict[str, object]:
    """Build the threshold-by-scale table and its attaining scales.

    No single threshold is treated as truth; the disagreement between them is
    part of the result.
    """

    curves = [
        skill_curve(
            reference,
            candidate,
            threshold_value=value,
            threshold_quantile=quantile,
            scales_pixels=scales_pixels,
            valid_mask=valid_mask,
        )
        for quantile, value in sorted(thresholds.items())
    ]
    return {
        "scales_pixels": list(scales_pixels),
        "levels": list(levels),
        "curves": [
            {
                "threshold_quantile": curve.threshold_quantile,
                "threshold_value": curve.threshold_value,
                "values": [float(v) for v in curve.values],
                "reference_active_fraction": curve.reference_active_fraction,
                "candidate_active_fraction": curve.candidate_active_fraction,
                "minimum_skilful_scale": {
                    str(level): minimum_skilful_scale(curve, level=level)
                    for level in levels
                },
            }
            for curve in curves
        ],
    }
