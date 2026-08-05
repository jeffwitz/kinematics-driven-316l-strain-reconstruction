"""Pure numerical metrics for paired crystal-slip comparisons.

The functions in this module deliberately do not import Matplotlib.  They
operate on pixel-mean per-system fields with shape ``(system, y, x)`` and keep
amplitude, distribution, and spatial-shape comparisons separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def validate_slip_system_order(
    observed: list[str] | tuple[str, ...], expected: list[str] | tuple[str, ...]
) -> None:
    """Fail explicitly when per-system labels do not match the MFront order."""

    if tuple(observed) != tuple(expected):
        raise ValueError("slip-system order does not match the registered MFront order")


@dataclass(frozen=True, slots=True)
class SlipMetricConfig:
    """Thresholds used by the comparison, recorded in its JSON output."""

    dominant_fraction_threshold: float = 0.05
    cumulative_dominance_threshold: float = 0.95
    numerical_zero_tolerance: float = 1.0e-12
    support_fractions: tuple[float, ...] = (0.10, 0.25)

    def __post_init__(self) -> None:
        if not 0.0 < self.dominant_fraction_threshold <= 1.0:
            raise ValueError("dominant_fraction_threshold must lie in (0, 1]")
        if not 0.0 < self.cumulative_dominance_threshold <= 1.0:
            raise ValueError("cumulative_dominance_threshold must lie in (0, 1]")
        if self.numerical_zero_tolerance < 0.0:
            raise ValueError("numerical_zero_tolerance must be non-negative")
        if any(not 0.0 < value <= 1.0 for value in self.support_fractions):
            raise ValueError("support fractions must lie in (0, 1]")


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _safe_ratio(numerator: float, denominator: float, tolerance: float = 0.0) -> float | None:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return None
    if abs(denominator) <= tolerance:
        return None
    return float(numerator / denominator)


def _zero(value: float, config: SlipMetricConfig) -> bool:
    return abs(value) <= max(config.numerical_zero_tolerance, np.finfo(float).eps)


def _rankdata(values: FloatArray, tolerance: float) -> FloatArray:
    """Average ranks with deterministic ties, including numerical zeros."""

    clean = np.asarray(values, dtype=np.float64).copy()
    clean[np.abs(clean) <= tolerance] = 0.0
    order = np.argsort(clean, kind="stable")
    ranks = np.empty(clean.size, dtype=np.float64)
    sorted_values = clean[order]
    start = 0
    while start < clean.size:
        stop = start + 1
        while stop < clean.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _correlation(
    first: FloatArray, second: FloatArray, *, rank: bool, tolerance: float
) -> float | None:
    if first.shape != second.shape or first.size < 2:
        return None
    left = _rankdata(first, tolerance) if rank else np.array(first, dtype=np.float64, copy=True)
    right = _rankdata(second, tolerance) if rank else np.array(second, dtype=np.float64, copy=True)
    left -= np.mean(left)
    right -= np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= max(tolerance, np.finfo(float).eps):
        return None
    return float(np.dot(left, right) / denominator)


def _cosine_similarity(first: FloatArray, second: FloatArray, tolerance: float) -> float | None:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= max(tolerance, np.finfo(float).eps):
        return None
    return float(np.dot(left, right) / denominator)


def _validate_fields(meric: FloatArray, srix: FloatArray) -> tuple[FloatArray, FloatArray]:
    left = np.asarray(meric, dtype=np.float64)
    right = np.asarray(srix, dtype=np.float64)
    if left.ndim != 3 or right.ndim != 3 or left.shape != right.shape:
        raise ValueError("paired slip fields must have the same shape (system, y, x)")
    if left.shape[0] != 12:
        raise ValueError(f"expected twelve slip systems, got {left.shape[0]}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("slip fields must be finite")
    return left, right


def amplitude_statistics(field: FloatArray) -> dict[str, float | None]:
    """Statistics of a scalar pixel field, with JSON-safe scalar values."""

    values = _finite(np.asarray(field, dtype=np.float64).ravel())
    if values.size == 0:
        return {
            name: None
            for name in (
                "minimum",
                "mean",
                "median",
                "maximum",
                "std",
                "q05",
                "q25",
                "q75",
                "q95",
                "q99",
                "integral",
            )
        }
    return {
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
        "std": float(np.std(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "integral": float(np.sum(values)),
    }


def _stat_value(statistics: dict[str, float | None], name: str) -> float:
    value = statistics[name]
    return float(value) if value is not None else float("nan")


def jaccard(
    first: set[int] | list[int] | tuple[int, ...], second: set[int] | list[int] | tuple[int, ...]
) -> float:
    left, right = set(first), set(second)
    union = left | right
    return 1.0 if not union else float(len(left & right) / len(union))


def _dominant_indices(fractions: FloatArray, threshold: float) -> tuple[int, ...]:
    order = np.argsort(-fractions, kind="stable")
    cumulative = 0.0
    selected: list[int] = []
    for index in order:
        selected.append(int(index))
        cumulative += float(fractions[index])
        if cumulative + 1.0e-15 >= threshold:
            break
    return tuple(selected)


def _system_distribution(field: FloatArray, config: SlipMetricConfig) -> dict[str, Any]:
    totals = np.sum(field, axis=(1, 2), dtype=np.float64)
    total = float(np.sum(totals))
    fractions = totals / total if not _zero(total, config) else np.zeros_like(totals)
    order = np.argsort(-fractions, kind="stable")
    ranks = np.empty(12, dtype=int)
    ranks[order] = np.arange(1, 13)
    s95 = _dominant_indices(fractions, config.cumulative_dominance_threshold) if total else ()
    s5 = tuple(
        int(index) for index in np.flatnonzero(fractions >= config.dominant_fraction_threshold)
    )
    return {
        "totals": totals,
        "fractions": fractions,
        "ranks": ranks,
        "order": tuple(int(index) for index in order),
        "s95": s95,
        "s5": s5,
        "total": total,
    }


def _distribution_comparison(
    meric: dict[str, Any], srix: dict[str, Any], config: SlipMetricConfig
) -> dict[str, Any]:
    mf, sf = meric["fractions"], srix["fractions"]
    cosine_denominator = float(np.linalg.norm(mf) * np.linalg.norm(sf))
    cosine = _safe_ratio(float(np.dot(mf, sf)), cosine_denominator, config.numerical_zero_tolerance)
    rank_correlation = _correlation(
        mf, sf, rank=True, tolerance=config.numerical_zero_tolerance
    )
    return {
        "meric": {
            "s95": list(meric["s95"]),
            "s5": list(meric["s5"]),
            "top_system": int(meric["order"][0]) if meric["order"] else None,
            "top3": list(meric["order"][:3]),
        },
        "srix": {
            "s95": list(srix["s95"]),
            "s5": list(srix["s5"]),
            "top_system": int(srix["order"][0]) if srix["order"] else None,
            "top3": list(srix["order"][:3]),
        },
        "s95_jaccard": jaccard(meric["s95"], srix["s95"]),
        "s5_jaccard": jaccard(meric["s5"], srix["s5"]),
        "s95_intersection": sorted(set(meric["s95"]) & set(srix["s95"])),
        "s95_union": sorted(set(meric["s95"]) | set(srix["s95"])),
        "s5_intersection": sorted(set(meric["s5"]) & set(srix["s5"])),
        "s5_union": sorted(set(meric["s5"]) | set(srix["s5"])),
        "top3_overlap": len(set(meric["order"][:3]) & set(srix["order"][:3])),
        "total_variation_distance": float(0.5 * np.sum(np.abs(mf - sf))),
        "cosine_similarity": cosine,
        "spearman_rank_correlation": rank_correlation,
        "fraction_difference_l1": float(np.sum(np.abs(mf - sf))),
    }


def _support(field: FloatArray, fraction: float) -> NDArray[np.bool_]:
    flat = np.asarray(field, dtype=np.float64).ravel()
    count = max(1, min(flat.size, ceil(fraction * flat.size)))
    indices = np.argpartition(flat, -count)[-count:]
    result = np.zeros(flat.size, dtype=bool)
    result[indices] = True
    return result.reshape(field.shape)


def _support_metrics(first: FloatArray, second: FloatArray, fraction: float) -> dict[str, float]:
    left, right = _support(first, fraction), _support(second, fraction)
    intersection = float(np.count_nonzero(left & right))
    union = float(np.count_nonzero(left | right))
    left_count, right_count = float(np.count_nonzero(left)), float(np.count_nonzero(right))
    return {
        "fraction": fraction,
        "iou": intersection / union if union else 1.0,
        "dice": 2.0 * intersection / (left_count + right_count)
        if left_count + right_count
        else 1.0,
        "meric_recall": intersection / left_count if left_count else 1.0,
        "srix_recall": intersection / right_count if right_count else 1.0,
    }


def _barycentre(field: FloatArray) -> tuple[float, float] | None:
    total = float(np.sum(field))
    if total <= np.finfo(float).eps:
        return None
    y, x = np.indices(field.shape, dtype=np.float64)
    return float(np.sum(x * field) / total), float(np.sum(y * field) / total)


def _spatial_system(
    first: FloatArray, second: FloatArray, config: SlipMetricConfig
) -> dict[str, Any]:
    first_total, second_total = float(np.sum(first)), float(np.sum(second))
    if _zero(first_total, config) or _zero(second_total, config):
        return {"status": "not_significant", "absolute": None, "normalized": None, "supports": None}
    difference = first - second
    first_normalized, second_normalized = first / first_total, second / second_total
    normalized_difference = first_normalized - second_normalized
    absolute = {
        "l1": float(np.sum(np.abs(difference))),
        "l2": float(np.linalg.norm(difference)),
        "relative_l2": float(
            np.linalg.norm(difference) / max(np.linalg.norm(first), np.finfo(float).eps)
        ),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "integral_ratio_meric_over_srix": first_total / second_total,
        "pearson": _correlation(
            first.ravel(), second.ravel(), rank=False, tolerance=config.numerical_zero_tolerance
        ),
        "spearman": _correlation(
            first.ravel(), second.ravel(), rank=True, tolerance=config.numerical_zero_tolerance
        ),
    }
    normalized = {
        "l1": float(np.sum(np.abs(normalized_difference))),
        "l2": float(np.linalg.norm(normalized_difference)),
        "cosine": _cosine_similarity(
            first_normalized.ravel(), second_normalized.ravel(), config.numerical_zero_tolerance
        ),
        "pearson": _correlation(
            first_normalized.ravel(),
            second_normalized.ravel(),
            rank=False,
            tolerance=config.numerical_zero_tolerance,
        ),
        "spearman": _correlation(
            first_normalized.ravel(),
            second_normalized.ravel(),
            rank=True,
            tolerance=config.numerical_zero_tolerance,
        ),
        "meric_barycentre_xy": _barycentre(first_normalized),
        "srix_barycentre_xy": _barycentre(second_normalized),
    }
    if normalized["meric_barycentre_xy"] and normalized["srix_barycentre_xy"]:
        normalized["barycentre_distance"] = float(
            np.linalg.norm(
                np.subtract(normalized["meric_barycentre_xy"], normalized["srix_barycentre_xy"])
            )
        )
    else:
        normalized["barycentre_distance"] = None
    return {
        "status": "significant",
        "absolute": absolute,
        "normalized": normalized,
        "supports": [
            _support_metrics(first, second, fraction) for fraction in config.support_fractions
        ],
    }


def _signed_pair_metrics(
    meric: FloatArray,
    srix: FloatArray,
    weights: FloatArray,
    config: SlipMetricConfig,
) -> dict[str, Any]:
    meric_active = np.abs(meric) > config.numerical_zero_tolerance
    srix_active = np.abs(srix) > config.numerical_zero_tolerance
    union_active = meric_active | srix_active
    both_active = meric_active & srix_active
    meric_only = meric_active & ~srix_active
    srix_only = srix_active & ~meric_active
    same_sign = both_active & (np.sign(meric) == np.sign(srix))
    opposite_sign = both_active & (np.sign(meric) != np.sign(srix))
    if not np.any(union_active):
        return {
            "both_active_fraction": None,
            "same_sign_fraction_among_both_active": None,
            "opposite_sign_fraction_among_both_active": None,
            "meric_only_fraction": None,
            "srix_only_fraction": None,
            "weighted_both_active_fraction": None,
            "weighted_same_sign_fraction_among_both_active": None,
            "weighted_opposite_sign_fraction_among_both_active": None,
            "weighted_meric_only_fraction": None,
            "weighted_srix_only_fraction": None,
            "comparable_pixels": 0,
            "status": "not_significant",
        }
    selected_weights = np.asarray(weights, dtype=np.float64)[union_active]
    weight_total = float(np.sum(selected_weights))
    both_weights = float(np.sum(np.asarray(weights, dtype=np.float64)[both_active]))
    union_count = float(np.count_nonzero(union_active))
    both_count = float(np.count_nonzero(both_active))
    return {
        "both_active_fraction": both_count / union_count,
        "same_sign_fraction_among_both_active": (
            float(np.count_nonzero(same_sign) / both_count) if both_count else None
        ),
        "opposite_sign_fraction_among_both_active": (
            float(np.count_nonzero(opposite_sign) / both_count) if both_count else None
        ),
        "meric_only_fraction": float(np.count_nonzero(meric_only) / union_count),
        "srix_only_fraction": float(np.count_nonzero(srix_only) / union_count),
        "weighted_both_active_fraction": (
            both_weights / weight_total if weight_total > np.finfo(float).eps else None
        ),
        "weighted_same_sign_fraction_among_both_active": (
            float(np.sum(np.asarray(weights, dtype=np.float64)[same_sign]) / both_weights)
            if both_weights > np.finfo(float).eps
            else None
        ),
        "weighted_opposite_sign_fraction_among_both_active": (
            float(np.sum(np.asarray(weights, dtype=np.float64)[opposite_sign]) / both_weights)
            if both_weights > np.finfo(float).eps
            else None
        ),
        "weighted_meric_only_fraction": (
            float(np.sum(np.asarray(weights, dtype=np.float64)[meric_only]) / weight_total)
            if weight_total > np.finfo(float).eps
            else None
        ),
        "weighted_srix_only_fraction": (
            float(np.sum(np.asarray(weights, dtype=np.float64)[srix_only]) / weight_total)
            if weight_total > np.finfo(float).eps
            else None
        ),
        "comparable_pixels": int(np.count_nonzero(union_active)),
        "status": "significant",
    }


def _signed_metrics(
    meric: FloatArray | None,
    srix: FloatArray | None,
    meric_magnitude: FloatArray | None,
    srix_magnitude: FloatArray | None,
    config: SlipMetricConfig,
) -> dict[str, Any] | None:
    if meric is None or srix is None or meric_magnitude is None or srix_magnitude is None:
        return None
    if (
        meric.shape != srix.shape
        or meric.shape != meric_magnitude.shape
        or meric.shape != srix_magnitude.shape
    ):
        raise ValueError("signed and magnitude fields must have identical shapes")
    per_system = [
        {
            "system": index,
            **_signed_pair_metrics(
                meric[index],
                srix[index],
                0.5 * (meric_magnitude[index] + srix_magnitude[index]),
                config,
            ),
        }
        for index in range(meric.shape[0])
    ]
    all_weights = 0.5 * (meric_magnitude + srix_magnitude)
    meric_active = np.abs(meric) > config.numerical_zero_tolerance
    srix_active = np.abs(srix) > config.numerical_zero_tolerance
    union_active = meric_active | srix_active
    both_active = meric_active & srix_active
    meric_only = meric_active & ~srix_active
    srix_only = srix_active & ~meric_active
    same_sign = both_active & (np.sign(meric) == np.sign(srix))
    opposite_sign = both_active & (np.sign(meric) != np.sign(srix))
    selected_weights = all_weights[union_active]
    both_weights = float(np.sum(all_weights[both_active]))
    weight_total = float(np.sum(selected_weights))
    return {
        "available": True,
        "chronology_available": False,
        "both_active_fraction_by_system": [item["both_active_fraction"] for item in per_system],
        "same_sign_fraction_among_both_active_by_system": [
            item["same_sign_fraction_among_both_active"] for item in per_system
        ],
        "opposite_sign_fraction_among_both_active_by_system": [
            item["opposite_sign_fraction_among_both_active"] for item in per_system
        ],
        "meric_only_fraction_by_system": [item["meric_only_fraction"] for item in per_system],
        "srix_only_fraction_by_system": [item["srix_only_fraction"] for item in per_system],
        "comparable_pixels_by_system": [item["comparable_pixels"] for item in per_system],
        "weighted_both_active_fraction": (
            both_weights / weight_total if weight_total > np.finfo(float).eps else None
        ),
        "weighted_same_sign_fraction_among_both_active": (
            float(np.sum(all_weights[same_sign]) / both_weights)
            if both_weights > np.finfo(float).eps
            else None
        ),
        "weighted_opposite_sign_fraction_among_both_active": (
            float(np.sum(all_weights[opposite_sign]) / both_weights)
            if both_weights > np.finfo(float).eps
            else None
        ),
        "weighted_meric_only_fraction": (
            float(np.sum(all_weights[meric_only]) / weight_total)
            if weight_total > np.finfo(float).eps
            else None
        ),
        "weighted_srix_only_fraction": (
            float(np.sum(all_weights[srix_only]) / weight_total)
            if weight_total > np.finfo(float).eps
            else None
        ),
        "per_system": per_system,
    }


def compare_slip_fields(
    meric_equivalent: FloatArray,
    srix_equivalent: FloatArray,
    *,
    meric_signed: FloatArray | None = None,
    srix_signed: FloatArray | None = None,
    config: SlipMetricConfig | None = None,
) -> dict[str, Any]:
    """Return JSON-serialisable metrics for final per-system slip fields."""

    settings = config or SlipMetricConfig()
    meric, srix = _validate_fields(meric_equivalent, srix_equivalent)
    if meric_signed is not None or srix_signed is not None:
        if meric_signed is None or srix_signed is None:
            raise ValueError("signed fields must be supplied for both laws")
        meric_signed, srix_signed = _validate_fields(meric_signed, srix_signed)
    meric_total_field, srix_total_field = np.sum(meric, axis=0), np.sum(srix, axis=0)
    meric_distribution = _system_distribution(meric, settings)
    srix_distribution = _system_distribution(srix, settings)
    systems: list[dict[str, Any]] = [
        {
            "system": index,
            "meric_total": float(meric_distribution["totals"][index]),
            "srix_total": float(srix_distribution["totals"][index]),
            "meric_fraction": float(meric_distribution["fractions"][index]),
            "srix_fraction": float(srix_distribution["fractions"][index]),
            "meric_rank": int(meric_distribution["ranks"][index]),
            "srix_rank": int(srix_distribution["ranks"][index]),
            "fraction_difference": float(
                meric_distribution["fractions"][index] - srix_distribution["fractions"][index]
            ),
            "amplitude_ratio_meric_over_srix": _safe_ratio(
                meric_distribution["totals"][index],
                srix_distribution["totals"][index],
                settings.numerical_zero_tolerance,
            ),
            "spatial": _spatial_system(meric[index], srix[index], settings),
        }
        for index in range(12)
    ]
    meric_statistics = amplitude_statistics(meric_total_field)
    srix_statistics = amplitude_statistics(srix_total_field)
    summary: dict[str, Any] = {
        "configuration": {
            "dominant_fraction_threshold": settings.dominant_fraction_threshold,
            "cumulative_dominance_threshold": settings.cumulative_dominance_threshold,
            "numerical_zero_tolerance": settings.numerical_zero_tolerance,
            "support_fractions": list(settings.support_fractions),
        },
        "global_amplitude": {
            "meric": meric_statistics,
            "srix": srix_statistics,
            "ratios_meric_over_srix": {
                name: _safe_ratio(
                    _stat_value(meric_statistics, name),
                    _stat_value(srix_statistics, name),
                    settings.numerical_zero_tolerance,
                )
                for name in ("mean", "median", "maximum", "integral", "q95")
            },
        },
        "system_distribution": _distribution_comparison(
            meric_distribution, srix_distribution, settings
        ),
        "systems": systems,
        "spatial_similarity": {
            "total_accumulated_system_slip": _spatial_system(
                meric_total_field, srix_total_field, settings
            ),
            "per_system": [
                {"system": item["system"], **cast(dict[str, Any], item["spatial"])}
                for item in systems
            ],
        },
        "signed_slip": _signed_metrics(
            meric_signed,
            srix_signed,
            meric,
            srix,
            settings,
        ),
        "incremental_similarity": {
            "available": False,
            "reason": "The archived P43 field files contain final fields only; no "
            "per-increment per-system slip history is available.",
        },
        "limitations": [
            "The comparison uses the pixel mean of the two TRI2 states.",
            "The archived source contains final per-system fields but no incremental "
            "per-system history.",
            "Signed-slip metrics compare final signed fields; sign changes through "
            "loading cannot be assessed.",
            "R is an analytical transposition, not a direct identification on P43.",
            "The Méric 16-increment calculation is converged numerically but not "
            "temporally converged.",
            "The orientation is homogeneous and is not an EBSD orientation map.",
        ],
    }
    return summary
