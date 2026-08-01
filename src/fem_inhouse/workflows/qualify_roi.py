"""Decide whether a ROI can test a diffusion length, from the local run alone.

The gate the P43 campaign lacked. A ROI is only worth a coupled matrix if the
local model already places the bands roughly right **and makes them measurably
too narrow**, because widening is what a scalar regularisation of `p` can do.
On P43 the local model had the *best* width error of any model and coupling
degraded it, so the mechanism had no room to act and ten hours of matrix could
not have identified anything.

Everything here reads the DIC and one local run. No coupled computation.

The width used throughout is the **section integral width** of
`band_profiles.measure_width`, the same operator any later campaign would test
with. The Otsu minor axis is reported beside it because the two differ by about
a factor `2.5` on P43, so a threshold expressed in MTF-50 means different things
depending on which is meant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.validation.band_profiles import (
    estimate_background,
    measure_position,
    measure_width,
    sample_normal_profile,
)
from fem_inhouse.validation.otsu_morphology import describe_morphology, otsu_threshold
from fem_inhouse.workflows.compare_observed_evm_candidates import (
    BORDER_MARGIN_PIXELS,
    CORRIDOR_HALF_WIDTH_PIXELS,
    MINIMUM_AREA_PIXELS,
    MTF50_PIXELS,
    SECTION_HALF_LENGTH_PIXELS,
    extract_bands,
)

FloatArray = NDArray[np.float64]

#: Registered qualification thresholds of the 2026-08-01 review.
MINIMUM_VALID_SECTION_FRACTION = 0.75
MINIMUM_DIC_WIDTH_MTF50 = 1.5
MINIMUM_WIDTH_DEFICIT = 0.25
MINIMUM_CONSISTENT_SIGN_FRACTION = 0.70
MAXIMUM_CENTRELINE_OFFSET_FRACTION = 0.25

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260801


@dataclass(frozen=True, slots=True)
class BandWidths:
    """Per-section widths of both fields on one band, and their validity."""

    name: str
    dic: FloatArray
    local: FloatArray
    valid: NDArray[np.bool_]
    centreline_offset: FloatArray


def _section_widths(
    field: FloatArray, band: dict[str, Any]
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    widths: list[float] = []
    offsets: list[float] = []
    valid: list[bool] = []
    for index, (origin, normal) in enumerate(zip(band["centreline"], band["normals"], strict=True)):
        profile = sample_normal_profile(
            field,
            origin=(float(origin[0]), float(origin[1])),
            normal=(float(normal[0]), float(normal[1])),
            half_length_pixels=SECTION_HALF_LENGTH_PIXELS,
            section_id=index,
            border_margin_pixels=BORDER_MARGIN_PIXELS,
        )
        if not profile.valid:
            widths.append(np.nan)
            offsets.append(np.nan)
            valid.append(False)
            continue
        background = estimate_background(
            profile, corridor_half_width_pixels=CORRIDOR_HALF_WIDTH_PIXELS
        )
        widths.append(measure_width(profile, background).integral_pixels)
        offsets.append(measure_position(profile, background)["centroid_offset"])
        valid.append(True)
    return (
        np.asarray(widths, dtype=np.float64),
        np.asarray(offsets, dtype=np.float64),
        np.asarray(valid, dtype=bool),
    )


def _ratio_interval(
    dic: FloatArray, local: FloatArray, usable: NDArray[np.bool_]
) -> dict[str, float]:
    """Bootstrap interval of the median width ratio, resampling sections."""

    paired = usable & np.isfinite(dic) & np.isfinite(local) & (local > 0.0)
    count = int(paired.sum())
    if count < 8:
        return {"count": count, "median": float("nan"), "q05": float("nan"), "q95": float("nan")}
    ratio = dic[paired] / local[paired]
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.array(
        [
            float(np.median(ratio[generator.integers(0, count, count)]))
            for _ in range(BOOTSTRAP_DRAWS)
        ]
    )
    return {
        "count": count,
        "median": float(np.median(ratio)),
        "q05": float(np.quantile(draws, 0.05)),
        "q95": float(np.quantile(draws, 0.95)),
    }


def qualify_roi(
    *,
    dic_evm: FloatArray,
    local_evm: FloatArray,
    partition_id: int,
) -> dict[str, Any]:
    """Apply the registered filter and return every measurement behind it."""

    if dic_evm.shape != local_evm.shape:
        raise ValueError("the two fields must share the same support")

    threshold = otsu_threshold(dic_evm)
    dic_morphology = describe_morphology(
        dic_evm, threshold=threshold, label_name="dic", minimum_area_pixels=MINIMUM_AREA_PIXELS
    )
    local_morphology = describe_morphology(
        local_evm,
        threshold=threshold,
        label_name="local",
        minimum_area_pixels=MINIMUM_AREA_PIXELS,
    )
    bands = extract_bands(dic_evm, threshold=threshold)

    per_band: dict[str, Any] = {}
    for name, band in bands.items():
        dic_widths, dic_offsets, dic_valid = _section_widths(dic_evm, band)
        local_widths, local_offsets, local_valid = _section_widths(local_evm, band)
        usable = dic_valid & local_valid
        deficit = local_widths < dic_widths
        comparable = usable & np.isfinite(dic_widths) & np.isfinite(local_widths)
        per_band[name] = {
            "sections": int(dic_valid.size),
            "valid_fraction": float(usable.sum() / dic_valid.size),
            "dic_width_median": float(np.nanmedian(dic_widths[usable]))
            if usable.any()
            else float("nan"),
            "local_width_median": float(np.nanmedian(local_widths[usable]))
            if usable.any()
            else float("nan"),
            "width_ratio": _ratio_interval(dic_widths, local_widths, usable),
            "narrower_fraction": (
                float(deficit[comparable].sum() / comparable.sum())
                if comparable.any()
                else float("nan")
            ),
            "centreline_offset_median": (
                float(np.nanmedian(np.abs(local_offsets[usable] - dic_offsets[usable])))
                if usable.any()
                else float("nan")
            ),
        }

    checks: dict[str, Any] = {}
    checks["same_object_count"] = {
        "passed": dic_morphology.object_count == local_morphology.object_count,
        "dic": dic_morphology.object_count,
        "local": local_morphology.object_count,
    }
    worst_valid = min((b["valid_fraction"] for b in per_band.values()), default=0.0)
    checks["enough_usable_sections"] = {
        "passed": worst_valid >= MINIMUM_VALID_SECTION_FRACTION,
        "worst_band_valid_fraction": worst_valid,
        "bound": MINIMUM_VALID_SECTION_FRACTION,
    }
    dic_width = min((b["dic_width_median"] for b in per_band.values()), default=float("nan"))
    checks["dic_band_is_resolved"] = {
        "passed": bool(dic_width >= MINIMUM_DIC_WIDTH_MTF50 * MTF50_PIXELS),
        "narrowest_dic_width_px": dic_width,
        "in_mtf50": dic_width / MTF50_PIXELS if np.isfinite(dic_width) else float("nan"),
        "bound_mtf50": MINIMUM_DIC_WIDTH_MTF50,
    }
    ratios = [b["width_ratio"] for b in per_band.values()]
    checks["local_is_measurably_narrower"] = {
        "passed": all(
            np.isfinite(r["median"]) and r["median"] >= 1.0 / (1.0 - MINIMUM_WIDTH_DEFICIT)
            for r in ratios
        ),
        "width_ratios": ratios,
        "bound": 1.0 / (1.0 - MINIMUM_WIDTH_DEFICIT),
    }
    checks["ratio_interval_excludes_one"] = {
        "passed": all(np.isfinite(r["q05"]) and r["q05"] > 1.0 for r in ratios),
        "q05": [r["q05"] for r in ratios],
    }
    narrower = [b["narrower_fraction"] for b in per_band.values()]
    checks["deficit_sign_is_consistent"] = {
        "passed": all(np.isfinite(v) and v >= MINIMUM_CONSISTENT_SIGN_FRACTION for v in narrower),
        "narrower_fraction": narrower,
        "bound": MINIMUM_CONSISTENT_SIGN_FRACTION,
    }
    offsets = [
        b["centreline_offset_median"] / b["dic_width_median"]
        if np.isfinite(b["dic_width_median"]) and b["dic_width_median"] > 0
        else float("nan")
        for b in per_band.values()
    ]
    checks["centreline_is_close"] = {
        "passed": all(np.isfinite(v) and v <= MAXIMUM_CENTRELINE_OFFSET_FRACTION for v in offsets),
        "offset_over_dic_width": offsets,
        "bound": MAXIMUM_CENTRELINE_OFFSET_FRACTION,
    }

    failed = sorted(name for name, check in checks.items() if not check["passed"])
    return {
        "partition_id": partition_id,
        "otsu_threshold": threshold,
        "morphology": {
            "dic_objects": dic_morphology.object_count,
            "local_objects": local_morphology.object_count,
            "dic_minor_axes_px": [o.axis_minor_pixels for o in dic_morphology.objects],
            "local_minor_axes_px": [o.axis_minor_pixels for o in local_morphology.objects],
        },
        "bands": per_band,
        "checks": checks,
        "failed": failed,
        "qualified": not failed,
    }


def format_report(result: dict[str, Any]) -> str:
    """A short human-readable verdict."""

    lines = [
        f"partition {result['partition_id']:03d}: "
        f"{'QUALIFIED' if result['qualified'] else 'REJECTED'}"
    ]
    if result["failed"]:
        lines.append(f"  fails: {', '.join(result['failed'])}")
    for name, check in result["checks"].items():
        mark = "pass" if check["passed"] else "FAIL"
        extra = {k: v for k, v in check.items() if k != "passed"}
        lines.append(
            f"  {mark:4s} {name:32s} {json.dumps(extra, default=lambda v: round(float(v), 4))}"
        )
    return "\n".join(lines)
