"""Validated scalar DIC noise references for displacement whitening."""

from __future__ import annotations

import json
from pathlib import Path


def load_dic_noise_reference(
    report_path: str | Path,
    *,
    pixel_size_mm: float,
    relative_tolerance: float = 1.0e-10,
) -> dict[str, float | str]:
    """Load and cross-check the registered per-state DIC noise estimate.

    The report stores both the robust estimate in pixels and its converted
    value in millimetres.  Keeping both values in the artifact makes a unit
    mismatch detectable instead of silently propagating a hard-coded scalar.
    """

    path = Path(report_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    temporal = payload["temporal_noise"]
    robust_px = float(temporal["robust_px"])
    robust_mm = float(temporal["robust_mm"])
    expected_mm = robust_px * float(pixel_size_mm)
    scale = max(abs(robust_mm), abs(expected_mm), 1.0e-30)
    if abs(robust_mm - expected_mm) / scale > relative_tolerance:
        raise ValueError(
            "DIC noise report has inconsistent pixel/mm values: "
            f"{robust_px} px * {pixel_size_mm} mm/px != {robust_mm} mm"
        )
    if robust_px <= 0.0 or robust_mm <= 0.0:
        raise ValueError("registered DIC noise must be positive")
    return {
        "report_path": str(path),
        "robust_px": robust_px,
        "robust_mm": robust_mm,
        "rms_px": float(temporal["rms_px"]),
        "rms_mm": float(temporal["rms_mm"]),
        "pixel_size_mm": float(pixel_size_mm),
    }
