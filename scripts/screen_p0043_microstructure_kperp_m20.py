#!/usr/bin/env python3
"""Preflight the P43 EBSD payload for microstructural ``k_perp`` screening.

The screening is deliberately gated by a defensible grain-label map.  An
orientation field described as a per-pixel grain mean is not silently treated
as a segmentation, because doing so would manufacture grain boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EBSD = Path(
    "/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5"
)
OUTPUT = ROOT / "validation/p0043_microstructure_kperp_screening_m20.json"
MARKDOWN = ROOT / "validation/p0043_microstructure_kperp_screening_m20.md"
CROP = (1610, 1630, 1075, 1095)
ORIENTATION_NAMES = ("phi1", "Phi", "phi2")


def _datasets(group: h5py.Group, prefix: str = "") -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name, value in group.items():
        path = f"{prefix}/{name}"
        if isinstance(value, h5py.Group):
            result.extend(_datasets(value, path))
        else:
            result.append(
                {
                    "path": path.lstrip("/"),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "attributes": {
                        str(key): value_attr.tolist()
                        if isinstance(value_attr, np.ndarray)
                        else value_attr.item()
                        if isinstance(value_attr, np.generic)
                        else value_attr
                        for key, value_attr in value.attrs.items()
                    },
                }
            )
    return result


def inspect_ebsd(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with h5py.File(path, "r") as handle:
        datasets = _datasets(handle)
        orientation = np.stack(
            [
                np.asarray(handle[f"orientation/{name}"][...], dtype=np.float64)
                for name in ORIENTATION_NAMES
            ],
            axis=-1,
        )
        crop = orientation[CROP[0] : CROP[1], CROP[2] : CROP[3]]
        grain_candidates = [
            item
            for item in datasets
            if any(
                token in item["path"].lower()
                for token in ("grain", "segment", "label", "boundary")
            )
        ]
        orientation_attributes = {
            name: next(
                item["attributes"]
                for item in datasets
                if item["path"] == f"orientation/{name}"
            )
            for name in ORIENTATION_NAMES
        }
        return {
            "source": str(path),
            "sha256": digest,
            "datasets": datasets,
            "orientation_shape": list(orientation.shape[:2]),
            "m20_crop": list(CROP),
            "m20_orientation_shape": list(crop.shape),
            "m20_unique_orientation_count": int(
                np.unique(crop.reshape(-1, 3), axis=0).shape[0]
            ),
            "orientation_attributes": orientation_attributes,
            "grain_label_candidates": grain_candidates,
            "grain_labels_available": bool(grain_candidates),
            "orientation_is_grain_mean": all(
                "grain-mean" in str(attributes.get("description", "")).lower()
                for attributes in orientation_attributes.values()
            ),
            "native_ebsd_step": {
                "metadata_pixel_size_um": orientation_attributes["phi1"].get(
                    "pixel_size_um"
                ),
                "independently_documented": False,
                "interpretation": (
                    "metadata value is present, but it is not independently "
                    "established as the native EBSD acquisition step"
                ),
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ebsd", type=Path, default=DEFAULT_EBSD)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--markdown", type=Path, default=MARKDOWN)
    args = parser.parse_args()
    if not args.ebsd.is_file():
        raise FileNotFoundError(f"EBSD source not found: {args.ebsd}")
    inspection = inspect_ebsd(args.ebsd)
    if inspection["grain_labels_available"]:
        status = "screening_not_run_pending_label_contract_review"
        verdict = "E_pending_review"
        reason = "candidate label datasets require explicit provenance review"
    else:
        status = "preflight_blocked_no_grain_labels"
        verdict = "E_data_geometry_insufficient"
        reason = (
            "CP_dataset contains orientation and Schmid fields but no grain ID "
            "or segmentation dataset; no P43 microstructural screening was run"
        )
    report = {
        "schema_version": 1,
        "status": status,
        "verdict": verdict,
        "reason": reason,
        "screening_performed": False,
        "methodological_scope": "P43 microstructure-to-k_perp screening preflight",
        "inspection": inspection,
        "required_before_screening": [
            "source grain IDs or a provenance-backed segmentation",
            "explicit label-to-orientation and boundary convention",
            "independent decision on whether the native EBSD step is known",
        ],
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = args.markdown if args.markdown.is_absolute() else ROOT / args.markdown
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(
        "# P43 microstructure-`k_perp` screening\n\n"
        "## Preflight status\n\n"
        f"**{verdict}** — {reason}.\n\n"
        "The source `CP_dataset.h5` contains per-pixel grain-mean Euler "
        "orientations and Schmid data, but no grain-ID or segmentation dataset. "
        "The M20 crop contains "
        f"{inspection['m20_unique_orientation_count']} unique orientation values, "
        "so it cannot supply a defensible grain-boundary geometry.\n\n"
        "The HDF5 metadata records `pixel_size_um = "
        f"{inspection['native_ebsd_step']['metadata_pixel_size_um']}`, but the "
        "native EBSD acquisition step is not independently documented. It must "
        "not be substituted for the DIC scale.\n\n"
        "No grain descriptors, mechanical perturbations, `k_perp` projections, "
        "figures or candidate rankings were computed. A provenance-backed grain "
        "map is required before resuming this screening.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
