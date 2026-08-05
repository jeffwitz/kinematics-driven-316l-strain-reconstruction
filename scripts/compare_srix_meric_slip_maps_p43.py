"""Quantify and plot the paired SRIX/Méric P43 slip-system comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from fem_inhouse.core.fcc_interaction_matrix import SLIP_SYSTEMS
from fem_inhouse.validation.crystal_slip_figures import generate_comparison_figures
from fem_inhouse.validation.crystal_slip_metrics import SlipMetricConfig, compare_slip_fields


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _label(index: int) -> str:
    burgers, normal = SLIP_SYSTEMS[index]
    return f"{index + 1:02d}  b[{','.join(map(str, burgers))}] n[{','.join(map(str, normal))}]"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolve_field_file(report_path: Path, report: dict[str, Any]) -> Path:
    raw = Path(str(report["field_file"]))
    candidates = (raw, report_path.parent / raw, Path.cwd() / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"field archive not found for {report_path}: {raw}")


def _load_fields(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        required = {"plastic_slip", "equivalent_plastic_slip"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{path}: missing fields: {', '.join(missing)}")
        signed = np.asarray(archive["plastic_slip"], dtype=np.float64)
        equivalent = np.asarray(archive["equivalent_plastic_slip"], dtype=np.float64)
    expected = (100, 100, 2, 12)
    if signed.shape != expected or equivalent.shape != expected:
        raise ValueError(f"{path}: expected fields with shape {expected}")
    signed_by_system = np.moveaxis(signed, -1, 0).mean(axis=3)
    equivalent_by_system = np.moveaxis(equivalent, -1, 0).mean(axis=3)
    return equivalent_by_system, signed_by_system


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _source_summary(report_path: Path, report: dict[str, Any], field_path: Path) -> dict[str, Any]:
    return {
        "report": str(report_path),
        "report_sha256": _hash_file(report_path),
        "field_file": str(field_path),
        "field_sha256": _hash_file(field_path),
        "execution_commit": report.get("execution_commit"),
        "archive_commit": report.get("archive_commit"),
        "behaviour": report.get("behaviour"),
    }


def _validate_pair(meric: dict[str, Any], srix: dict[str, Any]) -> None:
    paths = (
        "crystal_material/backbone/sha256",
        "crystal_material/backbone/slip_systems/crystal_structure",
        "crystal_material/backbone/slip_systems/family",
        "crystal_material/backbone/slip_systems/count",
        "crystal_material/backbone/interaction_matrix",
        "crystal_material/mfront_structure/structure_contract_sha256",
        "orientation/sha256",
        "crop_nodes",
        "mesh",
        "boundary_sha256",
        "units",
    )
    for path in paths:
        left: Any = meric
        right: Any = srix
        for part in path.split("/"):
            if (
                not isinstance(left, dict)
                or not isinstance(right, dict)
                or part not in left
                or part not in right
            ):
                raise ValueError(f"paired reports are missing {path}")
            left, right = left[part], right[part]
        if left != right:
            raise ValueError(f"paired reports mismatch at {path}: {left!r} != {right!r}")
    if meric.get("increments") != srix.get("increments"):
        raise ValueError("different increment counts are not comparable")


def _write_system_csv(path: Path, summary: dict[str, Any], labels: list[str]) -> None:
    rows = []
    for item in summary["systems"]:
        spatial = item["spatial"]
        normalized = spatial.get("normalized") or {}
        absolute = spatial.get("absolute") or {}
        rows.append(
            {
                "system": int(item["system"]) + 1,
                "label": labels[int(item["system"])],
                "meric_total": item["meric_total"],
                "srix_total": item["srix_total"],
                "meric_fraction": item["meric_fraction"],
                "srix_fraction": item["srix_fraction"],
                "meric_rank": item["meric_rank"],
                "srix_rank": item["srix_rank"],
                "fraction_difference": item["fraction_difference"],
                "amplitude_ratio_meric_over_srix": item["amplitude_ratio_meric_over_srix"],
                "status": spatial["status"],
                "absolute_l1": absolute.get("l1"),
                "absolute_relative_l2": absolute.get("relative_l2"),
                "absolute_pearson": absolute.get("pearson"),
                "normalized_l1": normalized.get("l1"),
                "normalized_cosine": normalized.get("cosine"),
                "normalized_pearson": normalized.get("pearson"),
                "normalized_spearman": normalized.get("spearman"),
                "barycentre_distance": normalized.get("barycentre_distance"),
            }
        )
    _write_csv(path, rows, list(rows[0].keys()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--srix-report", type=Path, required=True)
    parser.add_argument("--meric-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--docs-assets-dir", type=Path, default=None)
    parser.add_argument("--dominant-threshold", type=float, default=0.05)
    parser.add_argument("--cumulative-threshold", type=float, default=0.95)
    parser.add_argument("--numerical-zero-tolerance", type=float, default=1.0e-12)
    arguments = parser.parse_args()

    srix_report = json.loads(arguments.srix_report.read_text(encoding="utf-8"))
    meric_report = json.loads(arguments.meric_report.read_text(encoding="utf-8"))
    _validate_pair(meric_report, srix_report)
    srix_fields = _resolve_field_file(arguments.srix_report, srix_report)
    meric_fields = _resolve_field_file(arguments.meric_report, meric_report)
    srix_equivalent, srix_signed = _load_fields(srix_fields)
    meric_equivalent, meric_signed = _load_fields(meric_fields)
    config = SlipMetricConfig(
        dominant_fraction_threshold=arguments.dominant_threshold,
        cumulative_dominance_threshold=arguments.cumulative_threshold,
        numerical_zero_tolerance=arguments.numerical_zero_tolerance,
    )
    summary = compare_slip_fields(
        meric_equivalent,
        srix_equivalent,
        meric_signed=meric_signed,
        srix_signed=srix_signed,
        config=config,
    )
    labels = [_label(index) for index in range(12)]
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = generate_comparison_figures(
        meric_equivalent,
        srix_equivalent,
        summary=summary,
        labels=labels,
        output_dir=arguments.output_dir,
    )
    if arguments.docs_assets_dir is not None:
        arguments.docs_assets_dir.mkdir(parents=True, exist_ok=True)
        for path in figure_paths:
            shutil.copy2(path, arguments.docs_assets_dir / path.name)
    systems = summary.pop("systems")
    summary["provenance"] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "schema_version": 1,
        "srix": _source_summary(arguments.srix_report, srix_report, srix_fields),
        "meric": _source_summary(arguments.meric_report, meric_report, meric_fields),
        "paired_parameter_set": srix_report["crystal_material"]["paired_parameter_set"],
        "increments": srix_report["increments"],
        "mesh": srix_report["mesh"],
        "crop_nodes": srix_report["crop_nodes"],
        "orientation": srix_report["orientation"],
        "slip_system_order": labels,
        "pixel_mean_of_two_tri2_states": True,
    }
    summary["status"] = "complete_final_fields_history_unavailable"
    summary["figure_files"] = [path.name for path in figure_paths]
    summary["field_comparison_authorized"] = True
    summary["performance_comparison_authorized"] = False
    summary["limitations"].append(
        "The two laws are compared on the same registered path; the output does not "
        "authorize a performance comparison."
    )
    summary["system_distribution"]["top_system_meric"] = summary["system_distribution"]["meric"][
        "top_system"
    ]
    summary["system_distribution"]["top_system_srix"] = summary["system_distribution"]["srix"][
        "top_system"
    ]
    summary["system_distribution"]["s95_meric"] = summary["system_distribution"]["meric"]["s95"]
    summary["system_distribution"]["s95_srix"] = summary["system_distribution"]["srix"]["s95"]
    summary["system_distribution"]["s5_meric"] = summary["system_distribution"]["meric"]["s5"]
    summary["system_distribution"]["s5_srix"] = summary["system_distribution"]["srix"]["s5"]
    summary["system_distribution"]["s95_reciprocal_recall"] = {
        "meric_recalled_by_srix": len(
            set(summary["system_distribution"]["s95_intersection"])
            & set(summary["system_distribution"]["meric"]["s95"])
        )
        / max(len(summary["system_distribution"]["meric"]["s95"]), 1),
        "srix_recalled_by_meric": len(
            set(summary["system_distribution"]["s95_intersection"])
            & set(summary["system_distribution"]["srix"]["s95"])
        )
        / max(len(summary["system_distribution"]["srix"]["s95"]), 1),
    }
    (arguments.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": 1,
        "status": summary["status"],
        "comparison_summary": "comparison_summary.json",
        "generated_at_utc": summary["provenance"]["generated_at_utc"],
        "git_sha": summary["provenance"]["git_sha"],
        "source_reports": {
            "meric": summary["provenance"]["meric"],
            "srix": summary["provenance"]["srix"],
        },
        "paired_parameter_set": summary["provenance"]["paired_parameter_set"],
        "mesh": summary["provenance"]["mesh"],
        "crop_nodes": summary["provenance"]["crop_nodes"],
        "orientation": summary["provenance"]["orientation"],
        "increments": summary["provenance"]["increments"],
        "slip_system_order": labels,
        "configuration": summary["configuration"],
        "field_comparison_authorized": summary["field_comparison_authorized"],
        "performance_comparison_authorized": summary["performance_comparison_authorized"],
        "incremental_history_available": False,
        "figure_files": summary["figure_files"],
    }
    (arguments.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_system_csv(arguments.output_dir / "system_metrics.csv", {"systems": systems}, labels)
    _write_csv(
        arguments.output_dir / "spatial_metrics.csv",
        [
            {
                "system": int(item["system"]) + 1,
                "status": item["status"],
                **(item.get("absolute") or {}),
                **{
                    f"normalized_{key}": value
                    for key, value in (item.get("normalized") or {}).items()
                    if key not in {"meric_barycentre_xy", "srix_barycentre_xy"}
                },
            }
            for item in summary["spatial_similarity"]["per_system"]
        ],
        [
            "system",
            "status",
            "l1",
            "l2",
            "relative_l2",
            "maximum_absolute_difference",
            "integral_ratio_meric_over_srix",
            "pearson",
            "spearman",
            "normalized_l1",
            "normalized_l2",
            "normalized_cosine",
            "normalized_pearson",
            "normalized_spearman",
            "normalized_barycentre_distance",
        ],
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
