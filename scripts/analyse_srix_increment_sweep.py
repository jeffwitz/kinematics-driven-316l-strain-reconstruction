"""Analyse fixed-load-increment SRIX P43 reports against the 16-step reference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import cast

import numpy as np

FIELD_NAMES = (
    "displacement",
    "stress_in_plane_mpa",
    "reaction_forces",
    "accumulated_slip",
    "plastic_slip",
    "equivalent_plastic_slip",
)


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm((candidate - reference).ravel())
        / max(float(np.linalg.norm(reference.ravel())), 1.0e-30)
    )


def _relative_linf(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.max(np.abs(candidate - reference))
        / max(float(np.max(np.abs(reference))), 1.0e-30)
    )


def _s95(values: np.ndarray, threshold: float) -> list[int]:
    order = np.argsort(values)[::-1]
    cumulative = np.cumsum(values[order])
    count = int(np.searchsorted(cumulative, threshold, side="left")) + 1
    return [int(index) for index in order[:count]]


def _jaccard(left: list[int], right: list[int]) -> float:
    union = set(left) | set(right)
    return float(len(set(left) & set(right)) / len(union)) if union else 1.0


def _top_quantile_iou(left: np.ndarray, right: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.ceil(fraction * left.size)))
    left_threshold = np.partition(left.ravel(), -count)[-count]
    right_threshold = np.partition(right.ravel(), -count)[-count]
    left_mask = left >= left_threshold
    right_mask = right >= right_threshold
    union = np.count_nonzero(left_mask | right_mask)
    return float(np.count_nonzero(left_mask & right_mask) / union) if union else 1.0


def analyse_report(
    report_path: Path,
    reference_report: dict[str, object],
    reference_fields: dict[str, np.ndarray],
    *,
    cumulative_threshold: float,
    numerical_zero_tolerance: float,
) -> dict[str, object]:
    report = json.loads(report_path.read_text())
    field_path = report_path.with_suffix(".fields.npz")
    if not field_path.exists():
        return {
            "increments": report.get("increments"),
            "status": "missing_fields",
            "report": str(report_path),
        }
    with np.load(field_path) as loaded:
        fields = {name: np.asarray(loaded[name]) for name in loaded.files}
    errors: dict[str, dict[str, float]] = {}
    for name in FIELD_NAMES:
        if name not in fields or name not in reference_fields:
            continue
        errors[name] = {
            "relative_l2": _relative_l2(fields[name], reference_fields[name]),
            "relative_linf": _relative_linf(fields[name], reference_fields[name]),
            "maximum_absolute": float(np.max(np.abs(fields[name] - reference_fields[name]))),
        }

    candidate_slip = fields["equivalent_plastic_slip"]
    reference_slip = reference_fields["equivalent_plastic_slip"]
    candidate_system_integrals = candidate_slip.sum(axis=(0, 1, 2))
    reference_system_integrals = reference_slip.sum(axis=(0, 1, 2))
    candidate_fractions = candidate_system_integrals / max(
        float(candidate_system_integrals.sum()), 1.0e-30
    )
    reference_fractions = reference_system_integrals / max(
        float(reference_system_integrals.sum()), 1.0e-30
    )
    system_relative_errors = np.abs(
        candidate_system_integrals - reference_system_integrals
    ) / np.maximum(np.abs(reference_system_integrals), 1.0e-30)
    reference_integral_total = float(reference_system_integrals.sum())
    significant_systems = np.abs(reference_system_integrals) > max(
        numerical_zero_tolerance * reference_integral_total, 1.0e-30
    )
    candidate_s95 = _s95(candidate_fractions, cumulative_threshold)
    reference_s95 = _s95(reference_fractions, cumulative_threshold)
    candidate_total = candidate_slip.sum(axis=-1)
    reference_total = reference_slip.sum(axis=-1)
    return {
        "increments": int(report["increments"]),
        "status": str(report["status"]),
        "report": str(report_path),
        "field_file": str(field_path),
        "elapsed_seconds": float(report["elapsed_seconds"]),
        "material_evaluations": int(report["timings"]["material_evaluations"]),
        "material_evaluate_calls": int(report["timings"]["material_evaluate_calls"]),
        "newton_iterations": int(report["newton_iterations"]),
        "iterations_per_increment": list(report["iterations_per_increment"]),
        "krylov_iterations": int(report["krylov_iterations"]),
        "jacobian_matvec_calls": int(report["jacobian_matvec_calls"]),
        "preconditioner_calls": int(report["preconditioner_calls"]),
        "final_residual": float(report["final_residual"]),
        "timings": report["timings"],
        "errors_vs_reference_16": errors,
        "system_integral_ratio": float(
            candidate_system_integrals.sum() / max(reference_system_integrals.sum(), 1.0e-30)
        ),
        "system_integral_relative_error_max": float(np.max(system_relative_errors)),
        "system_integral_relative_error_max_significant": float(
            np.max(system_relative_errors[significant_systems])
            if np.any(significant_systems)
            else 0.0
        ),
        "system_integral_relative_error_defined_count": int(
            np.count_nonzero(significant_systems)
        ),
        "system_fraction_absolute_error_max": float(
            np.max(np.abs(candidate_fractions - reference_fractions))
        ),
        "system_fractions": candidate_fractions.tolist(),
        "reference_system_fractions": reference_fractions.tolist(),
        "s95": candidate_s95,
        "reference_s95": reference_s95,
        "s95_jaccard": _jaccard(candidate_s95, reference_s95),
        "total_accumulated_system_slip_top10_iou": _top_quantile_iou(
            candidate_total, reference_total, 0.10
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--reference-increments", type=int, default=16)
    parser.add_argument("--cumulative-threshold", type=float, default=0.95)
    parser.add_argument("--numerical-zero-tolerance", type=float, default=1.0e-12)
    arguments = parser.parse_args()
    directory = arguments.directory
    reference_report_path = directory / (
        f"crystal_tet2_srix_p43_m100_i{arguments.reference_increments}_post_22143ec.json"
    )
    reference_report = json.loads(reference_report_path.read_text())
    reference_field_path = reference_report_path.with_suffix(".fields.npz")
    with np.load(reference_field_path) as loaded:
        reference_fields = {name: np.asarray(loaded[name]) for name in loaded.files}

    rows: list[dict[str, object]] = []
    for report_path in sorted(directory.glob("crystal_tet2_srix_p43_m100_i*_post_22143ec.json")):
        rows.append(
            analyse_report(
                report_path,
                reference_report,
                reference_fields,
                cumulative_threshold=arguments.cumulative_threshold,
                numerical_zero_tolerance=arguments.numerical_zero_tolerance,
            )
        )
    rows.sort(key=lambda row: int(cast(int, row["increments"])))
    output = {
        "schema_version": 1,
        "status": "complete",
        "reference_report": str(reference_report_path),
        "reference_increments": arguments.reference_increments,
        "cumulative_dominance_threshold": arguments.cumulative_threshold,
        "numerical_zero_tolerance": arguments.numerical_zero_tolerance,
        "configuration": reference_report.get("provenance", {}),
        "results": rows,
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    csv_fields = (
        "increments",
        "status",
        "elapsed_seconds",
        "material_evaluations",
        "material_evaluate_calls",
        "newton_iterations",
        "krylov_iterations",
        "jacobian_matvec_calls",
        "preconditioner_calls",
        "final_residual",
        "displacement_relative_l2",
        "stress_relative_l2",
        "reaction_relative_l2",
        "equivalent_plastic_slip_relative_l2",
        "system_integral_relative_error_max",
        "system_integral_relative_error_max_significant",
        "system_fraction_absolute_error_max",
        "s95_jaccard",
        "total_accumulated_system_slip_top10_iou",
    )
    error_columns = {
        "displacement_relative_l2": "displacement",
        "stress_relative_l2": "stress_in_plane_mpa",
        "reaction_relative_l2": "reaction_forces",
        "equivalent_plastic_slip_relative_l2": "equivalent_plastic_slip",
    }
    with arguments.output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            csv_row = {
                key: row.get(key) for key in csv_fields if key not in error_columns
            }
            errors = cast(
                dict[str, dict[str, float]], row.get("errors_vs_reference_16", {})
            )
            for output_key, field_name in error_columns.items():
                csv_row[output_key] = errors.get(field_name, {}).get("relative_l2")
            writer.writerow(csv_row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
