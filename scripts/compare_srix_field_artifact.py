"""Compare a SRIX field artifact with a qualified reference artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(float(np.linalg.norm(reference)), 1.0e-30)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    candidate_report = json.loads(arguments.candidate_report.read_text())
    reference_report = json.loads(arguments.reference_report.read_text())
    candidate_fields = np.load(
        arguments.candidate_report.with_suffix(".fields.npz")
    )
    reference_fields = np.load(
        arguments.reference_report.with_suffix(".fields.npz")
    )
    metrics: dict[str, dict[str, float]] = {}
    for name in sorted(set(candidate_fields.files) & set(reference_fields.files)):
        candidate = np.asarray(candidate_fields[name])
        reference = np.asarray(reference_fields[name])
        metrics[name] = {
            "relative_l2": _relative_l2(candidate, reference),
            "relative_linf": float(
                np.max(np.abs(candidate - reference))
                / max(float(np.max(np.abs(reference))), 1.0e-30)
            ),
            "maximum_absolute_difference": float(np.max(np.abs(candidate - reference))),
        }
    report = {
        "schema_version": 1,
        "candidate_report": str(arguments.candidate_report),
        "reference_report": str(arguments.reference_report),
        "candidate_report_sha256": hashlib.sha256(
            arguments.candidate_report.read_bytes()
        ).hexdigest(),
        "reference_report_sha256": hashlib.sha256(
            arguments.reference_report.read_bytes()
        ).hexdigest(),
        "candidate_execution_commit": candidate_report.get("execution_commit"),
        "reference_execution_commit": reference_report.get("execution_commit"),
        "metrics": metrics,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
