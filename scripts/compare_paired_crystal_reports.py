"""Validate and compare manifests from paired Meric--SRIX reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_same(meric: dict[str, Any], srix: dict[str, Any], path: str, reasons: list[str]) -> None:
    def value(report: dict[str, Any]) -> Any:
        current: Any = report
        for part in path.split("/"):
            current = current[part]
        return current

    try:
        left, right = value(meric), value(srix)
    except (KeyError, TypeError):
        reasons.append(f"missing manifest field: {path}")
        return
    if left != right:
        reasons.append(f"mismatch at {path}: {left!r} != {right!r}")


def compare(meric: dict[str, Any], srix: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    for path in (
        "crystal_material/backbone/sha256",
        "crystal_material/backbone/slip_systems/crystal_structure",
        "crystal_material/backbone/slip_systems/family",
        "crystal_material/backbone/slip_systems/count",
        "crystal_material/backbone/interaction_matrix",
        "crystal_material/backbone/interaction_convention",
        "crystal_material/mfront_structure/structure_contract_sha256",
        "orientation/sha256",
        "crop_nodes",
        "mesh",
        "boundary_sha256",
        "units",
    ):
        _check_same(meric, srix, path, reasons)
    same_increments = meric.get("increments") == srix.get("increments")
    if not same_increments:
        reasons.append("different temporal discretizations and different flow rules")
    result = {
        "comparison_authorized": not reasons,
        "field_comparison_authorized": not reasons and same_increments,
        "performance_comparison_authorized": not reasons and same_increments,
        "reasons": reasons,
        "reports": {
            "meric": {"behaviour": meric.get("behaviour"), "increments": meric.get("increments")},
            "srix": {"behaviour": srix.get("behaviour"), "increments": srix.get("increments")},
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meric-report", type=Path, required=True)
    parser.add_argument("--srix-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(_read(args.meric_report), _read(args.srix_report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["comparison_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
