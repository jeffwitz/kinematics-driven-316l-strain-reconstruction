#!/usr/bin/env python3
"""Generate the machine-readable definition-level proof for the historical table."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from fem_inhouse.material import LudwikLaw, abaqus_plastic_table

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "validation"
    / "reference_data"
    / "historical_table_definition_v1"
    / "report.json"
)


def generate(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Evaluate the versioned generator contract and write a deterministic report."""
    law = LudwikLaw(
        yield_stress_mpa=250.0,
        hardening_coefficient_mpa=1000.0,
        hardening_exponent=0.5,
    )
    table = abaqus_plastic_table(law)
    expected_stress = law.stress(table[:, 1])
    report: dict[str, object] = {
        "schema_version": 1,
        "passed": bool(
            table.shape == (1000, 2)
            and table[0, 1] == 0.0
            and table[1, 1] == 1.0e-6
            and table[-1, 1] == 0.2
            and np.array_equal(table[:, 0], expected_stress)
        ),
        "scope": "definition-level reproduction of the available Abaqus-oriented table",
        "claim_exclusions": [
            "Abaqus finite-element parity",
            "Abaqus ODB extraction parity",
        ],
        "table": {
            "rows": int(table.shape[0]),
            "columns": ["yield_stress_mpa", "equivalent_plastic_strain"],
            "first_plastic_strain": float(table[0, 1]),
            "second_plastic_strain": float(table[1, 1]),
            "last_plastic_strain": float(table[-1, 1]),
            "maximum_formula_error_mpa": float(
                np.max(np.abs(table[:, 0] - expected_stress))
            ),
        },
        "generator": {
            "qualified_name": "fem_inhouse.material.abaqus_plastic_table",
            "source_sha256": hashlib.sha256(
                inspect.getsource(abaqus_plastic_table).encode("utf-8")
            ).hexdigest(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not generate(args.output)["passed"]:
        raise SystemExit("historical table definition verification failed")


if __name__ == "__main__":
    main()
