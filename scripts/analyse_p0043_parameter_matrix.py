#!/usr/bin/env python
"""Score the P43 (ell, alpha) matrix and apply the registered selection rule.

Protocol: `validation/p0043_small_parameter_matrix_preregistration.md`.

Runs only after `scripts/replay_p0043_matrix_observations.py`: no primary
indicator compares the DIC to a raw FEM field, so every point must have been
observed through the symmetric operator first.

The measurement floor `D_self` comes from the section 9 validation, which must
have been run and committed before the matrix is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fem_inhouse.validation.selection_indicators import (
    DEFECT_NAMES,
    PRINCIPAL_SCALE_PIXELS,
)
from fem_inhouse.workflows.select_p0043_parameters import (
    MatrixPoint,
    select_p0043_parameters,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0043_parameter_matrix import (  # noqa: E402
    ALPHAS,
    ARCHIVED,
    ELLS_MICROMETRES,
    _tag_alpha,
    _tag_ell,
)

OBSERVATIONS = ROOT / "validation/reference_data/p0043_matrix_observations_v1"
CONTROLS = ROOT / "validation/reference_data/observed_evm_controls_p0043_v1"
VALIDATION = ROOT / "validation/reference_data/p0043_small_parameter_matrix_v1"


def _label(alpha: float, ell: float) -> str:
    return f"{_tag_alpha(alpha)}-{_tag_ell(ell)}"


def build_points(profile: str) -> list[MatrixPoint]:
    """Every grid point, with its observation and whether it converged."""

    points: list[MatrixPoint] = []
    for ell in ELLS_MICROMETRES:
        for alpha in ALPHAS:
            label = _label(alpha, ell)
            name = f"archived-{label}" if (alpha, ell) in ARCHIVED else label
            flow = OBSERVATIONS / f"{name}_{profile}" / "observed_flow_pixels.npy"
            campaign = (
                ROOT / ARCHIVED[(alpha, ell)]
                if (alpha, ell) in ARCHIVED
                else ROOT / "results" / f"mm-id-p0043-{label}"
            )
            points.append(
                MatrixPoint(
                    label=label,
                    alpha=alpha,
                    ell_um=ell,
                    flow_path=flow,
                    converged=flow.is_file(),
                    campaign=campaign,
                )
            )
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="legacy_script_2021")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    validation_path = VALIDATION / "indicator_validation.json"
    if not validation_path.is_file():
        print(f"section 9 has not been run: {validation_path} missing", flush=True)
        return 1
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation["all_acceptance_criteria_passed"]:
        print(f"section 9 did not pass: {validation['failed_criteria']}", flush=True)
        return 1
    self_defects = validation["defects"]["repetition_residual"][str(PRINCIPAL_SCALE_PIXELS)]

    points = build_points(arguments.profile)
    missing = [point.label for point in points if not point.converged]
    print(f"{len(points)} grid points, {len(points) - len(missing)} observed", flush=True)
    if missing:
        print(f"  not yet observed or non-converged: {missing}", flush=True)
    print(f"floor D_self: { {k: round(v, 6) for k, v in self_defects.items()} }", flush=True)
    if arguments.dry_run:
        return 0

    output = Path(arguments.output or (VALIDATION / arguments.profile))
    report = select_p0043_parameters(
        prepared_case=ROOT / "data/processed/case_study",
        points=points,
        controls={
            "homogeneous": CONTROLS / "homogeneous/observed_flow_pixels.npy",
            "translated": CONTROLS / "translated/observed_flow_pixels.npy",
        },
        self_defects=self_defects,
        replicate=(
            "a2-ell40",
            OBSERVATIONS / f"a2-ell40-inc40_{arguments.profile}" / "observed_flow_pixels.npy",
        ),
        output_directory=output,
        profile=arguments.profile,
        overwrite=True,
    )

    print(f"\n{'point':16s}" + "".join(f"{n:>16s}" for n in DEFECT_NAMES) + f"{'minimax':>12s}")
    for label in sorted(report["raw_table"]):
        row = report["raw_table"][label]
        print(
            f"{label:16s}"
            + "".join(f"{row[name]:16.5g}" for name in DEFECT_NAMES)
            + f"{report['minimax'][label]:12.5g}"
        )
    print(f"\nnull D_null: { {k: round(v, 5) for k, v in report['null_defects'].items()} }")
    print(f"  taken from: {report['null_defect_source']}")
    floor = report["solver_reproducibility"]
    if floor["available"]:
        print(f"\nsolver reproducibility floor (20 vs 40 increments on {floor['twin']}):")
        print(f"  { {k: round(v, 5) for k, v in floor['floor'].items()} }")
    else:
        print(f"\nsolver reproducibility floor unavailable: {floor['reason']}")
    cutbacks = {
        label: value.get("cutbacks")
        for label, value in report["solver_diagnostics"].items()
        if value.get("available")
    }
    print(f"\ncutbacks: {cutbacks}")
    print(f"\npareto front: {report['pareto_front']}")
    print(f"zone: {report['zone']}")
    print(
        f"bootstrap: {report['bootstrap']['most_frequent']} wins "
        f"{report['bootstrap']['most_frequent_share']:.1%} -> {report['bootstrap']['verdict']}"
    )
    for pair in report["iso_achi_pairs"]:
        print(f"iso-Achi {pair['achi']:g}: {pair['members']} separation={pair['separation']}")
    print(f"\nCONCLUSION {report['conclusion']['case']}")
    print(f"  {report['conclusion']['statement']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
