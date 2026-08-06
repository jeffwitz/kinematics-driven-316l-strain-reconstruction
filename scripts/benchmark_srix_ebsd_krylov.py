"""Compare nonsymmetric Krylov policies on a small P43 EBSD crop."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    method: str
    linear_mode: str
    recycling: bool
    reference_update: str = "initial"


VARIANTS = (
    Variant("gmres_fixed", "gmres", "fixed", False),
    Variant("gmres_eisenstat_walker", "gmres", "eisenstat_walker", False),
    Variant("lgmres_eisenstat_walker", "lgmres", "eisenstat_walker", True),
    Variant(
        "lgmres_eisenstat_walker_no_recycling",
        "lgmres",
        "eisenstat_walker",
        False,
    ),
    Variant(
        "lgmres_eisenstat_walker_b0_per_increment",
        "lgmres",
        "eisenstat_walker",
        True,
        "per_increment",
    ),
    Variant(
        "lgmres_eisenstat_walker_b0_per_newton",
        "lgmres",
        "eisenstat_walker",
        True,
        "per_newton",
    ),
    Variant("gcrotmk_eisenstat_walker", "gcrotmk", "eisenstat_walker", True),
)


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(np.linalg.norm(reference), 1.0e-30)
    )


def _field_errors(candidate: Path, reference: Path) -> dict[str, float]:
    errors: dict[str, float] = {}
    with np.load(candidate) as candidate_fields, np.load(reference) as reference_fields:
        for name in (
            "displacement",
            "stress_in_plane_mpa",
            "reaction_forces",
            "plastic_slip",
            "equivalent_plastic_slip",
        ):
            if name in candidate_fields and name in reference_fields:
                errors[f"{name}_relative_l2_error"] = _relative_error(
                    np.asarray(candidate_fields[name]),
                    np.asarray(reference_fields[name]),
                )
    return errors


def _run_variant(
    variant: Variant,
    *,
    arguments: argparse.Namespace,
    output_directory: Path,
) -> dict[str, object]:
    report_path = output_directory / f"{variant.name}.json"
    log_path = output_directory / f"{variant.name}.log"
    command = [
        sys.executable,
        "scripts/qualify_crystal_tet2_p43.py",
        "--crop-nodes",
        *(str(value) for value in arguments.crop_nodes),
        "--increments",
        str(arguments.increments),
        "--paired-parameter-set",
        arguments.paired_parameter_set,
        "--ebsd-orientation-h5",
        str(arguments.ebsd_orientation_h5),
        "--mfront-threads",
        str(arguments.mfront_threads),
        "--maximum-newton-iterations",
        str(arguments.maximum_newton_iterations),
        "--krylov-method",
        variant.method,
        "--linear-mode",
        variant.linear_mode,
        "--reference-update",
        variant.reference_update,
        "--gmres-rtol",
        str(arguments.gmres_rtol),
        "--gmres-restart",
        str(arguments.gmres_restart),
        "--local-transverse-predictor",
        arguments.local_transverse_predictor,
        "--output",
        str(report_path),
        "--no-final-verification",
        (
            "--krylov-recycling"
            if variant.recycling
            else "--no-krylov-recycling"
        ),
    ]
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    wall_seconds = time.perf_counter() - started
    record: dict[str, object] = {
        "name": variant.name,
        "method": variant.method,
        "linear_mode": variant.linear_mode,
        "recycling": variant.recycling,
        "reference_update": variant.reference_update,
        "return_code": completed.returncode,
        "wall_seconds": wall_seconds,
        "report": str(report_path),
        "log": str(log_path),
        "command": command,
    }
    if completed.returncode == 0 and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        timings = report["timings"]
        record.update(
            {
                "status": "completed",
                "elapsed_seconds": report["elapsed_seconds"],
                "final_residual": report["final_residual"],
                "newton_iterations": report["newton_iterations"],
                "krylov_outer_callbacks": report["krylov_outer_callbacks"],
                "jacobian_matvec_calls": report["jacobian_matvec_calls"],
                "preconditioner_calls": report["preconditioner_calls"],
                "krylov_seconds": timings["gmres_seconds"],
                "jacobian_seconds": timings["jacobian_seconds"],
                "preconditioner_seconds": timings["preconditioner_seconds"],
                "krylov_overhead_seconds": timings["krylov_overhead_seconds"],
                "material_seconds": timings["material_seconds"],
                "attempt_cost_summary": report["attempt_cost_summary"],
                "field_file": report["field_file"],
            }
        )
        for status, values in report["attempt_cost_summary"].items():
            for name, value in values.items():
                record[f"{status}_{name}"] = value
    else:
        record["status"] = "failed"
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crop-nodes",
        nargs=4,
        type=int,
        default=(1610, 1630, 1075, 1095),
    )
    parser.add_argument("--increments", type=int, default=4)
    parser.add_argument("--ebsd-orientation-h5", type=Path, required=True)
    parser.add_argument(
        "--paired-parameter-set",
        default="316l_guilhem2013_nasri2018_meric_srix_rate_1e-3",
    )
    parser.add_argument("--mfront-threads", type=int, default=4)
    parser.add_argument("--maximum-newton-iterations", type=int, default=12)
    parser.add_argument("--gmres-rtol", type=float, default=1.0e-8)
    parser.add_argument("--gmres-restart", type=int, default=50)
    parser.add_argument(
        "--local-transverse-predictor",
        choices=("committed", "tangent"),
        default="tangent",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(variant.name for variant in VARIANTS),
        default=tuple(variant.name for variant in VARIANTS),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output_directory = arguments.output.with_suffix("")
    output_directory.mkdir(parents=True, exist_ok=True)
    selected = [variant for variant in VARIANTS if variant.name in arguments.variants]
    records = [
        _run_variant(variant, arguments=arguments, output_directory=output_directory)
        for variant in selected
    ]
    completed = [record for record in records if record["status"] == "completed"]
    if completed:
        reference_fields = Path(str(completed[0]["field_file"]))
        for record in completed:
            record["field_errors_to_first_completed"] = _field_errors(
                Path(str(record["field_file"])), reference_fields
            )
    payload = {
        "schema_version": 1,
        "status": "completed" if len(completed) == len(records) else "partial",
        "configuration": {
            "crop_nodes": arguments.crop_nodes,
            "increments": arguments.increments,
            "mfront_threads": arguments.mfront_threads,
            "maximum_newton_iterations": arguments.maximum_newton_iterations,
            "gmres_rtol": arguments.gmres_rtol,
            "gmres_restart": arguments.gmres_restart,
            "ebsd_orientation_h5": str(arguments.ebsd_orientation_h5),
        },
        "variants": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    csv_path = arguments.output.with_suffix(".csv")
    columns = (
        "name",
        "status",
        "elapsed_seconds",
        "newton_iterations",
        "krylov_outer_callbacks",
        "jacobian_matvec_calls",
        "preconditioner_calls",
        "krylov_seconds",
        "jacobian_seconds",
        "preconditioner_seconds",
        "krylov_overhead_seconds",
        "material_seconds",
        "final_residual",
        "accepted_attempts",
        "rejected_attempts",
        "accepted_newton_iterations",
        "rejected_newton_iterations",
        "accepted_jacobian_matvec_calls",
        "rejected_jacobian_matvec_calls",
        "accepted_preconditioner_calls",
        "rejected_preconditioner_calls",
        "accepted_krylov_seconds",
        "rejected_krylov_seconds",
        "accepted_jacobian_seconds",
        "rejected_jacobian_seconds",
        "accepted_preconditioner_seconds",
        "rejected_preconditioner_seconds",
        "accepted_krylov_overhead_seconds",
        "rejected_krylov_overhead_seconds",
        "accepted_material_seconds",
        "rejected_material_seconds",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"output": str(arguments.output), "csv": str(csv_path)}))
    return 0 if len(completed) == len(records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
