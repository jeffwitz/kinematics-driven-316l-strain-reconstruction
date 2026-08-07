"""Qualify monolithic and block-wise MFront condensation on P43 M100 EBSD."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BLOCK_SIZES: tuple[int | None, ...] = (None, 10000, 5000, 2500, 1250, 625)


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(np.linalg.norm(reference), 1.0e-30)
    )


def _field_errors(candidate: Path, reference: Path) -> dict[str, float]:
    errors: dict[str, float] = {}
    with np.load(candidate) as values, np.load(reference) as reference_values:
        for name in (
            "displacement",
            "reaction_forces",
            "stress_in_plane_mpa",
            "plastic_slip",
            "equivalent_plastic_slip",
            "accumulated_slip",
        ):
            if name in values and name in reference_values:
                errors[f"{name}_relative_l2_error"] = _relative_error(
                    np.asarray(values[name]), np.asarray(reference_values[name])
                )
    return errors


def _label(block_size: int | None) -> str:
    return "monolithic" if block_size is None else f"block_{block_size}"


def _run(
    *,
    block_size: int | None,
    arguments: argparse.Namespace,
    output_directory: Path,
) -> dict[str, object]:
    label = _label(block_size)
    report_path = output_directory / f"{label}.json"
    log_path = output_directory / f"{label}.log"
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
        "--local-transverse-predictor",
        "tangent",
        "--krylov-method",
        "lgmres",
        "--linear-mode",
        "eisenstat_walker",
        "--krylov-recycling",
        "--no-final-verification",
        "--output",
        str(report_path),
    ]
    if block_size is not None:
        command.extend(("--condensation-block-size", str(block_size)))
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
    record: dict[str, object] = {
        "label": label,
        "block_size": block_size,
        "return_code": completed.returncode,
        "wall_seconds": time.perf_counter() - started,
        "report": str(report_path),
        "log": str(log_path),
        "command": command,
    }
    if completed.returncode != 0 or not report_path.exists():
        record["status"] = "failed"
        return record
    report = json.loads(report_path.read_text(encoding="utf-8"))
    timings = report["timings"]
    record.update(
        {
            "status": "completed",
            "elapsed_seconds": report["elapsed_seconds"],
            "newton_iterations": report["newton_iterations"],
            "accepted_increments": report["accepted_increments"],
            "final_residual": report["final_residual"],
            "jacobian_matvec_calls": report["jacobian_matvec_calls"],
            "krylov_outer_callbacks": report["krylov_outer_callbacks"],
            "material_seconds": timings["material_seconds"],
            "material_integration_seconds": timings["material_integration_seconds"],
            "material_condensation_seconds": timings["material_condensation_seconds"],
            "material_point_integrations": timings["material_point_integrations"],
            "material_point_integrations_with_tangent": timings[
                "material_point_integrations_with_tangent"
            ],
            "material_point_integrations_without_tangent": timings[
                "material_point_integrations_without_tangent"
            ],
            "material_block_integration_calls": timings[
                "material_block_integration_calls"
            ],
            "material_block_count": timings["material_block_count"],
            "mgis_integrations": timings["mgis_integrations"],
            "krylov_seconds": timings["gmres_seconds"],
            "jacobian_seconds": timings["jacobian_seconds"],
            "preconditioner_seconds": timings["preconditioner_seconds"],
            "field_file": report["field_file"],
            "attempt_cost_summary": report["attempt_cost_summary"],
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crop-nodes", nargs=4, type=int, default=(1570, 1670, 1035, 1135)
    )
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--ebsd-orientation-h5", type=Path, required=True)
    parser.add_argument(
        "--paired-parameter-set",
        default="316l_guilhem2013_nasri2018_meric_srix_rate_1e-3",
    )
    parser.add_argument("--mfront-threads", type=int, default=4)
    parser.add_argument("--maximum-newton-iterations", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output_directory = arguments.output.with_suffix("")
    output_directory.mkdir(parents=True, exist_ok=True)
    records = [
        _run(
            block_size=block_size,
            arguments=arguments,
            output_directory=output_directory,
        )
        for block_size in BLOCK_SIZES
    ]
    completed = [record for record in records if record["status"] == "completed"]
    if completed:
        reference_fields = Path(str(completed[0]["field_file"]))
        for record in completed:
            record["field_errors_to_monolithic"] = _field_errors(
                Path(str(record["field_file"])), reference_fields
            )
    payload = {
        "schema_version": 1,
        "status": "completed" if len(completed) == len(records) else "partial",
        "configuration": {
            "crop_nodes": arguments.crop_nodes,
            "increments": arguments.increments,
            "ebsd_orientation_h5": str(arguments.ebsd_orientation_h5),
            "mfront_threads": arguments.mfront_threads,
            "maximum_newton_iterations": arguments.maximum_newton_iterations,
            "krylov_method": "lgmres",
            "linear_mode": "eisenstat_walker",
            "local_transverse_predictor": "tangent",
            "verify_final_state": False,
        },
        "variants": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    csv_path = arguments.output.with_suffix(".csv")
    columns = (
        "label",
        "block_size",
        "status",
        "elapsed_seconds",
        "material_seconds",
        "material_integration_seconds",
        "material_condensation_seconds",
        "material_point_integrations",
        "material_point_integrations_with_tangent",
        "material_point_integrations_without_tangent",
        "material_block_integration_calls",
        "material_block_count",
        "mgis_integrations",
        "newton_iterations",
        "jacobian_matvec_calls",
        "krylov_outer_callbacks",
        "krylov_seconds",
        "jacobian_seconds",
        "preconditioner_seconds",
        "final_residual",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"output": str(arguments.output), "csv": str(csv_path)}))
    return 0 if len(completed) == len(records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
