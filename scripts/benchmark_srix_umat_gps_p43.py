"""Benchmark the UMAT GPS backend against the condensed reference on P43 20x20.

Runs `qualify_crystal_tet2_p43.py` twice (reference and UMAT backends) on a
20x20 window at the centre of the registered P43 crop, with the qualified
solver configuration. Answers two questions:

1. does the UMAT backend complete the case (converged increments, no local
   failures);
2. how does its material time compare with the Python condensation.

Field agreement against the reference is reported from the archived fields
(displacement, stresses, slips). The two backends share the same local
equations and agree at the material-point level to about 1e-11; the field
differences come from the sub-stepping discretisation and the global
iteration path.

Usage:

    .venv/bin/python scripts/benchmark_srix_umat_gps_p43.py \
        --output validation/_generated/performance/srix_p43_20x20_umat_gps.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

#: 20x20 window at the centre of the registered P43 crop (1570..1670, 1035..1135).
CROP_20X20 = (1610, 1630, 1075, 1095)

EBSD_ORIENTATION_H5 = "/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5"
PAIRED_PARAMETER_SET = "316l_guilhem2013_nasri2018_meric_srix_rate_1e-3"

BACKENDS = ("mfront-3d-condensed-plane-stress", "mfront-native-generalised-plane-stress")


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(candidate) - np.asarray(reference))
        / max(np.linalg.norm(np.asarray(reference)), 1.0e-30)
    )


def _run(
    backend: str,
    arguments: argparse.Namespace,
    output_directory: Path,
) -> dict[str, object]:
    label = backend.replace("-", "_")
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
        "--material-backend",
        backend,
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
        "backend": backend,
        "return_code": completed.returncode,
        "wall_seconds": time.perf_counter() - started,
        "report": str(report_path),
        "log": str(log_path),
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
            "material_seconds": timings["material_seconds"],
            "material_integration_seconds": timings["material_integration_seconds"],
            "material_condensation_seconds": timings["material_condensation_seconds"],
            "material_point_integrations": timings["material_point_integrations"],
            "krylov_seconds": timings["gmres_seconds"],
            "jacobian_seconds": timings["jacobian_seconds"],
            "field_file": report["field_file"],
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=CROP_20X20)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument(
        "--ebsd-orientation-h5", type=Path, default=Path(EBSD_ORIENTATION_H5)
    )
    parser.add_argument("--paired-parameter-set", default=PAIRED_PARAMETER_SET)
    parser.add_argument("--mfront-threads", type=int, default=4)
    parser.add_argument("--maximum-newton-iterations", type=int, default=40)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/_generated/performance/srix_p43_20x20_umat_gps.json"),
    )
    arguments = parser.parse_args()
    output_directory = arguments.output.with_suffix("")
    output_directory.mkdir(parents=True, exist_ok=True)
    records = [
        _run(backend, arguments, output_directory) for backend in BACKENDS
    ]
    reference = next(r for r in records if r["backend"] == BACKENDS[0])
    candidate = next(r for r in records if r["backend"] == BACKENDS[1])
    if reference["status"] == "completed" and candidate["status"] == "completed":
        reference_fields = Path(str(reference["field_file"]))
        candidate_fields = Path(str(candidate["field_file"]))
        with np.load(reference_fields) as ref_values, np.load(candidate_fields) as can_values:
            errors: dict[str, float] = {}
            for name in (
                "displacement",
                "reaction_forces",
                "stress_in_plane_mpa",
                "plastic_slip",
                "equivalent_plastic_slip",
                "accumulated_slip",
            ):
                if name in ref_values and name in can_values:
                    errors[f"{name}_relative_l2"] = _relative_error(
                        np.asarray(can_values[name]), np.asarray(ref_values[name])
                    )
        candidate["field_errors_to_reference"] = errors
        candidate["material_speedup"] = (
            float(reference["material_seconds"]) / float(candidate["material_seconds"])
            if candidate["material_seconds"] > 0
            else None
        )
    payload = {
        "schema_version": 1,
        "status": "completed" if all(r["status"] == "completed" for r in records) else "partial",
        "configuration": {
            "crop_nodes": arguments.crop_nodes,
            "increments": arguments.increments,
            "ebsd_orientation_h5": str(arguments.ebsd_orientation_h5),
            "paired_parameter_set": arguments.paired_parameter_set,
            "mfront_threads": arguments.mfront_threads,
            "maximum_newton_iterations": arguments.maximum_newton_iterations,
        },
        "variants": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for record in records:
        status = record["status"]
        if status == "completed":
            print(
                f"{record['backend']}: completed in {record['elapsed_seconds']:.2f}s "
                f"(material {record['material_seconds']:.2f}s, "
                f"integration {record['material_integration_seconds']:.2f}s, "
                f"condensation {record['material_condensation_seconds']:.2f}s), "
                f"increments {record['accepted_increments']}, "
                f"Newton {record['newton_iterations']}"
            )
        else:
            print(f"{record['backend']}: FAILED (return {record['return_code']})")
            print(f"  log: {record['log']}")
    if candidate.get("material_speedup") is not None:
        print(f"material speedup UMAT/condensed: {candidate['material_speedup']:.2f}x")
    if "field_errors_to_reference" in candidate:
        for name, value in candidate["field_errors_to_reference"].items():
            print(f"  field {name}: {value:.3e}")
    print(f"output: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
