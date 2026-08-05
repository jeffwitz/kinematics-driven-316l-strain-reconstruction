"""Repeat the registered crystal P43 run under a fixed execution environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean, median

from fem_inhouse.core.crystal_parameter_pairs import PAIRED_PARAMETER_SET


def _git_head() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            or None
        )
    except OSError:
        return None


def _stats(values: list[float]) -> dict[str, float]:
    center = median(values)
    return {
        "minimum": min(values),
        "median": center,
        "mean": mean(values),
        "maximum": max(values),
        "mad": median([abs(value - center) for value in values]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=[1570, 1670, 1035, 1135])
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--behaviour", default="fcc_forest_rubin_srix")
    parser.add_argument("--mfront-threads", type=int, default=4)
    parser.add_argument("--krylov-method", default="lgmres")
    parser.add_argument("--linear-mode", default="eisenstat_walker")
    parser.add_argument("--reference-update", default="initial")
    arguments = parser.parse_args()
    if arguments.repeats < 2:
        raise SystemExit("--repeats must be at least 2")

    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "FFTW_NUM_THREADS": "1",
        }
    )
    script = Path(__file__).with_name("qualify_crystal_tet2_p43.py")
    runs: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="srix-p43-repeats-") as temporary:
        temporary_path = Path(temporary)
        for index in range(arguments.repeats):
            child_output = temporary_path / f"run_{index + 1}.json"
            command = [
                sys.executable,
                str(script),
                "--crop-nodes",
                *(str(value) for value in arguments.crop_nodes),
                "--increments",
                str(arguments.increments),
                "--tolerance",
                str(arguments.tolerance),
                "--behaviour",
                arguments.behaviour,
                "--paired-parameter-set",
                PAIRED_PARAMETER_SET,
                "--mfront-threads",
                str(arguments.mfront_threads),
                "--krylov-method",
                arguments.krylov_method,
                "--linear-mode",
                arguments.linear_mode,
                "--reference-update",
                arguments.reference_update,
                "--output",
                str(child_output),
            ]
            completed = subprocess.run(
                command,
                check=False,
                env=environment,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0 or not child_output.exists():
                print(completed.stdout)
                print(completed.stderr, file=sys.stderr)
                raise SystemExit(completed.returncode or 1)
            run = json.loads(child_output.read_text())
            run["field_file"] = None
            runs.append(run)

    timing_keys = (
        "elapsed_seconds",
        "material_seconds",
        "material_condensation_seconds",
        "material_integration_seconds",
        "gmres_seconds",
        "jacobian_seconds",
        "preconditioner_seconds",
    )
    timing_statistics = {
        key: _stats(
            [
                float(run["elapsed_seconds"])
                if key == "elapsed_seconds"
                else float(run["timings"][key])  # type: ignore[index]
                for run in runs
            ]
        )
        for key in timing_keys
    }
    field_hashes = [run["field_sha256"] for run in runs]
    environment_keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "FFTW_NUM_THREADS",
    )
    report = {
        "status": "completed_repeated_crystal_tet2_p43",
        "archive_commit": _git_head(),
        "execution_commits": sorted({run["execution_commit"] for run in runs}),
        "repeats": arguments.repeats,
        "environment": {
            key: environment[key]
            for key in environment_keys
        },
        "configuration": {
            "crop_nodes": arguments.crop_nodes,
            "increments": arguments.increments,
            "tolerance": arguments.tolerance,
            "behaviour": arguments.behaviour,
            "mfront_threads": arguments.mfront_threads,
            "krylov_method": arguments.krylov_method,
            "linear_mode": arguments.linear_mode,
            "reference_update": arguments.reference_update,
        },
        "timing_statistics": timing_statistics,
        "runs": runs,
        "field_sha256": field_hashes,
        "fields_identical": len(
            {json.dumps(value, sort_keys=True) for value in field_hashes}
        )
        == 1,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
