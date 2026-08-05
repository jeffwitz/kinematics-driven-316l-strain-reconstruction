"""Bounded temporal-refinement diagnostic for Meric-Cailletaud on P43."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subincrements", nargs="+", type=int, default=[16])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--mfront-threads", type=int, default=4)
    arguments = parser.parse_args()
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
    records: list[dict[str, object]] = []
    environment_keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "FFTW_NUM_THREADS",
    )

    def write_report() -> None:
        report = {
            "status": "running_meric_p43_time_refinement_diagnostic",
            "duration_normalization": "total pseudo-time is one; dt=1/increments",
            "subincrements": arguments.subincrements,
            "mfront_threads": arguments.mfront_threads,
            "environment": {key: environment[key] for key in environment_keys},
            "records": records,
            "interpretation": "Diagnostic only; not a production-law qualification.",
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    with tempfile.TemporaryDirectory(prefix="meric-p43-time-") as temporary:
        temporary_path = Path(temporary)
        for increments in arguments.subincrements:
            output = temporary_path / f"meric_{increments}.json"
            command = [
                sys.executable,
                str(script),
                "--crop-nodes",
                "1570",
                "1670",
                "1035",
                "1135",
                "--increments",
                str(increments),
                "--tolerance",
                "1e-8",
                "--behaviour",
                "fcc_meric_cailletaud",
                "--mfront-threads",
                str(arguments.mfront_threads),
                "--krylov-method",
                "lgmres",
                "--linear-mode",
                "eisenstat_walker",
                "--reference-update",
                "initial",
                "--output",
                str(output),
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=arguments.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                records.append(
                    {
                        "increments": increments,
                        "status": "timeout",
                        "timeout_seconds": arguments.timeout_seconds,
                    }
                )
                write_report()
                continue
            if completed.returncode == 0 and output.exists():
                record = json.loads(output.read_text())
                record["field_file"] = None
                records.append(record)
            else:
                records.append(
                    {
                        "increments": increments,
                        "status": "failed",
                        "returncode": completed.returncode,
                        "stderr_tail": completed.stderr[-2000:],
                    }
                )
            write_report()
    report = {
        "status": "completed_meric_p43_time_refinement_diagnostic",
        "duration_normalization": "total pseudo-time is one; dt=1/increments",
        "subincrements": arguments.subincrements,
        "mfront_threads": arguments.mfront_threads,
        "environment": {
            key: environment[key]
            for key in environment_keys
        },
        "records": records,
        "interpretation": "Diagnostic only; not a production-law qualification.",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
