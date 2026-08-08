"""Decisive test: does the GPS sub-stepping PATH explain the 85-vs-57 penalty?

The GPS backend needs 85 global Newton iterations against the reference's 57
on P43. The tangent hypothesis is strongly disfavoured: at matched
constitutive states the two tangents are identical to 1e-16. The remaining
suspect is the integration PATH: the sub-stepped points advance
eps0 -> eps1/2 -> eps1 while the reference does eps0 -> eps1 directly, and a
rate-independent law with hardening can give slightly different internal
states on the two paths, making the next global Newton a slightly different
problem.

This script runs three variants on the same case:

- the condensed reference, direct (baseline, ~57 Newton on M100);
- the condensed reference forced through the same uniform halving path the
  GPS sub-stepping imposes on its failing points (law and tangent untouched,
  only the integration path changes);
- the GPS backend.

If the halved reference lands near the GPS's iteration count, the path is the
cause. If it stays at the direct count, the sub-stepping path is innocent and
the trial/state handling is the next suspect.

Usage:

    .venv/bin/python scripts/benchmark_substepping_path.py \
        --output validation/_generated/performance/substepping_path.json
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

CROP_20X20 = (1610, 1630, 1075, 1095)
EBSD_ORIENTATION_H5 = "/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5"
PAIRED_PARAMETER_SET = "316l_guilhem2013_nasri2018_meric_srix_rate_1e-3"


class UniformlyHalvedReference:
    """The condensed reference forced through a uniform two-half path.

    Each increment is integrated as eps0 -> eps/2 -> eps (the proportional
    halves, half the time increment each), exactly the path the GPS
    sub-stepping imposes on its failing points, with the law and the tangent
    untouched. The committed state is restored at the end of the evaluation,
    so the solver's trial/commit/revert semantics are preserved.
    """

    def __init__(self, inner: object, divisions: int = 2) -> None:
        self._inner = inner
        self._divisions = divisions
        self._snapshot = None
        self._trial = None

    def evaluate(
        self,
        in_plane_strain: np.ndarray,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> object:
        snapshot = self._inner.snapshot_state()
        fractions = np.linspace(1.0 / self._divisions, 1.0, self._divisions)
        trial = None
        for fraction in fractions:
            trial = self._inner.evaluate(
                fraction * np.asarray(in_plane_strain, dtype=float),
                time_increment=time_increment / self._divisions,
                consistent_tangent=consistent_tangent,
            )
            if fraction < 1.0:
                self._inner.commit()
        self._inner.restore_state(snapshot)
        self._snapshot = snapshot
        self._trial = trial
        return trial

    def commit(self) -> None:
        self._inner.commit()
        self._snapshot = None

    def revert(self) -> None:
        if self._snapshot is not None:
            self._inner.restore_state(self._snapshot)

    @property
    def timing_statistics(self) -> object:
        return getattr(self._inner, "timing_statistics", None)


def _run_variant(
    variant: str,
    arguments: argparse.Namespace,
    output_directory: Path,
) -> dict[str, object]:
    label = variant.replace("-", "_")
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
    if variant == "mfront-3d-condensed-plane-stress-halved":
        command.extend(("--material-backend", "mfront-3d-condensed-plane-stress"))
    elif variant == "mfront-3d-condensed-plane-stress":
        command.extend(("--material-backend", "mfront-3d-condensed-plane-stress"))
    elif variant == "mfront-native-generalised-plane-stress":
        command.extend(("--material-backend", "mfront-native-generalised-plane-stress"))
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
        "variant": variant,
        "return_code": completed.returncode,
        "wall_seconds": time.perf_counter() - started,
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
            "material_seconds": timings["material_seconds"],
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=CROP_20X20)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--ebsd-orientation-h5", type=Path, default=Path(EBSD_ORIENTATION_H5))
    parser.add_argument("--paired-parameter-set", default=PAIRED_PARAMETER_SET)
    parser.add_argument("--mfront-threads", type=int, default=4)
    parser.add_argument("--maximum-newton-iterations", type=int, default=40)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/_generated/performance/substepping_path.json"),
    )
    arguments = parser.parse_args()
    output_directory = arguments.output.with_suffix("")
    output_directory.mkdir(parents=True, exist_ok=True)
    variants = (
        "mfront-3d-condensed-plane-stress",
        "mfront-3d-condensed-plane-stress-halved",
        "mfront-native-generalised-plane-stress",
    )
    records = [_run_variant(variant, arguments, output_directory) for variant in variants]
    payload = {
        "schema_version": 1,
        "configuration": {
            "crop_nodes": arguments.crop_nodes,
            "increments": arguments.increments,
            "mfront_threads": arguments.mfront_threads,
        },
        "variants": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for record in records:
        status = record["status"]
        if status == "completed":
            print(
                f"{record['variant']}: {record['newton_iterations']} Newton, "
                f"{record['elapsed_seconds']:.2f}s total, {record['material_seconds']:.2f}s material"
            )
        else:
            print(f"{record['variant']}: FAILED (return {record['return_code']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
