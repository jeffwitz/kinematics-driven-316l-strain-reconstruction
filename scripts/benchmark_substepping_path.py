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

    Each increment is integrated from the committed state through the
    mid-point: eps0 -> eps0 + (eps1 - eps0)/2 -> eps1, half the time
    increment each -- exactly the path the GPS sub-stepping imposes on its
    failing points, with the law and the tangent untouched.

    The transaction is explicit: the sub-stepped final state is captured and
    re-instated on commit, so the solver's commit() really advances the
    committed state to the end of the sub-stepped path, and revert() returns
    to the state before the evaluation.

    The wrapper keeps its own committed-total bookkeeping in the convention of
    its interface -- engineering ``[e11, e22, gamma12]`` in the global frame --
    and never reads it from the manager: ``s0.gradients`` is Kelvin and, with
    EBSD rotations, in the crystal frame, so mixing it with the engineering
    input would corrupt the interpolation (shear scaled by sqrt(2), and every
    component rotated) from the second increment on.
    """

    def __init__(self, inner: object, divisions: int = 2) -> None:
        self._inner = inner
        self._divisions = divisions
        self._snapshot = None
        self._final_state = None
        self._trial = None
        self._committed_in_plane_engineering = np.zeros((inner.point_count, 3), dtype=float)
        self._last_requested_engineering: np.ndarray | None = None

    def _manager(self) -> object | None:
        inner = self._inner
        manager = getattr(inner, "_manager", None)
        if manager is None:
            bridge = getattr(inner, "_bridge", None)
            manager = getattr(bridge, "_manager", None)
        return manager

    def _capture_s1(self) -> tuple[np.ndarray, ...]:
        manager = self._manager()
        return (
            np.asarray(manager.s1.gradients).copy(),
            np.asarray(manager.s1.internal_state_variables).copy(),
            np.asarray(manager.s1.thermodynamic_forces).copy(),
        )

    def _restore_s1(self, state: tuple[np.ndarray, ...]) -> None:
        manager = self._manager()
        manager.s1.gradients[:, :] = state[0]
        manager.s1.internal_state_variables[:, :] = state[1]
        manager.s1.thermodynamic_forces[:, :] = state[2]

    def evaluate(
        self,
        in_plane_strain: np.ndarray,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> object:
        eps1 = np.asarray(in_plane_strain, dtype=float)
        eps0 = self._committed_in_plane_engineering
        self._last_requested_engineering = eps1.copy()
        snapshot = self._inner.snapshot_state()
        trial = None
        for step in range(1, self._divisions + 1):
            alpha = step / self._divisions
            eps_step = eps0 + alpha * (eps1 - eps0)
            trial = self._inner.evaluate(
                eps_step,
                time_increment=time_increment / self._divisions,
                consistent_tangent=consistent_tangent,
            )
            if step < self._divisions:
                self._inner.commit()
        # Capture the sub-stepped final state, restore the committed one so
        # revert() works, and re-instate the final state on commit().
        self._final_state = self._capture_s1()
        self._inner.restore_state(snapshot)
        self._snapshot = snapshot
        self._trial = trial
        return trial

    def commit(self) -> None:
        if self._final_state is not None:
            self._restore_s1(self._final_state)
        self._inner.commit()
        if self._last_requested_engineering is not None:
            self._committed_in_plane_engineering = self._last_requested_engineering.copy()
        self._snapshot = None
        self._final_state = None
        self._last_requested_engineering = None

    def revert(self) -> None:
        if self._snapshot is not None:
            self._inner.restore_state(self._snapshot)
        self._final_state = None
        self._last_requested_engineering = None

    @property
    def point_count(self) -> int:
        return self._inner.point_count

    @property
    def timing_statistics(self) -> object:
        return getattr(self._inner, "timing_statistics", None)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


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
    if variant == "mfront-native-generalised-plane-stress":
        command.extend(("--material-backend", "mfront-native-generalised-plane-stress"))
    elif variant.startswith("mfront-3d-condensed-plane-stress"):
        command.extend(("--material-backend", "mfront-3d-condensed-plane-stress"))
        if "-halved" in variant:
            command.extend(("--material-backend", "mfront-3d-condensed-plane-stress-halved"))
        tolerance = arguments.closure_tolerances.get(variant)
        if tolerance is not None:
            command.extend(("--local-closure-tolerance", str(tolerance)))
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
    arguments.closure_tolerances = {
        "mfront-3d-condensed-plane-stress-tol-1e-10": 1.0e-10,
        "mfront-3d-condensed-plane-stress-tol-1e-12": 1.0e-12,
    }
    output_directory = arguments.output.with_suffix("")
    output_directory.mkdir(parents=True, exist_ok=True)
    variants = (
        "mfront-3d-condensed-plane-stress",
        "mfront-3d-condensed-plane-stress-tol-1e-10",
        "mfront-3d-condensed-plane-stress-tol-1e-12",
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
                f"{record['elapsed_seconds']:.2f}s total, "
                f"{record['material_seconds']:.2f}s material"
            )
        else:
            print(f"{record['variant']}: FAILED (return {record['return_code']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
