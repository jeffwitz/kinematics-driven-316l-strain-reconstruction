"""Compare GPS baseline and selective composite-FD tangent on M20."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.benchmark_tri2_j2_krylov import _load_case
from scripts.diagnose_gps_tangent_localisation import GPS, _run_backend


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _run(
    args: SimpleNamespace,
    grid: object,
    ys: np.ndarray,
    coeff: np.ndarray,
    boundary: np.ndarray,
    fd: bool,
):
    args.composite_fd_tangent = fd
    args.shadow_scope = None
    return _run_backend(GPS, args, grid, ys, coeff, boundary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default="build/mfront/src/libBehaviour.so")
    parser.add_argument(
        "--ebsd-orientation-h5",
        type=Path,
        default=Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=(1610, 1630, 1075, 1095))
    parser.add_argument("--increments", type=int, default=8)
    parsed = parser.parse_args()
    crop = tuple(parsed.crop_nodes)
    base = SimpleNamespace(
        crop_nodes=crop,
        increments=parsed.increments,
        library=parsed.library,
        ebsd_orientation_h5=parsed.ebsd_orientation_h5,
        paired_parameter_set="316l_guilhem2013_nasri2018_meric_srix_rate_1e-3",
        mfront_threads=4,
        maximum_newton_iterations=40,
        checkpoint_increment=6,
    )
    mesh = crop[1] - crop[0]
    grid, _, ys, coeff, boundary = _load_case(mesh, crop)
    runs = {}
    for name, fd in (("gps", False), ("composite_fd", True)):
        started = time.perf_counter()
        material, _, result = _run(base, grid, ys, coeff, boundary, fd)
        elapsed = time.perf_counter() - started
        runs[name] = {
            "elapsed_seconds": elapsed,
            "newton": int(sum(result.diagnostics.iterations_per_increment)),
            "per_increment": list(result.diagnostics.iterations_per_increment),
            "residual": float(result.diagnostics.relative_residual_history[-1]),
            "displacement_sha256": _digest(np.asarray(result.displacement)),
            "displacement": np.asarray(result.displacement).copy(),
            "stress": np.asarray(result.stress_in_plane_mpa).copy(),
            "reactions": np.asarray(result.reaction_forces).copy(),
            "observables": {
                name: np.asarray(value).copy()
                for name, value in result.observables.items()
            },
            "timing": material.timing_statistics,
        }
    gps = runs["gps"]
    fd = runs["composite_fd"]
    summary = {
        "gps_newton": gps["newton"],
        "composite_fd_newton": fd["newton"],
        "gps_elapsed_seconds": gps["elapsed_seconds"],
        "composite_fd_elapsed_seconds": fd["elapsed_seconds"],
        "composite_fd_speed_ratio": float(
            fd["elapsed_seconds"] / max(gps["elapsed_seconds"], 1.0e-30)
        ),
        "gps_per_increment": gps["per_increment"],
        "composite_fd_per_increment": fd["per_increment"],
        "gps_residual": gps["residual"],
        "composite_fd_residual": fd["residual"],
        "displacement_relative_l2": float(
            np.linalg.norm(fd["displacement"] - gps["displacement"])
            / max(np.linalg.norm(gps["displacement"]), 1.0e-30)
        ),
        "stress_relative_l2": float(
            np.linalg.norm(fd["stress"] - gps["stress"])
            / max(np.linalg.norm(gps["stress"]), 1.0e-30)
        ),
        "reaction_relative_l2": float(
            np.linalg.norm(fd["reactions"] - gps["reactions"])
            / max(np.linalg.norm(gps["reactions"]), 1.0e-30)
        ),
        "observable_relative_l2": {
            name: float(
                np.linalg.norm(fd["observables"][name] - gps["observables"][name])
                / max(np.linalg.norm(gps["observables"][name]), 1.0e-30)
            )
            for name in gps["observables"]
            if name in fd["observables"]
        },
        "gps_displacement_sha256": gps["displacement_sha256"],
        "composite_fd_displacement_sha256": fd["displacement_sha256"],
        "composite_fd_points": fd["timing"].composite_fd_points,
        "composite_fd_trajectories": fd["timing"].composite_fd_trajectories,
        "composite_fd_seconds": fd["timing"].composite_fd_seconds,
        "composite_fd_partition_changes": fd["timing"].composite_fd_partition_changes,
        "composite_fd_mgis_calls": fd["timing"].composite_fd_mgis_calls,
        "composite_fd_actual_point_integrations": fd[
            "timing"
        ].composite_fd_actual_point_integrations,
        "composite_fd_snapshot_seconds": fd["timing"].composite_fd_snapshot_seconds,
        "composite_fd_restore_seconds": fd["timing"].composite_fd_restore_seconds,
        "composite_fd_integration_seconds": fd["timing"].composite_fd_integration_seconds,
        "composite_fd_other_seconds": fd["timing"].composite_fd_other_seconds,
        "crop_nodes": crop,
        "increments": parsed.increments,
    }
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
