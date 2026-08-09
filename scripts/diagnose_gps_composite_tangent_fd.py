"""Finite-difference the complete GPS constitutive application on M20 points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.benchmark_tri2_j2_krylov import _load_case
from scripts.diagnose_gps_tangent_localisation import (
    CROP_20X20,
    EBSD_ORIENTATION_H5,
    GPS,
    PAIRED_PARAMETER_SET,
    _checkpoint_calls,
    _run_backend,
)


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-30))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=CROP_20X20)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--library", default="build/mfront/src/libBehaviour.so")
    parser.add_argument("--ebsd-orientation-h5", type=Path, default=Path(EBSD_ORIENTATION_H5))
    parser.add_argument("--paired-parameter-set", default=PAIRED_PARAMETER_SET)
    parser.add_argument("--mfront-threads", type=int, default=4)
    parser.add_argument("--maximum-newton-iterations", type=int, default=40)
    parser.add_argument("--checkpoint-increment", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.shadow_scope = "all"

    mesh = arguments.crop_nodes[1] - arguments.crop_nodes[0]
    grid, _, yield_stress, coefficient, boundary = _load_case(mesh, arguments.crop_nodes)
    material, recording, result = _run_backend(
        GPS, arguments, grid, yield_stress, coefficient, boundary
    )
    calls = _checkpoint_calls(recording, arguments.checkpoint_increment)
    rows: list[dict[str, object]] = []
    for point in (96, 95, 59):
        selected = next(
            (
                (index, item)
                for index, item in enumerate(calls)
                if bool(np.asarray(item["substep_mask"])[point])
            ),
            (0, calls[0]),
        )
        call_index, call = selected
        diagnostic = call.get("shadow_diagnostics")
        if not isinstance(diagnostic, dict):
            raise RuntimeError("shadow telemetry is missing from the selected call")
        committed_before = int(call["committed_before"])
        snapshot = recording.committed_snapshots[committed_before - 1]
        strain = np.asarray(call["strain"], dtype=float).copy()
        dt = float(call["time_increment"])
        base_mask = np.asarray(call["substep_mask"], dtype=bool).copy()
        base_gps = np.asarray(diagnostic["gps_tangent"], dtype=float)[point]
        base_shadow = np.asarray(diagnostic["shadow_tangent"], dtype=float)[point]
        h_rows: list[dict[str, object]] = []
        for h in (1.0e-5, 1.0e-6, 1.0e-7):
            fd = np.zeros((3, 3), dtype=float)
            same_partition = True
            for column in range(3):
                plus = strain.copy()
                minus = strain.copy()
                plus[:, column] += h
                minus[:, column] -= h
                recording.restore_state(snapshot)
                trial_plus = recording.evaluate(
                    plus, time_increment=dt, consistent_tangent=True
                )
                plus_mask = np.asarray(material.last_substep_mask, dtype=bool)
                recording.restore_state(snapshot)
                trial_minus = recording.evaluate(
                    minus, time_increment=dt, consistent_tangent=True
                )
                minus_mask = np.asarray(material.last_substep_mask, dtype=bool)
                same_partition &= bool(
                    plus_mask[point] == base_mask[point]
                    and minus_mask[point] == base_mask[point]
                )
                fd[:, column] = (
                    np.asarray(trial_plus.stress_in_plane_mpa)[point]
                    - np.asarray(trial_minus.stress_in_plane_mpa)[point]
                ) / (2.0 * h)
            h_rows.append(
                {
                    "h": h,
                    "same_partition_at_point": same_partition,
                    "fd_tangent": fd.tolist(),
                    "relative_fd_to_gps": _relative(fd, base_gps),
                    "relative_fd_to_shadow": _relative(fd, base_shadow),
                }
            )
        rows.append(
            {
                "point": point,
                "call": call_index,
                "substep": bool(base_mask[point]),
                "divisions": int(np.asarray(call["substep_divisions"])[point]),
                "gps_tangent": base_gps.tolist(),
                "shadow_tangent": base_shadow.tolist(),
                "finite_difference": h_rows,
            }
        )
    payload = {
        "schema_version": 1,
        "configuration": {
            "crop_nodes": arguments.crop_nodes,
            "checkpoint_increment": arguments.checkpoint_increment,
            "h_values": [1.0e-5, 1.0e-6, 1.0e-7],
            "shadow_scope": "all",
        },
        "reference": {
            "gps_newton": int(sum(result.diagnostics.iterations_per_increment)),
        },
        "rows": rows,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for row in rows:
        print(
            f"point {row['point']} substep={row['substep']} divisions={row['divisions']} "
            + " ".join(
                f"h={item['h']:.0e}: FD/GPS={item['relative_fd_to_gps']:.3e} "
                f"FD/shadow={item['relative_fd_to_shadow']:.3e}"
                for item in row["finite_difference"]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
