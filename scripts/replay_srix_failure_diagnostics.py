"""Replay isolated SRIX GPS failures with compact ``abs(dg)`` candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from fem_inhouse.core.crystal_orientation import rotations_from_euler_bunge_deg
from fem_inhouse.core.crystal_parameter_pairs import resolve_paired_crystal_parameters
from fem_inhouse.core.mfront_behaviours import MFRONT_BEHAVIOURS
from fem_inhouse.core.mfront_gps.adapter import MFrontNativeGeneralisedPlaneStressBatch


def _orientations(path: Path, crop: tuple[int, int, int, int]) -> np.ndarray:
    import h5py

    x0, x1, y0, y1 = crop
    with h5py.File(path, "r") as handle:
        angles = np.stack(
            [
                np.asarray(handle[f"orientation/{name}"][x0:x1, y0:y1], dtype=float)
                for name in ("phi1", "Phi", "phi2")
            ],
            axis=-1,
        )
    rotations = rotations_from_euler_bunge_deg(angles).reshape(-1, 3, 3)
    # The qualification mesh has two material states per EBSD pixel.
    return np.repeat(rotations, 2, axis=0)


def _make_material(
    q: np.ndarray, delta: float, zero_derivative: float = -1.0,
) -> MFrontNativeGeneralisedPlaneStressBatch:
    spec = MFRONT_BEHAVIOURS.get("fcc_forest_rubin_srix_gps")
    parameters, _ = resolve_paired_crystal_parameters(
        paired_parameter_set="316l_guilhem2013_nasri2018_meric_srix_rate_1e-3",
        law="forest_rubin_srix",
    )
    parameters["SrixSlipSmoothingDelta"] = float(delta)
    parameters["SrixSlipZeroDerivative"] = float(zero_derivative)
    return MFrontNativeGeneralisedPlaneStressBatch(
        os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", "build/mfront/src/libBehaviour.so"),
        behaviour_spec=spec,
        point_count=1,
        rotation_global_to_material=q[None, :, :],
        thread_count=1,
        behaviour_name=spec.behaviour_name("condensed_3d"),
        behaviour_parameters=parameters,
        backend_label="failure-replay",
    )


def replay(
    input_path: Path,
    orientations: np.ndarray,
    deltas: list[float],
    zero_derivative: float = -1.0,
) -> dict[str, object]:
    data = np.load(input_path)
    results: dict[str, object] = {"input": str(input_path), "candidates": {}}
    points = np.asarray(data["point"], dtype=int)
    for delta in deltas:
        statuses: list[int] = []
        for row, point in enumerate(points):
            material = _make_material(orientations[point], delta, zero_derivative)
            material._manager.s0.gradients[0, :] = data["s0_gradient"][row]
            material._manager.s0.thermodynamic_forces[0, :] = data[
                "s0_thermodynamic_forces"
            ][row]
            material._manager.s0.internal_state_variables[0, :] = data[
                "s0_internal_state_variables"
            ][row]
            status = material._integrate_once(
                np.asarray(data["target_in_plane_kelvin"][row], dtype=float)[None, :],
                float(data["time_increment"][row]),
                None,
                (0, 1),
            )
            statuses.append(int(status))
        statuses_array = np.asarray(statuses, dtype=int)
        results["candidates"][str(delta)] = {
            "records": int(len(statuses)),
            "failed_trials": int(np.sum(statuses_array != 1)),
            "rescued_trials": int(np.sum(statuses_array == 1)),
            "status_histogram": {
                str(value): int(np.sum(statuses_array == value))
                for value in np.unique(statuses_array)
            },
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--orientation-h5", type=Path, required=True)
    parser.add_argument("--crop-nodes", nargs=4, type=int, required=True)
    parser.add_argument("--delta", type=float, action="append", required=True)
    parser.add_argument("--zero-derivative", type=float, default=-1.0, choices=(-1.0, 0.0, 1.0))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(
        args.input,
        _orientations(args.orientation_h5, tuple(args.crop_nodes)),
        args.delta,
        args.zero_derivative,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
