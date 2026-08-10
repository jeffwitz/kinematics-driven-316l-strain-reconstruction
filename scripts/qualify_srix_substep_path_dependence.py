"""Separate the subgradient choice from the sub-step partition it removes.

Part 1 of ``validation/srix_semismooth_subgradient_preregistration.md`` showed
that the historical and zero-subgradient conventions seek the same root. The
M200 campaign nevertheless differs by ``~1e-4`` on the slip fields, and the
remaining candidate is that the two runs compose the same constitutive
evolution over different discrete partitions: the baseline sub-steps 978
points, the zero-subgradient run sub-steps none.

This script reproduces the bridge's own sub-step sequence explicitly on the 380
archived states and compares three integrations of the same increment:

``A`` historical, sub-stepped as ``_substep_span`` would;
``B`` zero-subgradient, forced onto exactly the divisions ``A`` needed;
``C`` zero-subgradient, one shot.

``A`` against ``B`` isolates the convention at fixed partition. ``B`` against
``C`` isolates the partition at fixed convention.
"""

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

#: Kelvin indices of the in-plane components, matching the GPS adapter.
IN_PLANE = (0, 1, 3)

#: Same cap as the production bridge.
MAXIMUM_SUBSTEPS = 1024


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
    # The qualification mesh carries two material states per EBSD pixel.
    return np.repeat(rotations, 2, axis=0)


def _make_material(
    q: np.ndarray, zero_derivative: float
) -> MFrontNativeGeneralisedPlaneStressBatch:
    spec = MFRONT_BEHAVIOURS.get("fcc_forest_rubin_srix_gps")
    parameters, _ = resolve_paired_crystal_parameters(
        paired_parameter_set="316l_guilhem2013_nasri2018_meric_srix_rate_1e-3",
        law="forest_rubin_srix",
    )
    parameters["SrixSlipSmoothingDelta"] = 0.0
    parameters["SrixSlipZeroDerivative"] = float(zero_derivative)
    return MFrontNativeGeneralisedPlaneStressBatch(
        os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", "build/mfront/src/libBehaviour.so"),
        behaviour_spec=spec,
        point_count=1,
        rotation_global_to_material=q[None, :, :],
        thread_count=1,
        behaviour_name=spec.behaviour_name("condensed_3d"),
        behaviour_parameters=parameters,
        backend_label="substep-path",
    )


def _seed(material: MFrontNativeGeneralisedPlaneStressBatch, data, row: int) -> None:
    material._manager.s0.gradients[0, :] = data["s0_gradient"][row]
    material._manager.s0.thermodynamic_forces[0, :] = data["s0_thermodynamic_forces"][row]
    material._manager.s0.internal_state_variables[0, :] = data[
        "s0_internal_state_variables"
    ][row]


def _run_partition(
    material: MFrontNativeGeneralisedPlaneStressBatch,
    data,
    row: int,
    origin: np.ndarray,
    target: np.ndarray,
    divisions: int,
) -> int:
    """Integrate the increment in ``divisions`` equal stages, advancing s0.

    The interpolation is on the TOTAL in-plane strain, exactly as
    ``_substep_span`` does it: a fraction of a total strain is not a strain
    increment, and getting that wrong is what the bridge comment warns about.
    """

    _seed(material, data, row)
    step = float(data["time_increment"][row]) / divisions
    for index in range(divisions):
        weight = (index + 1) / divisions
        stage = origin + weight * (target - origin)
        status = material._integrate_once(stage[None, :], step, None, (0, 1))
        if status != 1:
            return status
        if index < divisions - 1:
            material._advance_span((0, 1))
    return 1


def _state(material: MFrontNativeGeneralisedPlaneStressBatch) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(material._manager.s1.thermodynamic_forces[0, :], dtype=float).copy(),
        np.asarray(
            material._manager.s1.internal_state_variables[0, :], dtype=float
        ).copy(),
    )


def _relative(candidate: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.linalg.norm(reference))
    difference = float(np.linalg.norm(candidate - reference))
    return difference / scale if scale > 0.0 else difference


def qualify(input_path: Path, orientations: np.ndarray) -> dict[str, object]:
    data = np.load(input_path)
    points = np.asarray(data["point"], dtype=int)
    s0_gradient = np.asarray(data["s0_gradient"], dtype=float)
    full_target = np.asarray(data["target_in_plane_kelvin"], dtype=float)

    names: dict[str, slice] = {}
    rows: list[dict[str, object]] = []

    for row, point in enumerate(points):
        origin = s0_gradient[row, list(IN_PLANE)]
        target = full_target[row]

        historical = _make_material(orientations[point], -1.0)
        if not names:
            names.update(historical._observable_slices)

        # A -- historical, sub-stepped exactly as the bridge would.
        divisions = 2
        status = -1
        while divisions <= MAXIMUM_SUBSTEPS:
            status = _run_partition(historical, data, row, origin, target, divisions)
            if status == 1:
                break
            divisions *= 2
        if status != 1:
            rows.append({"row": row, "point": int(point), "historical_substepped": False})
            continue
        stress_a, internal_a = _state(historical)

        # B -- zero subgradient on exactly that partition.
        forced = _make_material(orientations[point], 0.0)
        status_b = _run_partition(forced, data, row, origin, target, divisions)
        # C -- zero subgradient, one shot.
        single = _make_material(orientations[point], 0.0)
        status_c = _run_partition(single, data, row, origin, target, 1)
        if status_b != 1 or status_c != 1:
            rows.append(
                {
                    "row": row,
                    "point": int(point),
                    "historical_substepped": True,
                    "divisions": divisions,
                    "zero_subgradient_forced_status": status_b,
                    "zero_subgradient_single_status": status_c,
                }
            )
            continue
        stress_b, internal_b = _state(forced)
        stress_c, internal_c = _state(single)

        rows.append(
            {
                "row": row,
                "point": int(point),
                "historical_substepped": True,
                "divisions": divisions,
                "a_vs_b": {
                    "stress": _relative(stress_b, stress_a),
                    **{
                        name: _relative(internal_b[span], internal_a[span])
                        for name, span in names.items()
                    },
                },
                "b_vs_c": {
                    "stress": _relative(stress_c, stress_b),
                    **{
                        name: _relative(internal_c[span], internal_b[span])
                        for name, span in names.items()
                    },
                },
            }
        )

    complete = [r for r in rows if "a_vs_b" in r]
    keys = ["stress", *names]
    summary = {
        "records": len(rows),
        "comparable_records": len(complete),
        "divisions_histogram": {
            str(d): sum(1 for r in complete if r["divisions"] == d)
            for d in sorted({int(r["divisions"]) for r in complete})
        },
        "a_vs_b_max": {k: max(r["a_vs_b"][k] for r in complete) for k in keys}
        if complete
        else {},
        "b_vs_c_max": {k: max(r["b_vs_c"][k] for r in complete) for k in keys}
        if complete
        else {},
        "b_vs_c_records_above_1e-6_on_slip": sum(
            1 for r in complete if r["b_vs_c"]["plastic_slip"] > 1.0e-6
        ),
    }
    return {"input": str(input_path), "summary": summary, "records": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--orientation-h5", type=Path, required=True)
    parser.add_argument("--crop-nodes", nargs=4, type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(
        args.input, _orientations(args.orientation_h5, tuple(args.crop_nodes))
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
