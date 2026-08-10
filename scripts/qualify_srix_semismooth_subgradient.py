"""Decide whether ``sign(0) = 0`` changes the SRIX law or only its Jacobian.

The historical law assigns ``d|dg|/ddg = -1`` at exactly ``dg = 0``, the
left-hand branch of ``dg > 0 ? 1 : -1``. That value is one arbitrary element of
the Clarke subdifferential ``[-1, +1]`` of ``|x|`` at the cusp. Selecting the
symmetric element ``0`` rescues every archived isolated failure, which raises
the question this script answers: is that a different constitutive law, or the
same law solved with a better generalised Jacobian?

The protocol is in ``validation/srix_semismooth_subgradient_preregistration.md``
and its thresholds are frozen. Each archived failing state is replayed at
several fractions of its increment, because the historical convention does not
converge at the full one and a failed integration offers nothing to compare.
Only records where both variants converge enter the comparison.
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

VARIANTS: dict[str, tuple[float, float]] = {
    # label: (SrixSlipSmoothingDelta, SrixSlipZeroDerivative)
    "historical": (0.0, -1.0),
    "zero_subgradient": (0.0, 0.0),
    "compact_delta": (1.0e-5, -1.0),
}


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
    q: np.ndarray,
    delta: float,
    zero_derivative: float,
    epsilon: float | None = None,
) -> MFrontNativeGeneralisedPlaneStressBatch:
    spec = MFRONT_BEHAVIOURS.get("fcc_forest_rubin_srix_gps")
    parameters, _ = resolve_paired_crystal_parameters(
        paired_parameter_set="316l_guilhem2013_nasri2018_meric_srix_rate_1e-3",
        law="forest_rubin_srix",
    )
    parameters["SrixSlipSmoothingDelta"] = float(delta)
    parameters["SrixSlipZeroDerivative"] = float(zero_derivative)
    if epsilon is not None:
        # Convergence criterion of the local Newton, normally @Epsilon 1.e-14.
        # Sweeping it separates a solution difference that shrinks with the
        # tolerance -- the same root reached less precisely -- from one that
        # plateaus, which would mean a different root.
        parameters["epsilon"] = float(epsilon)
    return MFrontNativeGeneralisedPlaneStressBatch(
        os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", "build/mfront/src/libBehaviour.so"),
        behaviour_spec=spec,
        point_count=1,
        rotation_global_to_material=q[None, :, :],
        thread_count=1,
        behaviour_name=spec.behaviour_name("condensed_3d"),
        behaviour_parameters=parameters,
        backend_label="semismooth-subgradient",
    )


def _integrate(
    material: MFrontNativeGeneralisedPlaneStressBatch,
    data: np.ndarray,
    row: int,
    target: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Transplant s0 verbatim, integrate to ``target``, return s1."""

    material._manager.s0.gradients[0, :] = data["s0_gradient"][row]
    material._manager.s0.thermodynamic_forces[0, :] = data["s0_thermodynamic_forces"][row]
    material._manager.s0.internal_state_variables[0, :] = data[
        "s0_internal_state_variables"
    ][row]
    status = material._integrate_once(
        target[None, :], float(data["time_increment"][row]), None, (0, 1)
    )
    return (
        int(status),
        np.asarray(material._manager.s1.thermodynamic_forces[0, :], dtype=float).copy(),
        np.asarray(
            material._manager.s1.internal_state_variables[0, :], dtype=float
        ).copy(),
    )


def _relative(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Relative L2 difference, falling back to absolute on a null reference."""

    scale = float(np.linalg.norm(reference))
    difference = float(np.linalg.norm(candidate - reference))
    return difference / scale if scale > 0.0 else difference


#: Internal-state family compared for H2. The declared families are located by
#: the bridge itself, from MGIS metadata, rather than by parsing names here.
SLIP_FAMILY = "plastic_slip"


def qualify(
    input_path: Path,
    orientations: np.ndarray,
    fractions: list[float],
    epsilon: float | None = None,
) -> dict[str, object]:
    data = np.load(input_path)
    points = np.asarray(data["point"], dtype=int)
    s0_gradient = np.asarray(data["s0_gradient"], dtype=float)
    full_target = np.asarray(data["target_in_plane_kelvin"], dtype=float)

    names: dict[str, slice] = {}
    per_fraction: list[dict[str, object]] = []

    for fraction in fractions:
        statuses: dict[str, list[int]] = {label: [] for label in VARIANTS}
        stresses: dict[str, list[np.ndarray]] = {label: [] for label in VARIANTS}
        internals: dict[str, list[np.ndarray]] = {label: [] for label in VARIANTS}

        for row, point in enumerate(points):
            origin = s0_gradient[row, list(IN_PLANE)]
            target = origin + fraction * (full_target[row] - origin)
            for label, (delta, zero_derivative) in VARIANTS.items():
                material = _make_material(
                    orientations[point], delta, zero_derivative, epsilon
                )
                if not names:
                    names.update(material._observable_slices)
                status, stress, internal = _integrate(material, data, row, target)
                statuses[label].append(status)
                stresses[label].append(stress)
                internals[label].append(internal)

        converged = {
            label: np.asarray(statuses[label], dtype=int) == 1 for label in VARIANTS
        }
        record: dict[str, object] = {
            "fraction": fraction,
            "records": len(points),
            "converged": {label: int(mask.sum()) for label, mask in converged.items()},
            "comparisons": {},
        }

        reference_stress = np.asarray(stresses["historical"])
        reference_internal = np.asarray(internals["historical"])
        for label in ("zero_subgradient", "compact_delta"):
            both = converged["historical"] & converged[label]
            if not both.any():
                record["comparisons"][label] = {"comparable_records": 0}
                continue
            candidate_stress = np.asarray(stresses[label])
            candidate_internal = np.asarray(internals[label])
            stress_errors = [
                _relative(candidate_stress[i], reference_stress[i])
                for i in np.flatnonzero(both)
            ]
            per_variable: dict[str, float] = {}
            for name, span in names.items():
                per_variable[name] = max(
                    _relative(candidate_internal[i, span], reference_internal[i, span])
                    for i in np.flatnonzero(both)
                )
            all_internal = [
                _relative(candidate_internal[i], reference_internal[i])
                for i in np.flatnonzero(both)
            ]
            record["comparisons"][label] = {
                "comparable_records": int(both.sum()),
                "max_stress_relative": max(stress_errors),
                "max_internal_relative": max(all_internal),
                "max_relative_per_variable": per_variable,
                "records_above_1e-8_on_slip": int(
                    sum(
                        1
                        for i in np.flatnonzero(both)
                        if _relative(
                            candidate_internal[i, names[SLIP_FAMILY]],
                            reference_internal[i, names[SLIP_FAMILY]],
                        )
                        > 1.0e-8
                    )
                ),
            }
        per_fraction.append(record)

    return {
        "input": str(input_path),
        "local_newton_epsilon": epsilon,
        "variants": {
            label: {"SrixSlipSmoothingDelta": d, "SrixSlipZeroDerivative": z}
            for label, (d, z) in VARIANTS.items()
        },
        "internal_variable_order": {
            name: [span.start, span.stop] for name, span in names.items()
        },
        "fractions": per_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--orientation-h5", type=Path, required=True)
    parser.add_argument("--crop-nodes", nargs=4, type=int, required=True)
    parser.add_argument("--fraction", type=float, action="append", required=True)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(
        args.input,
        _orientations(args.orientation_h5, tuple(args.crop_nodes)),
        args.fraction,
        args.epsilon,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
