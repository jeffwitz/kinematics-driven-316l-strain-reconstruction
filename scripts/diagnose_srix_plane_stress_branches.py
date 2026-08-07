"""Diagnose the multiple-root branch structure of the SRIX plane-stress problem.

Exploratory diagnostic, NOT a decision procedure (like
validation/gradient_fluctuation_criteria_diagnostic.md). It answers four
questions about the fixed-point disagreement measured in
validation/srix_umat_gps_closure_results.md:

1. where do the reference and the UMAT trajectories first diverge;
2. how many distinct roots the raw 3D law has at the divergence state
   (perturbed starts -> converged states -> clustering);
3. which slip systems are active in each root;
4. how sensitive each backend's selected branch is to its own start.

Usage:

    .venv/bin/python scripts/diagnose_srix_plane_stress_branches.py \
        --output validation/_generated/performance/srix_plane_stress_branches.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

INCREMENTS = 12
MAX_STRAIN = 0.02
PERTURBATION = 1.0e-4
STARTS = 10


def _history() -> np.ndarray:
    return np.array(
        [
            (i / INCREMENTS) * MAX_STRAIN * np.array([1.0, -0.4, 0.0])
            for i in range(1, INCREMENTS + 1)
        ]
    )


def _make(backend: str, library: str, parameter_set: str) -> object:
    from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch

    return create_plane_stress_material_batch(
        backend,
        np.full((1, 1), 250.0),
        np.full((1, 1), 500.0),
        0.245,
        young_modulus_mpa=205000.0,
        poisson_ratio=0.3,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1000,
        first_positive_plastic_strain=1e-6,
        mfront_library=library,
        mfront_threads=1,
        mfront_behaviour_id="fcc_forest_rubin_srix",
        constitutive_options={"parameter_set": parameter_set},
    )


def _active_systems(slips: np.ndarray, threshold: float = 1.0e-9) -> list[int]:
    return [int(i) for i, value in enumerate(np.asarray(slips)[0]) if abs(value) > threshold]


def _trajectories(
    library: str, parameter_set: str
) -> tuple[list[dict[str, object]], int]:
    """Drive both backends through the history; return per-increment records."""

    reference = _make("mfront-3d-condensed-plane-stress", library, parameter_set)
    candidate = _make("mfront-native-generalised-plane-stress", library, parameter_set)
    records: list[dict[str, object]] = []
    first_divergence = -1
    for index, in_plane in enumerate(_history()):
        strain = np.tile(in_plane, (1, 1))
        trial_ref = reference.evaluate(strain, time_increment=1.0 / INCREMENTS)
        try:
            trial_gps = candidate.evaluate(strain, time_increment=1.0 / INCREMENTS)
        except Exception as error:
            records.append(
                {
                    "increment": index + 1,
                    "gps_failure": str(error),
                }
            )
            if first_divergence < 0:
                first_divergence = index + 1
            break
        stress_error = float(
            np.max(np.abs(trial_ref.stress_in_plane_mpa - trial_gps.stress_in_plane_mpa))
        )
        records.append(
            {
                "increment": index + 1,
                "stress_error_mpa": stress_error,
                "reference_eps_zz": float(trial_ref.full_strain_tensor[0, 2, 2]),
                "gps_eps_zz": float(trial_gps.full_strain_tensor[0, 2, 2]),
                "reference_slip_sum": float(
                    np.sum(trial_ref.observables["plastic_slip"])
                ),
                "gps_slip_sum": float(np.sum(trial_gps.observables["plastic_slip"])),
                "reference_active": _active_systems(trial_ref.observables["plastic_slip"]),
                "gps_active": _active_systems(trial_gps.observables["plastic_slip"]),
            }
        )
        if first_divergence < 0 and stress_error > 1.0e-6:
            first_divergence = index + 1
        reference.commit()
        candidate.commit()
    return records, first_divergence


def _raw_3d_roots(
    library: str,
    parameter_set: str,
    *,
    in_plane: np.ndarray,
    transverse: np.ndarray,
    time_increment: float,
) -> list[dict[str, object]]:
    """Count distinct roots of the raw 3D law at one full strain state.

    Drives the raw law through the elastic pre-history, then evaluates the
    target strain from many slightly perturbed committed states. Distinct
    converged states are clustered by their stress and slip vectors.
    """

    from fem_inhouse.core.mfront import (
        MFront3DMaterialPointBatch,
        _ENGINEERING_TO_KELVIN_STRAIN_SCALE,
        _PLANE_STRESS_COMPONENTS,
        _TRANSVERSE_COMPONENTS_3D,
        _SQRT_TWO,
    )
    from fem_inhouse.core.mfront_behaviours import MFRONT_BEHAVIOURS

    # Pre-history: increment 1 of INCREMENTS, with the closure transverse the
    # two backends agree on (both commit -0.000735 at increment 1), so the raw
    # 3D law starts increment 2 from the SAME state as the backends.
    pre = (1.0 / INCREMENTS) * MAX_STRAIN * np.array([1.0, -0.4, 0.0])
    pre_kelvin = np.zeros(6)
    pre_kelvin[_PLANE_STRESS_COMPONENTS] = pre * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
    pre_kelvin[_TRANSVERSE_COMPONENTS_3D] = np.array(
        [-0.000735, 0.0, 0.0]
    ) * np.array([1.0, 1.0 / _SQRT_TWO, 1.0 / _SQRT_TWO])
    target_kelvin = np.zeros(6)
    target_kelvin[_PLANE_STRESS_COMPONENTS] = (
        in_plane * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
    )
    target_kelvin[_TRANSVERSE_COMPONENTS_3D] = transverse * np.array(
        [1.0, 1.0 / _SQRT_TWO, 1.0 / _SQRT_TWO]
    )

    roots: list[dict[str, object]] = []
    seen: list[np.ndarray] = []
    for start in range(STARTS):
        bridge = MFront3DMaterialPointBatch(
            library,
            behaviour_spec=MFRONT_BEHAVIOURS.get("fcc_forest_rubin_srix"),
            point_count=1,
            rotation_global_to_material=np.eye(3)[None, :, :],
            thread_count=1,
            behaviour_name="Fcc316LForestRubinSrix",
            behaviour_parameters=None,
        )
        bridge.evaluate(pre_kelvin[None, :], time_increment=time_increment)
        bridge.commit()
        if start > 0:
            # Perturb the committed internal state.
            isv = np.asarray(bridge._manager.s0.internal_state_variables).copy()
            isv[:, 6:18] += PERTURBATION * (2 * np.random.default_rng(start).random(isv[:, 6:18].shape) - 1)
            for state in (bridge._manager.s0, bridge._manager.s1):
                state.internal_state_variables[:, :] = isv
        trial = bridge.evaluate(target_kelvin[None, :], time_increment=time_increment)
        state = np.concatenate(
            (
                np.asarray(trial.stress_kelvin_mpa).reshape(-1),
                np.asarray(trial.observables["plastic_slip"]).reshape(-1),
            )
        )
        cluster = None
        for index, previous in enumerate(seen):
            if np.max(np.abs(state - previous)) < 1.0e-6:
                cluster = index
                break
        if cluster is None:
            cluster = len(seen)
            seen.append(state)
            roots.append(
                {
                    "root_index": cluster,
                    "sigma_zz_mpa": float(trial.stress_kelvin_mpa[0, 2]),
                    "slip_sum": float(np.sum(trial.observables["plastic_slip"])),
                    "active": _active_systems(trial.observables["plastic_slip"]),
                    "starts": [start],
                }
            )
        else:
            roots[cluster]["starts"].append(start)
    return roots


def _start_sensitivity(
    backend: str,
    library: str,
    parameter_set: str,
    *,
    divergence_increment: int,
) -> dict[str, object]:
    """Drive the backend to the divergence state from perturbed starts."""

    results: dict[str, object] = {}
    for start in range(STARTS):
        batch = _make(backend, library, parameter_set)
        history = _history()
        try:
            for index in range(1, divergence_increment):
                strain = np.tile(history[index - 1], (1, 1))
                batch.evaluate(strain, time_increment=1.0 / INCREMENTS)
                batch.commit()
            if start > 0:
                isv = np.asarray(batch._manager.s0.internal_state_variables).copy()
                isv[:, 6:18] += PERTURBATION * (
                    2 * np.random.default_rng(100 + start).random(isv[:, 6:18].shape) - 1
                )
                for state in (batch._manager.s0, batch._manager.s1):
                    state.internal_state_variables[:, :] = isv
            strain = np.tile(history[divergence_increment - 1], (1, 1))
            trial = batch.evaluate(strain, time_increment=1.0 / INCREMENTS)
            results[start] = {
                "stress_in_plane_mpa": trial.stress_in_plane_mpa[0].tolist(),
                "eps_zz": float(trial.full_strain_tensor[0, 2, 2]),
                "slip_sum": float(np.sum(trial.observables["plastic_slip"])),
                "active": _active_systems(trial.observables["plastic_slip"]),
            }
        except Exception as error:
            results[start] = {"failure": str(error)}
    return results


def _main() -> int:
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        default=os.environ.get("MFRONT_BEHAVIOUR_LIBRARY"),
    )
    parser.add_argument(
        "--parameter-set",
        default="316l_srix_transposed_from_nasri2018_rate_1e-3",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/_generated/performance/srix_plane_stress_branches.json"),
    )
    arguments = parser.parse_args()
    if not arguments.library:
        parser.error("--library is required")
    library = str(Path(arguments.library).resolve())

    report: dict[str, object] = {}
    trajectories, first_divergence = _trajectories(library, arguments.parameter_set)
    report["trajectories"] = trajectories
    report["first_divergence_increment"] = first_divergence
    if first_divergence < 0:
        report["note"] = "no divergence: the two backends agree on the whole history"
    else:
        divergence = trajectories[first_divergence - 1]
        in_plane = _history()[first_divergence - 1]
        transverse_reference = np.array(
            [divergence["reference_eps_zz"], 0.0, 0.0]
        )
        transverse_gps = np.array([divergence["gps_eps_zz"], 0.0, 0.0])
        report["roots_at_reference_transverse"] = _raw_3d_roots(
            library,
            arguments.parameter_set,
            in_plane=in_plane,
            transverse=transverse_reference,
            time_increment=1.0 / INCREMENTS,
        )
        report["roots_at_gps_transverse"] = _raw_3d_roots(
            library,
            arguments.parameter_set,
            in_plane=in_plane,
            transverse=transverse_gps,
            time_increment=1.0 / INCREMENTS,
        )
        report["reference_start_sensitivity"] = _start_sensitivity(
            "mfront-3d-condensed-plane-stress",
            library,
            arguments.parameter_set,
            divergence_increment=first_divergence,
        )
        report["gps_start_sensitivity"] = _start_sensitivity(
            "mfront-native-generalised-plane-stress",
            library,
            arguments.parameter_set,
            divergence_increment=first_divergence,
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
