"""Qualify the generic structural plane-stress tangent against a live 3-D Schur.

The generic and raw behaviours are advanced through the same strain history.
Before each generic increment, its complete committed snapshot is transplanted
into a raw 3-D bridge by variable name.  The raw bridge is then integrated to
the exact generic target strain, and its 6x6 tangent is condensed once.  This
is deliberately a material-point qualification: no spectral solver, cached
JSON tangent, or independent state trajectory is involved.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from fem_inhouse.core.crystal_orientation import rotation_from_euler_bunge_deg
from fem_inhouse.core.crystal_parameter_pairs import resolve_paired_crystal_parameters
from fem_inhouse.core.mfront_behaviours import MFRONT_BEHAVIOURS
from fem_inhouse.core.mfront_condensation import MFront3DCondensedPlaneStressBatch
from fem_inhouse.core.mfront_gps.adapter import MFrontNativeGeneralisedPlaneStressBatch
from scripts.diagnose_gps_tangent_blocks import (
    _assert_same_physical_committed_state,
    _make_transplanted_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "build/mfront/src/libBehaviour.so"
RAW_LIBRARY = LIBRARY
PAIRED_PARAMETER_SET = "316l_guilhem2013_nasri2018_meric_srix_rate_1e-3"
Q = rotation_from_euler_bunge_deg(35.0, 20.0, 15.0)
HISTORY = np.array(
    [
        [2.0e-5, -4.0e-6, 6.0e-6],
        [5.0e-5, -1.0e-5, 1.5e-5],
        [1.0e-4, -2.0e-5, 3.0e-5],
        [2.0e-4, -4.0e-5, 6.0e-5],
        [5.0e-4, -1.0e-4, 1.5e-4],
        [1.0e-3, -2.0e-4, 3.0e-4],
    ],
    dtype=float,
)


def _generate_meric_library(directory: Path) -> Path:
    generated = directory / "Fcc316LMericCailletaudStructuralPlaneStress.mfront"
    environment = {
        **os.environ,
        "STRUCTURAL_BEHAVIOUR_NAME": "Fcc316LMericCailletaudStructuralPlaneStress",
        "STRUCTURAL_PLANE_STRESS_OUTPUT": str(generated),
        "STRUCTURAL_PLANE_STRESS_GENERATE_ONLY": "1",
    }
    command = [
        "scripts/generate_structural_plane_stress.sh",
        "mfront/Fcc316LMericCailletaud.mfront",
        "Fcc316LMericCailletaudStructuralPlaneStress",
    ]
    subprocess.run(command, cwd=ROOT, env={**environment}, check=True, text=True)
    subprocess.run(
        ["mfront", "--obuild", "--interface=generic", str(generated)],
        cwd=directory,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    library = directory / "src/libBehaviour.so"
    if not library.is_file():
        raise RuntimeError(f"Méric structural library was not built: {library}")
    return library


def _materials(law: str, generic_library: Path):
    if law == "srix":
        raw_spec = MFRONT_BEHAVIOURS.get("fcc_forest_rubin_srix")
        generic_spec = MFRONT_BEHAVIOURS.get(
            "fcc_forest_rubin_srix_structural_plane_stress"
        )
        raw_name = "Fcc316LForestRubinSrix"
        generic_name = "Fcc316LForestRubinSrixStructuralPlaneStress"
        flow_rule = "forest_rubin_srix"
        generic_path = generic_library
        dt = 1.0
    else:
        raw_spec = MFRONT_BEHAVIOURS.get("fcc_meric_cailletaud")
        generic_spec = replace(
            raw_spec,
            identifier="fcc_meric_cailletaud_structural_plane_stress",
            tridimensional_behaviour="Fcc316LMericCailletaudStructuralPlaneStress",
        )
        raw_name = "Fcc316LMericCailletaud"
        generic_name = "Fcc316LMericCailletaudStructuralPlaneStress"
        flow_rule = "meric_cailletaud"
        generic_path = generic_library
        dt = 1.0e-3

    parameters, _ = resolve_paired_crystal_parameters(
        paired_parameter_set=PAIRED_PARAMETER_SET,
        law=flow_rule,
    )
    generic = MFrontNativeGeneralisedPlaneStressBatch(
        generic_path,
        behaviour_spec=generic_spec,
        point_count=1,
        rotation_global_to_material=Q[None, :, :],
        thread_count=1,
        behaviour_name=generic_name,
        behaviour_parameters=parameters,
    )
    raw = MFront3DCondensedPlaneStressBatch(
        RAW_LIBRARY,
        behaviour_spec=raw_spec,
        point_count=1,
        rotation_global_to_material=Q[None, :, :],
        thread_count=1,
        behaviour_name=raw_name,
        behaviour_parameters=parameters,
    )
    return generic, raw, dt


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)) / max(float(np.max(np.abs(right))), 1.0e-30))


def qualify(law: str, generic_library: Path) -> dict[str, object]:
    generic, raw, dt = _materials(law, generic_library)
    rows: list[dict[str, object]] = []
    for state_index, target_in_plane in enumerate(HISTORY):
        generic_snapshot = generic.snapshot_state()
        raw_template = raw.snapshot_state()
        transplanted = _make_transplanted_snapshot(
            raw,
            raw_template,
            generic,
            generic_snapshot,
            0,
            q_global_to_material=Q,
        )
        same_state = _assert_same_physical_committed_state(
            generic,
            generic_snapshot,
            raw,
            transplanted,
            0,
            Q,
            tolerance=1.0e-13,
        )
        generic_trial = generic.evaluate(
            target_in_plane[None, :], time_increment=dt, consistent_tangent=True
        )
        generic_tangent = np.asarray(generic_trial.tangent_in_plane_mpa[0], dtype=float)
        raw.restore_state(transplanted)
        raw_trial = raw.evaluate(target_in_plane[None, :], time_increment=dt)
        raw_schur = np.asarray(raw_trial.tangent_in_plane_mpa[0], dtype=float)
        # Both adapters expose engineering in-plane stress/tangent.
        generic_stress = np.asarray(generic_trial.stress_in_plane_mpa[0], dtype=float)
        raw_stress_engineering = np.asarray(raw_trial.stress_in_plane_mpa[0], dtype=float)
        tangent_error = _relative(generic_tangent, raw_schur)
        stress_error = _relative(generic_stress, raw_stress_engineering)
        rows.append(
            {
                "state_index": state_index,
                "target_in_plane_engineering": target_in_plane.tolist(),
                "target_total_kelvin": np.asarray(generic._latest_total_kelvin[0]).tolist(),
                "time_increment": dt,
                "same_state_differences": same_state,
                "generic_vs_raw_stress_relative": stress_error,
                "generic_vs_raw_schur_relative": tangent_error,
                "generic_tangent_engineering": generic_tangent.tolist(),
                "raw_schur_engineering": raw_schur.tolist(),
            }
        )
        generic.commit()
        raw.revert()
    return {
        "law": law,
        "generic_behaviour": generic._behaviour_name,
        "raw_behaviour": raw._bridge.behaviour_name,
        "orientation_bunge_deg": [35.0, 20.0, 15.0],
        "states": rows,
        "max_tangent_relative_error": max(
            float(row["generic_vs_raw_schur_relative"]) for row in rows
        ),
        "max_stress_relative_error": max(
            float(row["generic_vs_raw_stress_relative"]) for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "validation/_generated/performance/"
            "structural_plane_stress_same_state_schur.json"
        ),
    )
    parser.add_argument("--srix-library", type=Path, default=LIBRARY)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="meric-structural-plane-stress-") as name:
        meric_library = _generate_meric_library(Path(name))
        report = {
            "protocol": "same committed state, same target, live raw 3-D Schur",
            "history": HISTORY.tolist(),
            "srix": qualify("srix", arguments.srix_library),
            "meric": qualify("meric", meric_library),
        }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]["max_tangent_relative_error"]
                for key in ("srix", "meric")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
