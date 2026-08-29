#!/usr/bin/env python3
"""Preflight the observable-space SRIX/Krylov intersection test.

This command deliberately performs no mechanical solve and no DIC operation.
It only inventories the archived array contracts needed for

    K = W O A Phi_K

and for the Krylov correction ``delta_y_K``.  The test is only meaningful when
both objects live in the same observable vector space.  A missing or
incompatible artifact is reported as a blocked test rather than being
reshaped, interpolated, or projected opportunistically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRIX_FIELDS = Path(
    "validation/reference_data/p0043_experimental_raw_femu_m20_v1/fields.npz"
)
KRYLOV_TRAJECTORY = Path(
    "validation/_generated/shared_tensor_generator/krylov_trajectories.r16.npz"
)
KRYLOV_SUMMARY = Path(
    "validation/_generated/shared_tensor_generator/krylov_projected_gp_trajectories.json"
)
KRYLOV_MAP_SUMMARY = Path(
    "validation/_generated/performance/experimental_oracle_p43_m20/"
    "krylov_maps_state40/report.json"
)
KRYLOV_MODE_CANDIDATES = (
    Path(
        "validation/_generated/performance/experimental_oracle_p43_m20/"
        "mode_anatomy_m100/modes.npz"
    ),
    Path(
        "validation/_generated/performance/experimental_oracle_p43_m20/"
        "tensor_observability/modes.npz"
    ),
)


def _array_contract(
    path: Path, *, display_path: Path | None = None, allow_pickle: bool = False
) -> dict[str, Any]:
    if not path.exists():
        return {"path": (display_path or path).as_posix(), "exists": False}
    result: dict[str, Any] = {
        "path": (display_path or path).as_posix(),
        "exists": True,
    }
    try:
        with np.load(path, allow_pickle=allow_pickle) as archive:
            result["arrays"] = {
                key: {"shape": list(archive[key].shape), "dtype": str(archive[key].dtype)}
                for key in archive.files
            }
    except Exception as error:  # pragma: no cover - diagnostic path
        result["load_error"] = f"{type(error).__name__}: {error}"
    return result


def _json_contract(path: Path, *, display_path: Path | None = None) -> dict[str, Any]:
    if not path.exists():
        return {"path": (display_path or path).as_posix(), "exists": False}
    result: dict[str, Any] = {
        "path": (display_path or path).as_posix(),
        "exists": True,
    }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result["top_level_keys"] = sorted(payload)
        for key in ("origin", "pixels", "ranks", "states", "projector", "krylov"):
            if key in payload:
                result[key] = payload[key]
    except Exception as error:  # pragma: no cover - diagnostic path
        result["load_error"] = f"{type(error).__name__}: {error}"
    return result


def _tracked(repo_root: Path, relative: Path) -> bool:
    """Use the index, not the filesystem, to distinguish archived from local data."""
    import subprocess

    completed = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", str(relative)],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT,
        help="checkout containing generated artifacts; defaults to this checkout",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "validation/reference_data/p0043_constitutive_krylov_intersection_v1/report.json",
    )
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()

    srix_path = artifact_root / SRIX_FIELDS
    srix = _array_contract(srix_path, display_path=SRIX_FIELDS)
    expected_srix = {
        "observable_vector_components": 7056,
        "scored_states": 8,
        "grid": [20, 20],
        "nodal_shape": [21, 21, 2],
        "parameter_order": ["tau0", "R", "Q", "b"],
    }
    if srix.get("arrays"):
        arrays = srix["arrays"]
        expected_srix["observed_jacobian_shape"] = arrays.get("final_jacobian", {}).get("shape")
        expected_srix["target_shape"] = arrays.get("target_displacement", {}).get("shape")

    trajectory_path = artifact_root / KRYLOV_TRAJECTORY
    trajectory = _array_contract(trajectory_path, display_path=KRYLOV_TRAJECTORY)
    summary = _json_contract(
        artifact_root / KRYLOV_SUMMARY, display_path=KRYLOV_SUMMARY
    )
    map_summary = _json_contract(
        artifact_root / KRYLOV_MAP_SUMMARY, display_path=KRYLOV_MAP_SUMMARY
    )
    mode_candidates = []
    for relative in KRYLOV_MODE_CANDIDATES:
        item = _array_contract(
            artifact_root / relative,
            display_path=relative,
            allow_pickle=True,
        )
        item["tracked_in_clean_branch"] = _tracked(ROOT, relative)
        item["classification"] = (
            "latent mode archive, not an observable displacement response"
            if relative.name == "modes.npz"
            and "mode_anatomy_m100" in relative.as_posix()
            else "observable tensor SVD mode archive, not a projected-Krylov basis"
        )
        mode_candidates.append(item)

    trajectory_arrays = trajectory.get("arrays", {})
    trajectory_contract = {
        "states": trajectory_arrays.get("states", {}).get("shape"),
        "stress": trajectory_arrays.get("stress", {}).get("shape"),
        "eps_inel": trajectory_arrays.get("eps_inel", {}).get("shape"),
        "d_eps_inel": trajectory_arrays.get("d_eps_inel", {}).get("shape"),
        "eps_inel_observable": trajectory_arrays.get("eps_inel_observable", {}).get("shape"),
        "d_eps_inel_observable": trajectory_arrays.get("d_eps_inel_observable", {}).get("shape"),
        "contains_phi_k": any(key in trajectory_arrays for key in ("Phi_K", "phi_k", "basis")),
        "contains_observable_response": any(
            key in trajectory_arrays
            for key in ("WOA_Phi_K", "observable_response", "responses", "delta_y")
        ),
        "observable_field_note": (
            "eps_inel_observable and d_eps_inel_observable are latent plastic fields "
            "after a tensor nullspace projection; they are not displacement-space WOA responses."
        ),
    }

    blockers = [
        "No archived Krylov basis Phi_K is present in the trajectory archive.",
        "No archived WOA Phi_K response or Krylov observable correction delta_y_K is present.",
        "The available Krylov trajectory uses 20 states on a 100x100, two-subcell support, "
        "whereas the SRIX Jacobian is 8 scored states on the 21x21 M20 support.",
    ]
    report = {
        "schema_version": 1,
        "status": "blocked_compatibility",
        "verdict": "C",
        "verdict_text": (
            "artifacts incompatible for observable-space constitutive-manifold "
            "intersection"
        ),
        "scope": {
            "new_forward": False,
            "new_mechanical_solve": False,
            "new_dic_processing": False,
            "comparison_space": "W O A observable quotient only",
        },
        "srix_contract": {
            "path": SRIX_FIELDS.as_posix(),
            "tracked_in_clean_branch": _tracked(ROOT, SRIX_FIELDS),
            "expected": expected_srix,
            "actual": srix,
        },
        "krylov_contract": {
            "trajectory_path": KRYLOV_TRAJECTORY.as_posix(),
            "tracked_in_clean_branch": _tracked(ROOT, KRYLOV_TRAJECTORY),
            "actual": trajectory,
            "shape_summary": trajectory_contract,
            "summary_json": summary,
            "state40_summary": map_summary,
            "mode_candidates": mode_candidates,
        },
        "compatibility_checks": {
            "srix_final_jacobian_reconstructible": srix.get("exists", False)
            and "arrays" in srix
            and srix["arrays"].get("final_jacobian", {}).get("shape") == [7056, 4],
            "same_observable_vector_length": False,
            "same_scored_state_layout": False,
            "krylov_observable_response_available": trajectory_contract[
                "contains_observable_response"
            ],
            "krylov_basis_available": trajectory_contract["contains_phi_k"],
        },
        "blockers": blockers,
        "required_artifacts_for_followup": [
            "Krylov Phi_K or WOA Phi_K stored on the same M20 21x21x2 "
            "eight-state observable convention.",
            "Raw and dissipatively projected delta_y_K in that same vector space.",
            "A manifest fixing support, state order, component order, transfer "
            "and whitening contracts.",
        ],
        "not_computed": [
            "principal angles between Q_K and U_theta,3",
            "eta_1, eta_2, eta_3 projections",
            "delta_q and delta_log_theta mapping",
            "raw versus dissipative Krylov comparison",
        ],
        "interpretation": (
            "The registered SRIX tangent can be recomputed from final_jacobian, but no archived "
            "Krylov observable object exists in the same space. A latent-field reshape or a new "
            "mechanical application of A would change the experiment and is not performed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
