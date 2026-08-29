#!/usr/bin/env python3
"""Compare an observable-matched Krylov fixture with the archived SRIX tangent.

The analysis is deliberately linear and offline. It consumes the displacement
space M20 fixture and the registered raw-FEMU Jacobian; no SRIX forward,
finite-difference, or FEMU optimization is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRIX_ROOT = Path("validation/reference_data/p0043_experimental_raw_femu_m20_v1")
FIXTURE_ROOT = Path("validation/reference_data/p0043_krylov_srix_intersection_m20_v1")
OUTPUT = Path(
    "validation/reference_data/p0043_constitutive_krylov_intersection_v1/report.json"
)
SCORED_INDICES = np.asarray([3, 7, 11, 15, 19, 23, 27, 31], dtype=int)


def _stack_states(values: np.ndarray) -> np.ndarray:
    """Flatten state-major ``(8, 21, 21, 2)`` arrays to 7056 rows."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (8, 21, 21, 2):
        raise ValueError(f"expected eight nodal displacement states, got {array.shape}")
    return array.reshape(-1)


def _subspace_basis(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return an SVD basis and singular values for the columns of ``values``."""
    u, singular, _ = np.linalg.svd(values, full_matrices=False)
    scale = max(float(singular[0]) if singular.size else 0.0, 1e-300)
    rank = int(np.count_nonzero(singular > 1e-12 * scale))
    return u[:, :rank], singular


def _principal_angles(left: np.ndarray, right: np.ndarray) -> list[float]:
    """Return sorted principal angles in degrees, smallest first."""
    q_left, _ = _subspace_basis(left)
    overlaps = np.linalg.svd(q_left.T @ right, compute_uv=False)
    angles = np.degrees(np.arccos(np.clip(overlaps, -1.0, 1.0)))
    return np.sort(angles).tolist()


def _eta(values: np.ndarray, u: np.ndarray, count: int) -> float:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    denominator = max(float(vector @ vector), 1e-300)
    projected = u[:, :count].T @ vector
    return float((projected @ projected) / denominator)


def _tangent_equivalent(
    values: np.ndarray, u3: np.ndarray, singular: np.ndarray, vt3: np.ndarray
) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    coefficients = u3.T @ vector
    delta_q = coefficients / np.maximum(singular[:3], 1e-300)
    delta_log_theta = vt3.T @ delta_q
    return {
        "observable_projection_norm_fraction": float(
            (coefficients @ coefficients) / max(vector @ vector, 1e-300)
        ),
        "delta_q": delta_q.tolist(),
        "delta_log_theta_order_tau0_R_Q_b": delta_log_theta.tolist(),
        "parameter_factor_order_tau0_R_Q_b": np.exp(delta_log_theta).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    srix_root = root / SRIX_ROOT
    fixture_root = root / FIXTURE_ROOT

    source_fields = np.load(srix_root / "fields.npz", allow_pickle=False)
    source_report = json.loads((srix_root / "report.json").read_text(encoding="utf-8"))
    fixture_fields = np.load(fixture_root / "fields.npz", allow_pickle=False)
    fixture_report = json.loads(
        (fixture_root / "report.json").read_text(encoding="utf-8")
    )

    final_jacobian = np.asarray(source_fields["final_jacobian"], dtype=np.float64)
    if final_jacobian.shape != (7056, 4):
        raise ValueError(f"unexpected final_jacobian shape: {final_jacobian.shape}")
    target = np.asarray(source_fields["target_displacement"], dtype=np.float64)[
        SCORED_INDICES
    ]
    best = np.asarray(source_fields["best_displacement"], dtype=np.float64)[
        SCORED_INDICES
    ]
    final_residual = _stack_states(target - best)
    u, singular, vt = np.linalg.svd(final_jacobian, full_matrices=False)
    u3 = u[:, :3]
    vt3 = vt[:3, :]
    if singular[3] > 1e-4 * singular[0]:
        raise ValueError("SRIX tangent is not rank-3 under the declared diagnostic gate")

    mode_results: dict[str, Any] = {}
    eta_results: dict[str, Any] = {}
    mapping_results: dict[str, Any] = {}
    angles: dict[str, Any] = {}
    for rank in (8, 16):
        key = str(rank)
        contributions = np.asarray(
            fixture_fields[f"trajectory_mode_contributions_raw_r{rank}"],
            dtype=np.float64,
        )
        if contributions.shape != (7056, rank):
            raise ValueError(
                f"unexpected contribution shape for rank {rank}: {contributions.shape}"
            )
        basis, contribution_singular = _subspace_basis(contributions)
        angles[key] = {
            "principal_angles_degrees": _principal_angles(basis, u3),
            "fitted_krylov_trajectory_contribution_rank": int(basis.shape[1]),
            "contribution_singular_values": contribution_singular.tolist(),
        }
        raw = fixture_fields[f"delta_y_raw_r{rank}"]
        dissipative = fixture_fields[f"delta_y_dissipative_r{rank}"]
        eta_results[key] = {
            "raw": {f"eta{k}": _eta(raw, u, k) for k in (1, 2, 3)},
            "dissipative": {
                f"eta{k}": _eta(dissipative, u, k) for k in (1, 2, 3)
            },
        }
        mapping_results[key] = {
            "raw": _tangent_equivalent(raw, u3, singular, vt3),
            "dissipative": _tangent_equivalent(dissipative, u3, singular, vt3),
        }
        mode_results[key] = {
            "raw_norm": float(np.linalg.norm(raw)),
            "dissipative_norm": float(np.linalg.norm(dissipative)),
            "raw_fit_relative_error_by_state": fixture_report["fit"][
                "raw_relative_observable_error_by_state"
            ][key],
            "dissipative_fit_relative_error_by_state": fixture_report["fit"][
                "dissipative_relative_observable_error_by_state"
            ][key],
        }

    final_eta = {f"eta{k}": _eta(final_residual, u, k) for k in (1, 2, 3)}
    final_mapping = _tangent_equivalent(final_residual, u3, singular, vt3)
    max_raw_norm = max(item["raw_norm"] for item in mode_results.values())
    if max_raw_norm <= 1e-15:
        verdict = "D"
        verdict_text = "the observable-matched Krylov fixture produces no meaningful correction"
    else:
        eta3_max = max(item["raw"]["eta3"] for item in eta_results.values())
        smallest_angle = min(
            angle
            for item in angles.values()
            for angle in item["principal_angles_degrees"][:3]
        )
        if smallest_angle < 30.0 and eta3_max > 0.5:
            verdict = "A"
            verdict_text = (
                "the current SRIX tangent has substantial local geometric overlap "
                "with the fitted Krylov correction"
            )
        elif eta3_max > 0.2 or smallest_angle < 45.0:
            verdict = "B"
            verdict_text = (
                "SRIX explains part of the correction, but the remaining/final "
                "residual is not fully aligned with its local tangent"
            )
        else:
            verdict = "C"
            verdict_text = (
                "the current four-parameter SRIX tangent has weak overlap with "
                "the data-driven correction"
            )

    report = {
        "schema_version": 2,
        "status": "observable_matched_intersection_complete",
        "verdict": verdict,
        "verdict_text": verdict_text,
        "scope": {
            "new_srix_forward": False,
            "new_femu": False,
            "finite_differences": False,
            "comparison_space": "raw displacement observable, state-major 7056-vector",
            "experimental_status": "P43 registered-case methodological diagnostic only",
        },
        "srix_source": {
            "fields": SRIX_ROOT.joinpath("fields.npz").as_posix(),
            "report": SRIX_ROOT.joinpath("report.json").as_posix(),
            "source_git_sha": source_report.get("git_sha"),
            "scored_source_indices": SCORED_INDICES.tolist(),
            "scored_steps": source_report.get("scored_steps"),
            "parameter_order": ["tau0", "R", "Q", "b"],
            "jacobian_shape": list(final_jacobian.shape),
            "observation_profile": source_report.get("observation_profile"),
        },
        "fixture": {
            "fields": FIXTURE_ROOT.joinpath("fields.npz").as_posix(),
            "report": FIXTURE_ROOT.joinpath("report.json").as_posix(),
            "state_order": fixture_report["observable_contract"]["state_order"],
            "row_contract": fixture_report["observable_contract"],
            "basis": fixture_report["basis"],
            "gates": fixture_report["gates"],
            "fit": fixture_report["fit"],
        },
        "srix_tangent": {
            "singular_values": singular.tolist(),
            "normalized_singular_values": (singular / singular[0]).tolist(),
            "right_singular_vectors": vt.tolist(),
            "rank3": True,
        },
        "principal_angles": angles,
        "eta": {
            "raw_and_dissipative_krylov_by_rank": eta_results,
            "final_srix_residual": final_eta,
        },
        "tangent_equivalent_parameter_displacement": {
            "interpretation": (
                "local tangent-equivalent parameter displacement, not identified "
                "parameters"
            ),
            "raw_and_dissipative_krylov_by_rank": mapping_results,
            "final_srix_residual": final_mapping,
        },
        "final_residual": {
            "source": "target_displacement - best_displacement on the eight scored states",
            "norm": float(np.linalg.norm(final_residual)),
            "mode_results": mode_results,
        },
        "claims": {
            "trajectory_contribution_subspace_not_latent_field_distance": True,
            "no_claim_of_true_slip_recovery": True,
            "no_claim_of_material_calibration": True,
        },
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
