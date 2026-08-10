"""Analyse isolated failed SRIX GPS point snapshots.

The qualification driver writes these snapshots only when
``--gps-failure-diagnostics`` is enabled.  The records are captured after a
single-point MGIS integration has returned failure, before the substepper
restores the committed state.  This script reconstructs the SRIX overstress
and branch indicators from those raw rows; it does not change the behaviour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from diagnose_gps_direct_sensitivity import _MUS

from fem_inhouse.core.crystal_parameter_pairs import get_paired_crystal_parameter_set
from fem_inhouse.core.fcc_interaction_matrix import build_interaction_matrix


def analyse(path: Path, parameter_set: str) -> dict[str, object]:
    data = np.load(path)
    pair = get_paired_crystal_parameter_set(parameter_set)
    backbone = pair.backbone
    tau0 = backbone.tau0_mpa
    q_hard = backbone.q_mpa
    b = backbone.b
    c_hard = backbone.c_mpa
    d_hard = backbone.d
    mus = np.asarray(_MUS, dtype=float)
    interaction = build_interaction_matrix(pair.backbone.interaction_matrix)

    stress = np.asarray(data["s1_thermodynamic_forces"], dtype=float)
    s0 = np.asarray(data["s0_internal_state_variables"], dtype=float)
    s1 = np.asarray(data["s1_internal_state_variables"], dtype=float)
    dg = s1[:, 6:18]
    p = s0[:, 21:33]
    a = s0[:, 33:45]
    overstress = np.empty((len(stress), 12), dtype=float)
    for row in range(len(stress)):
        exp_bp = np.exp(-b * (p[row] + np.abs(dg[row])))
        for i in range(12):
            tau = float(np.dot(stress[row], mus[i]))
            radius = tau0 + q_hard * np.sum(
                interaction[i, :] * (1.0 - exp_bp)
            )
            da = (dg[row, i] - d_hard * a[row, i] * abs(dg[row, i])) / (
                1.0 + d_hard * abs(dg[row, i])
            )
            backstress = c_hard * (a[row, i] + da)
            overstress[row, i] = abs(tau - backstress) - radius

    active = overstress > 0.0
    signs = np.signbit(dg)
    result: dict[str, object] = {
        "input": str(path),
        "records": len(stress),
        "unique_points": len(np.unique(data["point"])),
        "parameter_set": parameter_set,
        "overstress_mpa": overstress.tolist(),
        "active_mask": active.tolist(),
        "dg": dg.tolist(),
        "dg_sign_negative": signs.tolist(),
        "local_iterations": {
            "available": False,
            "reason": (
                "The generated local counter is only promoted on a successful "
                "integration; failed MGIS trials leave it at zero."
            ),
        },
        "summary": {
            "min_overstress_mpa": float(np.min(overstress)),
            "max_overstress_mpa": float(np.max(overstress)),
            "near_threshold_count_abs_lt_1e-3_mpa": int(np.sum(np.abs(overstress) < 1.0e-3)),
            "near_threshold_count_abs_lt_1e-2_mpa": int(np.sum(np.abs(overstress) < 1.0e-2)),
            "near_threshold_count_abs_lt_1e-1_mpa": int(np.sum(np.abs(overstress) < 1.0e-1)),
            "near_threshold_count_abs_lt_1_mpa": int(np.sum(np.abs(overstress) < 1.0)),
            "near_zero_dg_count_abs_lt_1e-12": int(np.sum(np.abs(dg) < 1.0e-12)),
            "active_systems_min": int(np.min(np.sum(active, axis=1))),
            "active_systems_max": int(np.max(np.sum(active, axis=1))),
            "negative_dg_count": int(np.sum(signs)),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--parameter-set",
        default="316l_guilhem2013_nasri2018_meric_srix_rate_1e-3",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyse(args.input, args.parameter_set)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")


if __name__ == "__main__":
    main()
