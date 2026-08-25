#!/usr/bin/env python3
"""Promote the converged F centered-FD Jacobian as the F observability oracle."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/reference_data/p0043_f_mapping_reidentification_v1"
SRC = ROOT / "validation/reference_data/p0043_f_mapping_reidentification_prior_v2"
GLOBAL = ROOT / "validation/reference_data/p0043_global_srix_observability_v1"


def main() -> int:
    jfd = np.load(SRC / "full_jacobian_f_fd.npy")
    np.save(OUT / "full_jacobian_f.npy", jfd)
    u, s, vh = np.linalg.svd(jfd, full_matrices=False)
    v = vh.T
    np.save(OUT / "full_jacobian_f_fd.npy", jfd)
    (OUT / "svd_f.json").write_text(json.dumps({
        "method": "centered finite differences, F mapping, h=0.0015",
        "singular_values": s.tolist(), "normalized_singular_values": (s / s[0]).tolist(),
        "right_singular_vectors": v.tolist(), "fd_source": str(SRC / "projected_shadow_f_qualification.json"),
    }, indent=2, sort_keys=True) + "\n")
    c = np.asarray(np.load(GLOBAL / "global_observability.npz")["eigenvectors"], float)
    rank7 = np.degrees(subspace_angles(c[:, :7], v[:, :7]))
    weak = np.column_stack([[1, 1, 1, 1, 1, 1, 0, 1, 0], [0, 0, 0, 0, 0, 1, -1, 0, 0]])
    weak_angles = np.degrees(subspace_angles(v[:, 7:], weak))
    (OUT / "c_vs_f_subspace_angles.json").write_text(json.dumps({
        "source": "F centered-FD oracle", "rank7_angles_deg": rank7.tolist(),
        "rank7_max_deg": float(np.max(rank7)), "rank7_mean_deg": float(np.mean(rank7)),
    }, indent=2, sort_keys=True) + "\n")
    (OUT / "weak_modes_f.json").write_text(json.dumps({
        "source": "F centered-FD oracle", "weak_subspace_angles_to_scale_and_q_over_b_deg": weak_angles.tolist(),
        "v8": v[:, 7].tolist(), "v9": v[:, 8].tolist(),
    }, indent=2, sort_keys=True) + "\n")
    q = json.loads((SRC / "projected_shadow_f_qualification.json").read_text())
    q["shadow_qualified"] = False
    q["note"] = "F centered-FD is the current oracle; projected shadow F is not qualified and must not drive optimization."
    (OUT / "projected_shadow_f_qualification.json").write_text(json.dumps(q, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rank7_max_angle_deg": float(np.max(rank7)), "weak_angles_deg": weak_angles.tolist(),
                      "normalized_singular_values": (s / s[0]).tolist()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
