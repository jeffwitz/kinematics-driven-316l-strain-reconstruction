#!/usr/bin/env python3
"""Rebuild geometry-derived fields from an archived SRIX Jacobian artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles

from scripts.qualify_srix_regm_information_geometry import (
    PARAMETER_NAMES,
    _cumulative_geometry,
    _geometry,
    _plot,
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: rebuild_srix_regm_information_geometry OUTPUT_DIR")
    output = Path(sys.argv[1])
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    with np.load(output / "jacobians.npz") as values:
        matrices = {name: np.asarray(values[name], dtype=np.float64) for name in values.files}
    block_size = matrices["FEMU_observed"].shape[0] // len(report["states_scored"])
    geometries = {}
    for label, matrix in matrices.items():
        geometry = _geometry(matrix)
        geometry["cumulative"] = _cumulative_geometry(matrix, block_size)
        geometries[label] = geometry
    pairs = (
        ("REGM_exact", "REGM_observed"),
        ("REGM_exact", "FEMU_observed"),
        ("REGM_observed", "FEMU_observed"),
    )
    angles = {}
    for left, right in pairs:
        maximum = min(
            geometries[left]["numerical_rank"],
            geometries[right]["numerical_rank"],
            3,
        )
        angles[f"{left}__{right}"] = {}
        for count in range(1, maximum + 1):
            left_vectors = np.asarray(geometries[left]["right_singular_vectors"])[:, :count]
            right_vectors = np.asarray(geometries[right]["right_singular_vectors"])[:, :count]
            angles[f"{left}__{right}"][str(count)] = np.degrees(
                subspace_angles(left_vectors, right_vectors)
            ).tolist()
    report["parameter_names"] = PARAMETER_NAMES
    report["geometries"] = geometries
    report["subspace_principal_angles_degrees"] = angles
    report["geometry_rebuilt_from_archived_jacobians"] = True
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _plot(geometries, output)


if __name__ == "__main__":
    main()
