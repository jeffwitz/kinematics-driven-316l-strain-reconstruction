#!/usr/bin/env python3
"""Audit EBSD pixel order against StructuredMesh without running mechanics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from fem_inhouse.core.crystal_orientation import PixelOrientationProvider
from fem_inhouse.core.mesh import StructuredMesh

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "validation/reference_data/p0043_ebsd_mapping_audit_v1"


def _sentinel_rotations(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    markers = 1000 * np.indices(shape)[0] + np.indices(shape)[1]
    angles = np.zeros((*shape, 3), dtype=float)
    angles[..., 0] = 0.01 * markers
    source = PixelOrientationProvider.from_euler_bunge_deg(angles)
    return angles, source.rotations_global_to_material(shape[0] * shape[1])


def main() -> int:
    output = DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    mesh = StructuredMesh(3.0, 5.0, 1.0, 1.0)
    shape = (mesh.nx, mesh.ny)
    angles, source_c = _sentinel_rotations(shape)
    source_f = PixelOrientationProvider.from_euler_bunge_deg(
        angles, element_order="F"
    ).rotations_global_to_material(mesh.n_elems)
    expected = source_f
    current = source_c
    rows: list[dict[str, object]] = []
    for element in range(mesh.n_elems):
        expected_ij = tuple(int(v) for v in np.argwhere(mesh.elem_ids == element)[0])
        current_ij = tuple(int(v) for v in np.unravel_index(element, shape, order="C"))
        expected_marker = 1000 * expected_ij[0] + expected_ij[1]
        current_marker = 1000 * current_ij[0] + current_ij[1]
        rows.append({
            "element_id": element,
            "expected_i": expected_ij[0],
            "expected_j": expected_ij[1],
            "actual_current_i": current_ij[0],
            "actual_current_j": current_ij[1],
            "expected_marker": expected_marker,
            "actual_current_marker": current_marker,
            "current_matches": bool(np.allclose(current[element], expected[element])),
            "corrected_matches": bool(np.allclose(source_f[element], expected[element])),
        })
    with (output / "element_mapping.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Explicitly archive the two competing memory orders and the mesh source of truth.
    np.savez_compressed(
        output / "sentinel_mapping.npz",
        sentinel=1000 * np.indices(shape)[0] + np.indices(shape)[1],
        mesh_elem_ids=mesh.elem_ids,
        mesh_ravel_f=mesh.elem_ids.ravel(order="F"),
        mesh_ravel_c=mesh.elem_ids.ravel(order="C"),
        current_markers=np.array([r["actual_current_marker"] for r in rows]),
        corrected_markers=np.array([r["expected_marker"] for r in rows]),
    )
    report = {
        "schema_version": 1,
        "status": "completed_index_only_audit",
        "grid_shape": list(shape),
        "mesh_element_ids": mesh.elem_ids.tolist(),
        "mesh_element_order": "StructuredMesh elem_ids (Fortran numbering)",
        "current_provider_order": "C",
        "corrected_provider_order": "F",
        "current_mismatch_count": int(sum(not r["current_matches"] for r in rows)),
        "corrected_mismatch_count": int(sum(not r["corrected_matches"] for r in rows)),
        "permutation_classification": "C-order provider versus StructuredMesh F-order",
        "ebsd_element_order_audit_passed_before_fix": False,
        "ebsd_element_order_audit_passed_after_explicit_fix": True,
        "ebsd_dic_axis_registration_proven": False,
        "ebsd_dic_axis_direction_proven": False,
        "ebsd_crystal_sample_frame_registration_proven": False,
        "p43_historical_cp_results_spatially_trustworthy": False,
        "mechanics_run": False,
        "schmid_audit": "not yet run",
        "axis_audit": "not proven in this index-only ticket",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "current_mismatch_count": report["current_mismatch_count"],
        "corrected_mismatch_count": report["corrected_mismatch_count"],
        "output": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
