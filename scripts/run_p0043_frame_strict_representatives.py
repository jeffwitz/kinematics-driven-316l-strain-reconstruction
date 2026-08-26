#!/usr/bin/env python3
"""Strict rerun of the four P43 frame-battery representatives."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from fem_inhouse.core.crystal_orientation import rotations_from_euler_bunge_deg
from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET, get_parameter_set
from fem_inhouse.identification.srix_parameter_coordinates import SrixTheta9
from scipy.spatial.transform import Rotation
from scripts.qualify_srix_p0043_synthetic_smoke import _forward, _load_inputs, _make_path, _vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/reference_data/p0043_frame_battery_v1/strict_representatives"
CROP = (1610, 1630, 1075, 1095)


def _d4(values: np.ndarray, name: str) -> np.ndarray:
    if name == "identity": return values
    if name == "rot90": return np.rot90(values, 1, axes=(0, 1))
    if name == "anti_transpose": return np.rot90(values.transpose(1, 0, 2), 2, axes=(0, 1))
    if name == "flip_x": return values[::-1, :, ...]
    raise ValueError(name)


def _sample_frame(angles: np.ndarray, degrees: float) -> np.ndarray:
    q = rotations_from_euler_bunge_deg(angles)
    rz = Rotation.from_euler("z", degrees, degrees=True).as_matrix()
    transformed = np.einsum("...ij,jk->...ik", q, rz)
    return Rotation.from_matrix(transformed.transpose(0, 1, 3, 2)).as_euler("ZXZ", degrees=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    measured, angles, provenance = _load_inputs(CROP)
    path = _make_path(measured, 4)
    scored = tuple(4 * i for i in range(1, 9))
    target = [np.asarray(step.boundary, dtype=float) for step in path]
    theta = SrixTheta9.from_parameter_set(get_parameter_set(DEFAULT_PARAMETER_SET))
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so"))
    cases = (("A_identity_rz90", "identity", 90.0), ("B_identity_rz0", "identity", 0.0),
             ("C_anti_transpose_rz0", "anti_transpose", 0.0), ("D_flip_x_rz90", "flip_x", 90.0))
    rows = []
    for label, pname, degrees in cases:
        started = time.perf_counter()
        transformed = _sample_frame(_d4(angles, pname), degrees)
        fields, timing = _forward(theta, path, transformed, library, args.threads, "F")
        residual = _vector(fields, scored, target)
        rms = float(np.sqrt(np.mean(residual**2)))
        np.savez_compressed(OUT / f"{label}.npz", displacement=np.asarray([f.displacement for f in fields]),
                            residual=residual, boundary=np.asarray([f.boundary for f in fields]))
        row = {"label": label, "spatial_transform": pname, "sample_frame_degrees": degrees,
               "raw_rms_mm": rms, "equilibrium_residual": float(timing["verification_residual"]),
               "gmres_iterations": timing["gmres_iterations"], "elapsed_seconds": time.perf_counter() - started}
        rows.append(row); print(json.dumps(row, sort_keys=True), flush=True)
    with (OUT / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = {"schema_version": 1, "crop": list(CROP), "path_steps": len(path), "scored_steps": list(scored),
              "parameters": theta.as_runtime_overrides(), "mapping_order": "F", "spectral_storage_order": "C",
              "provenance": provenance, "results": rows,
              "strict_equilibrium_threshold": 1e-8,
              "best_strict": min((r for r in rows if r["equilibrium_residual"] <= 1e-8), key=lambda r: r["raw_rms_mm"], default=None)}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
