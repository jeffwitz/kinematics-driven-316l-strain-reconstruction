#!/usr/bin/env python3
"""Run the fixed-parameter P43 M20 spatial/frame registration battery."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from fem_inhouse.core.crystal_orientation import rotations_from_euler_bunge_deg
from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET, get_parameter_set
from fem_inhouse.identification.srix_parameter_coordinates import SrixTheta9
from scripts.plot_p0043_raw_svd7_evm_maps import _evm
from scripts.qualify_srix_p0043_synthetic_smoke import (
    _forward, _load_inputs, _make_path, _vector,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "validation/reference_data/p0043_frame_battery_v1"
CROP = (1610, 1630, 1075, 1095)
PIXEL_SIZE_MM = 0.00184


def _d4(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "identity": values,
        "rot90": np.rot90(values, 1, axes=(0, 1)),
        "rot180": np.rot90(values, 2, axes=(0, 1)),
        "rot270": np.rot90(values, 3, axes=(0, 1)),
        "flip_x": values[::-1, :, ...],
        "flip_y": values[:, ::-1, ...],
        "transpose": values.transpose(1, 0, 2),
        "anti_transpose": np.rot90(values.transpose(1, 0, 2), 2, axes=(0, 1)),
    }


def _rotate_sample_frame(angles: np.ndarray, degrees: float) -> np.ndarray:
    q = rotations_from_euler_bunge_deg(angles)
    rz = Rotation.from_euler("z", degrees, degrees=True).as_matrix()
    transformed = np.einsum("...ij,jk->...ik", q, rz)
    # scipy's ZXZ convention applied to Q.T is the inverse of the Bunge
    # matrix constructor used by the repository; this round-trip is tested.
    return Rotation.from_matrix(transformed.transpose(0, 1, 3, 2)).as_euler(
        "ZXZ", degrees=True
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    measured, angles, provenance = _load_inputs(CROP)
    path = _make_path(measured, 4)
    scored = tuple(4 * i for i in range(1, 9))
    target = [np.asarray(step.boundary, dtype=float) for step in path]
    theta = SrixTheta9.from_parameter_set(get_parameter_set(DEFAULT_PARAMETER_SET))
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so"))
    results: list[dict[str, object]] = []
    final_fields: dict[str, np.ndarray] = {}
    rms_matrix = np.empty((8, 4), dtype=float)
    started = time.perf_counter()
    for pi, (pname, p_angles) in enumerate(_d4(angles).items()):
        for si, degrees in enumerate((0.0, 90.0, 180.0, 270.0)):
            sname = f"rz{int(degrees)}"
            transformed = _rotate_sample_frame(p_angles, degrees)
            fields, timing = _forward(theta, path, transformed, library, args.threads, "F")
            residual = _vector(fields, scored, target)
            rms = float(np.sqrt(np.mean(residual**2)))
            rms_matrix[pi, si] = rms
            key = f"{pname}__{sname}"
            final_fields[key] = np.asarray(fields[-1].displacement)
            results.append({
                "spatial_transform": pname, "sample_frame": sname,
                "sample_frame_degrees": degrees, "raw_rms_mm": rms,
                "equilibrium_residual": float(timing["verification_residual"]),
                "steps": len(fields),
            })
            print(json.dumps(results[-1], sort_keys=True), flush=True)
    order = np.argsort([float(row["raw_rms_mm"]) for row in results])
    for rank, index in enumerate(order, 1):
        results[index]["rank"] = rank
    with (output / "ps_raw_matrix.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(results)
    np.savez_compressed(output / "fields.npz", **final_fields, rms_matrix=rms_matrix)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(rms_matrix * 1e6, cmap="viridis", aspect="auto")
    ax.set_xticks(range(4), ["Rz0", "Rz90", "Rz180", "Rz270"])
    ax.set_yticks(range(8), list(_d4(angles)))
    ax.set_xlabel("sample-frame S"); ax.set_ylabel("spatial mapping P")
    ax.set_title("P43 M20 — RAW displacement RMS (µm)")
    fig.colorbar(image, ax=ax, label="RMS (µm)")
    fig.savefig(output / "ps_raw_heatmap.png", dpi=220); plt.close(fig)
    best = results[int(order[0])]
    report = {
        "schema_version": 1, "crop": list(CROP), "path_steps": len(path),
        "scored_steps": list(scored), "mapping_order": "F",
        "spectral_storage_order": "C", "parameters": theta.as_runtime_overrides(),
        "provenance": provenance, "results": results,
        "best_joint_candidate": best, "elapsed_seconds": time.perf_counter() - started,
        "statuses": {"d4_battery_completed": True, "sample_frame_battery_completed": True,
                     "translation_scan_completed": False, "experimentally_proven": False},
    }
    (output / "final_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "best": best}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
