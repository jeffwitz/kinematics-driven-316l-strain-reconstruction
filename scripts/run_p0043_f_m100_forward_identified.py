#!/usr/bin/env python3
"""Run and archive one corrected-F P43 M100 forward from the M20 optimum."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fem_inhouse.identification.srix_parameter_coordinates import SrixTheta9
from scripts.plot_p0043_raw_svd7_evm_maps import _evm
from scripts.qualify_srix_p0043_synthetic_smoke import _load_inputs, _make_path, _vector
from scripts.run_p0043_f_m100_shadow_identification import (
    FULL_CROP,
    M20_REPORT,
    _forward,
    _plot_evm,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "validation/reference_data/p0043_f_m100_forward_identified_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--subdivisions", type=int, default=8)
    parser.add_argument("--element-order", choices=("C", "F"), default="F")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report_m20 = json.loads(M20_REPORT.read_text(encoding="utf-8"))
    eta = np.asarray(report_m20["final_eta"], dtype=float)
    measured, angles, provenance = _load_inputs(FULL_CROP)
    path = _make_path(measured, args.subdivisions)
    scored = tuple(args.subdivisions * i for i in range(1, 9))
    target = [np.asarray(step.boundary, dtype=float) for step in path]
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )
    fields, timing = _forward(
        SrixTheta9.from_log_coordinates(eta), path, angles, library, args.threads,
        args.element_order,
    )
    residual = _vector(fields, scored, target)
    final = fields[-1]
    strain = np.asarray([f.sample_strain for f in fields])
    stress = np.asarray([f.stress_in_plane_mpa for f in fields])
    tangent = np.asarray([f.algorithmic_tangent_in_plane_mpa for f in fields])
    displacement = np.asarray([f.displacement for f in fields])
    boundaries = np.asarray([f.boundary for f in fields])
    np.savez_compressed(
        output / "fields.npz",
        displacement=displacement,
        boundary=boundaries,
        sample_strain=strain,
        stress_in_plane_mpa=stress,
        algorithmic_tangent_in_plane_mpa=tangent,
        scored_displacement=displacement[np.asarray(scored) - 1],
        dic_displacement=np.asarray([target[i - 1] for i in scored]),
    )
    selected = [fields[i - 1] for i in scored]
    _plot_evm(output, measured, selected, selected, tuple(range(1, 9)))
    # A compact field panel for quick visual inspection of the final state.
    evm_sim = _evm(final.displacement)
    evm_dic = _evm(target[-1])
    evm_res = evm_sim - evm_dic
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    panels = [(evm_dic, "DIC EVM"), (evm_sim, "F forward EVM"), (evm_res, "EVM difference")]
    for ax, (field, title) in zip(axes, panels):
        image = ax.imshow(field.T, origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("x pixel")
        ax.set_ylabel("y pixel")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(output / "field_overview.png", dpi=180)
    plt.close(fig)
    result = {
        "schema_version": 1,
        "method": "single corrected-F P43 M100 forward from identified M20 parameters",
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()),
        "crop": list(FULL_CROP),
        "mesh": list(angles.shape[:2]),
        "path_steps": len(path),
        "subdivisions_per_state": args.subdivisions,
        "scored_steps": list(scored),
        "element_order": args.element_order,
        "spectral_batch_order": "C",
        "parameters": SrixTheta9.from_log_coordinates(eta).as_runtime_overrides(),
        "eta": eta.tolist(),
        "provenance": provenance,
        "timing": timing,
        "verification_residual": float(timing["verification_residual"]),
        "raw_rms_mm": float(np.sqrt(np.mean(residual**2))),
        "final_evm_dic_rms": float(np.sqrt(np.mean(evm_dic**2))),
        "final_evm_sim_rms": float(np.sqrt(np.mean(evm_sim**2))),
        "final_evm_difference_rms": float(np.sqrt(np.mean(evm_res**2))),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "steps": len(path), "raw_rms_mm": result["raw_rms_mm"], "verification_residual": result["verification_residual"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
