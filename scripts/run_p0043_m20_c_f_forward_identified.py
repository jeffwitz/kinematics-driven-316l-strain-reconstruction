#!/usr/bin/env python3
"""Run paired P43 M20 forwards with historical C and corrected F mapping."""

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
from scripts.qualify_srix_p0043_synthetic_smoke import (
    CROP,
    _forward,
    _load_inputs,
    _make_path,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPORT = ROOT / (
    "validation/reference_data/"
    "p0043_experimental_raw_svd7_f_provisional_v1/report.json"
)
DEFAULT_OUTPUT = ROOT / "validation/reference_data/p0043_m20_c_f_forward_identified_v1"


def _plot_evm(
    output: Path,
    dic: np.ndarray,
    c_map: np.ndarray,
    f_map: np.ndarray,
    labels: list[str],
) -> None:
    common = max(float(np.nanmax(dic)), float(np.nanmax(c_map)), float(np.nanmax(f_map)))
    differences = [c_map - dic, f_map - dic, f_map - c_map]
    diff_limit = max(float(np.nanmax(np.abs(item))) for item in differences)
    diff_limit = max(diff_limit, 1.0e-12)
    fig, axes = plt.subplots(
        len(labels), 6, figsize=(18, 2.55 * len(labels)), squeeze=False
    )
    for row, label in enumerate(labels):
        panels = (
            (dic[row], "DIC", "viridis", 0.0, common),
            (c_map[row], "C historique", "viridis", 0.0, common),
            (f_map[row], "F corrigé", "viridis", 0.0, common),
            (differences[0][row], "C - DIC", "coolwarm", -diff_limit, diff_limit),
            (differences[1][row], "F - DIC", "coolwarm", -diff_limit, diff_limit),
            (differences[2][row], "F - C", "coolwarm", -diff_limit, diff_limit),
        )
        for column, (field, title, cmap, vmin, vmax) in enumerate(panels):
            axis = axes[row, column]
            image = axis.imshow(
                100.0 * field.T,
                origin="lower",
                aspect="equal",
                cmap=cmap,
                vmin=100.0 * vmin,
                vmax=100.0 * vmax,
            )
            axis.set_title(f"{title}\nétat {label}", fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02, label="%")
    fig.suptitle("P43 M20 — EVM : DIC, mapping C historique et mapping F corrigé")
    fig.tight_layout()
    fig.savefig(output / "evm_c_vs_f_identified.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--subdivisions", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    measured, angles, provenance = _load_inputs(CROP)
    path = _make_path(measured, args.subdivisions)
    scored = tuple(args.subdivisions * index for index in range(1, 9))
    target = np.asarray([path[index - 1].boundary for index in scored])
    report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    theta = SrixTheta9.from_log_coordinates(np.asarray(report["final_eta"], dtype=float))
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )

    runs: dict[str, dict[str, object]] = {}
    evm_maps: dict[str, np.ndarray] = {"dic": np.stack([_evm(item) for item in target])}
    for order in ("C", "F"):
        started = time.perf_counter()
        fields, timing = _forward(theta, path, angles, library, args.threads, order)
        displacement = np.asarray([field.displacement for field in fields])
        selected = displacement[np.asarray(scored) - 1]
        evm = np.stack([_evm(item) for item in selected])
        residual = selected - target
        runs[order] = {
            "element_order": order,
            "elapsed_seconds_wall": time.perf_counter() - started,
            "timing": timing,
            "raw_rms_mm": float(np.sqrt(np.mean(residual**2))),
            "equilibrium_residual": float(timing["verification_residual"]),
        }
        evm_maps[order.lower()] = evm
        np.savez_compressed(
            output / f"fields_{order}.npz",
            displacement=displacement,
            scored_displacement=selected,
            dic_displacement=target,
            evm=evm,
            dic_evm=evm_maps["dic"],
        )

    _plot_evm(output, evm_maps["dic"], evm_maps["c"], evm_maps["f"], [str(i) for i in range(1, 9)])
    np.savez_compressed(output / "evm_fields.npz", **evm_maps)
    result = {
        "schema_version": 1,
        "method": "P43 M20 identified-parameter forward, C/F EBSD mapping comparison",
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "machine": platform.node(),
        "crop": list(CROP),
        "path_steps": len(path),
        "scored_steps": list(scored),
        "parameters": theta.as_runtime_overrides(),
        "source_report": str(SOURCE_REPORT.relative_to(ROOT)),
        "provenance": provenance,
        "runs": runs,
        "evm_rms_percent": {
            name: np.sqrt(np.mean(np.square(values), axis=(1, 2))).tolist()
            for name, values in evm_maps.items()
        },
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "runs": runs}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
