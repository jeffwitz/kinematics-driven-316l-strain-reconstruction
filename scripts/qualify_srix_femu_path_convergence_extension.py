#!/usr/bin/env python3
"""Run the preregistered final L2-to-L3 path refinement extension."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fem_inhouse.identification.dic_whitening import DICSpectralTransfer
from scripts.qualify_srix_femu_common_path_gate import _common_path
from scripts.qualify_srix_femu_direct_sensitivity import FD_STEP, ROOT, _direct_jacobian
from scripts.qualify_srix_femu_path_convergence_rebaseline import (
    TRANSFER,
    _compare,
    _geometry_with_contrast,
    _mandatory_refine,
    _nearest_indices,
    _observed_forward,
    _repair_base_path,
)
from scripts.qualify_srix_regm_transfer_noise import _WrapFreeTransfer
from scripts.qualify_srix_regm_twin import _orientation_map, _theta_from_preset

DEFAULT_SOURCE = ROOT / "validation/reference_data/srix_femu_path_convergence_v3"
DEFAULT_OUTPUT = ROOT / "validation/reference_data/srix_femu_path_convergence_v4"


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--library", default="build/mfront/src/libBehaviour.so")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_report = json.loads((source / "report.json").read_text())
    arrays = np.load(source / "path_convergence.npz")
    l2_fractions = np.asarray(arrays["end_fractions_L2"], dtype=np.float64)
    l2_path = _common_path(l2_fractions.tolist(), pixels=args.pixels)
    l3_path = _mandatory_refine(l2_path, pixels=args.pixels)
    print(f"L3: mandatory path with {len(l3_path)} steps", flush=True)
    repair_started = time.perf_counter()
    l3_path, fields, repairs = _repair_base_path(
        l3_path,
        pixels=args.pixels,
        library=args.library,
        threads=args.threads,
        max_repairs=128,
    )
    print(
        f"L3: converged with {len(l3_path)} steps and {len(repairs)} local repairs",
        flush=True,
    )
    targets = source_report["scored_target_fractions"]
    scored = _nearest_indices(fields, targets)
    transfer = _WrapFreeTransfer(DICSpectralTransfer.from_sinusoidal_csv(TRANSFER))
    direct, direct_timing = _direct_jacobian(
        fields=fields,
        scored=scored,
        orientations=_orientation_map(args.pixels),
        theta=_theta_from_preset(),
        library=args.library,
        threads=args.threads,
        transfer=transfer,
        h=FD_STEP,
    )
    l2 = {
        "steps": int(arrays["end_fractions_L2"].size),
        "forward_observed": arrays["forward_L2"],
        "jacobian": arrays["jacobian_L2"],
        "geometry": _geometry_with_contrast(arrays["jacobian_L2"]),
    }
    l3 = {
        "steps": len(l3_path),
        "forward_observed": _observed_forward(fields, scored, transfer),
        "jacobian": direct,
        "geometry": _geometry_with_contrast(direct),
    }
    comparison = _compare(l2, l3)
    primary_claim = (
        comparison["forward_observed_relative_l2"] < 5.0e-3
        and all(value < 2.0e-2 for value in comparison["column_relative_l2"][:3])
        and all(value > 0.999 for value in comparison["column_cosines"][:3])
        and max(comparison["rank3_principal_angles_degrees"]) < 2.0
        and max(comparison["normalized_singular_values"]["relative_change_first_three"]) < 0.05
    )
    report = {
        "schema_version": 1,
        "method": "final L2-to-L3 nested direct FEMU path refinement extension",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "source": str(source),
        "source_report_git_sha": source_report["git_sha"],
        "fd_step_log": FD_STEP,
        "levels_reused": ["L0", "L1", "L2"],
        "L3": {
            "steps": len(l3_path),
            "local_repairs": repairs,
            "local_repair_count": len(repairs),
            "scored": list(scored),
            "maximum_interval_width": max(
                step.end_fraction - step.start_fraction for step in l3_path
            ),
            "minimum_interval_width": min(
                step.end_fraction - step.start_fraction for step in l3_path
            ),
            "repair_and_forward_seconds": time.perf_counter() - repair_started,
            "direct_sensitivity": direct_timing,
            "normalized_singular_values": l3["geometry"]["normalized_singular_values"],
            "condition_number": l3["geometry"]["condition_number"],
            "q_minus_b_contrast_alignment_v4": l3["geometry"][
                "q_minus_b_contrast_alignment_v4"
            ],
        },
        "comparison_L2_to_L3": comparison,
        "claims": {
            "path_convergence_extension_gate": primary_claim,
            "fourth_mode_identifiable": False,
            "identification_authorized": False,
            "p43_authorized": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    np.savez_compressed(
        output / "path_convergence.npz",
        end_fractions_L3=np.asarray([step.end_fraction for step in l3_path]),
        forward_L3=l3["forward_observed"],
        jacobian_L3=l3["jacobian"],
        jacobian_L2=l2["jacobian"],
        forward_L2=l2["forward_observed"],
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].semilogy(
        ["L2", "L3"],
        [l2["geometry"]["normalized_singular_values"][i] for i in range(4)],
        "o",
        label="L2",
    )
    axes[0].semilogy(
        ["L2", "L3"],
        [l3["geometry"]["normalized_singular_values"][i] for i in range(4)],
        "x",
        label="L3",
    )
    axes[0].set(xlabel="level", ylabel="sigma / sigma1")
    axes[0].legend()
    axes[1].bar(["forward", "tau0", "R", "Q", "b"], [
        comparison["forward_observed_relative_l2"],
        *comparison["column_relative_l2"],
    ])
    axes[1].axhline(0.02, color="tab:red", linestyle="--")
    axes[1].set(ylabel="relative change")
    figure.savefig(output / "path_convergence_extension.png", dpi=180)
    plt.close(figure)
    print(
        json.dumps({"claims": report["claims"], "comparison": comparison}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
