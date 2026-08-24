#!/usr/bin/env python3
"""Qualify convergence of direct FEMU sensitivities under path refinement."""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import subspace_angles

from fem_inhouse.identification.dic_whitening import DICSpectralTransfer
from fem_inhouse.spectral2d.newton_two_state import TwoStateIncrementFields
from fem_inhouse.spectral2d.step_control import LoadPathStep
from scripts.qualify_srix_femu_common_path_gate import _common_path
from scripts.qualify_srix_femu_direct_sensitivity import (
    FD_STEP,
    ROOT,
    _direct_jacobian,
    _geometry,
    _oracle_config,
)
from scripts.qualify_srix_femu_fixed_path_gate import _fixed_path_trajectory
from scripts.qualify_srix_regm_transfer_noise import _WrapFreeTransfer
from scripts.qualify_srix_regm_twin import _orientation_map, _theta_from_preset

TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
SOURCE = ROOT / "validation/reference_data/srix_femu_common_path_gate_v9"
DEFAULT_OUTPUT = ROOT / "validation/reference_data/srix_femu_path_convergence_v1"


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _refine_path(path: list[LoadPathStep]) -> list[LoadPathStep]:
    refined: list[LoadPathStep] = []
    previous = np.zeros_like(path[0].boundary)
    for step in path:
        for part in range(2):
            left = part / 2.0
            right = (part + 1) / 2.0
            refined.append(
                LoadPathStep(
                    index=len(refined) + 1,
                    start_fraction=step.start_fraction
                    + left * (step.end_fraction - step.start_fraction),
                    end_fraction=step.start_fraction
                    + right * (step.end_fraction - step.start_fraction),
                    boundary=previous + right * (step.boundary - previous),
                    time_increment=step.time_increment / 2.0,
                )
            )
        previous = np.asarray(step.boundary, dtype=np.float64)
    return refined


def _nearest_indices(
    fields: list[TwoStateIncrementFields], targets: list[float]
) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(
            int(np.argmin([abs(field.end_fraction - target) for field in fields])) + 1
            for target in targets
        )
    )


def _observed_forward(
    fields: list[TwoStateIncrementFields],
    scored: tuple[int, ...],
    transfer: Any,
) -> np.ndarray:
    return np.concatenate(
        [np.asarray(transfer.apply(fields[index - 1].displacement)).reshape(-1) for index in scored]
    )


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-30))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--library", default="build/mfront/src/libBehaviour.so")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    run_sha = _git("rev-parse HEAD")
    run_dirty = bool(_git("status --porcelain"))
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    path_data = np.load(SOURCE / "common_path.npz")
    base_path = _common_path(path_data["end_fractions"].tolist(), pixels=args.pixels)
    targets = json.loads((SOURCE / "report.json").read_text())[
        "target_fractions_normalized_from_archived_indices"
    ]
    transfer = _WrapFreeTransfer(DICSpectralTransfer.from_sinusoidal_csv(TRANSFER))
    theta = _theta_from_preset()
    levels: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    paths: list[list[LoadPathStep]] = [base_path]
    paths.append(_refine_path(paths[-1]))
    paths.append(_refine_path(paths[-1]))
    for path in paths:
        try:
            fields = _fixed_path_trajectory(
                theta=theta,
                path=path,
                initial_displacement=np.zeros_like(path[0].boundary),
                pixels=args.pixels,
                library=args.library,
                threads=args.threads,
                config=_oracle_config(),
            )
        except RuntimeError as error:
            failure = {
                "steps": len(path),
                "error": str(error),
            }
            break
        scored = _nearest_indices(fields, targets)
        direct, timing = _direct_jacobian(
            fields=fields,
            scored=scored,
            orientations=_orientation_map(args.pixels),
            theta=theta,
            library=args.library,
            threads=args.threads,
            transfer=transfer,
            h=FD_STEP,
        )
        geometry = _geometry(direct)
        levels.append(
            {
                "steps": len(path),
                "end_fractions": [step.end_fraction for step in path],
                "fields": fields,
                "scored": scored,
                "forward_observed": _observed_forward(fields, scored, transfer),
                "jacobian": direct,
                "geometry": geometry,
                "timing": timing,
            }
        )

    comparisons: list[dict[str, Any]] = []
    for coarse, fine in itertools.pairwise(levels):
        errors = []
        cosines = []
        for column in range(4):
            left = coarse["jacobian"][:, column]
            right = fine["jacobian"][:, column]
            errors.append(_relative(left, right))
            cosines.append(
                float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
            )
        angles = np.degrees(
            subspace_angles(
                coarse["jacobian"][:, :3],
                fine["jacobian"][:, :3],
            )
        ).tolist()
        comparisons.append(
            {
                "coarse_steps": coarse["steps"],
                "fine_steps": fine["steps"],
                "forward_observed_relative_l2": _relative(
                    coarse["forward_observed"], fine["forward_observed"]
                ),
                "column_relative_l2": errors,
                "column_cosines": cosines,
                "rank3_principal_angles_degrees": angles,
                "singular_value_ratios": {
                    "coarse": coarse["geometry"]["normalized_singular_values"],
                    "fine": fine["geometry"]["normalized_singular_values"],
                },
            }
        )

    primary = comparisons[-1] if comparisons else None
    primary_claim = (
        failure is None
        and primary is not None
        and primary["forward_observed_relative_l2"] < 5.0e-3
        and all(value < 2.0e-2 for value in primary["column_relative_l2"][:3])
        and all(value > 0.999 for value in primary["column_cosines"][:3])
        and max(primary["rank3_principal_angles_degrees"]) < 2.0
    )
    report = {
        "schema_version": 1,
        "method": "direct FEMU path-discretization convergence",
        "git_sha": run_sha,
        "dirty": run_dirty,
        "machine": platform.node(),
        "pixels": args.pixels,
        "threads": args.threads,
        "fd_step_log": FD_STEP,
        "source_common_path": str(SOURCE / "common_path.npz"),
        "scored_target_fractions": targets,
        "levels": [
            {
                "steps": level["steps"],
                "scored": list(level["scored"]),
                "timing": level["timing"],
                "normalized_singular_values": level["geometry"]["normalized_singular_values"],
                "condition_number": level["geometry"]["condition_number"],
            }
            for level in levels
        ],
        "comparisons": comparisons,
        "failure": failure,
        "status": "blocked_path_level" if failure is not None else "converged",
        "claims": {
            "path_convergence_primary_gate": primary_claim,
            "fourth_mode_identifiable": False,
            "identification_authorized": False,
            "p43_authorized": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    arrays: dict[str, np.ndarray] = {}
    for level in levels:
        key = str(level["steps"])
        arrays[f"forward_{key}"] = level["forward_observed"]
        arrays[f"jacobian_{key}"] = level["jacobian"]
    np.savez_compressed(output / "path_convergence.npz", **arrays)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    steps = [level["steps"] for level in levels]
    spectra = [level["geometry"]["normalized_singular_values"] for level in levels]
    axes[0].plot(steps, [level["geometry"]["condition_number"] for level in levels], "o-")
    axes[0].set(xlabel="path steps", ylabel="condition number", yscale="log")
    for index in range(4):
        axes[1].plot(
            steps,
            [spectrum[index] for spectrum in spectra],
            "o-",
            label=f"sigma{index + 1}/sigma1",
        )
    axes[1].set(xlabel="path steps", ylabel="normalized singular value", yscale="log")
    axes[1].legend(fontsize=8)
    axes[2].plot(
        steps[1:],
        [comparison["forward_observed_relative_l2"] for comparison in comparisons],
        "o-",
    )
    axes[2].set(xlabel="fine path steps", ylabel="forward change", yscale="log")
    fig.tight_layout()
    fig.savefig(output / "path_convergence.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"claims": report["claims"], "comparisons": comparisons}, sort_keys=True))


if __name__ == "__main__":
    main()
