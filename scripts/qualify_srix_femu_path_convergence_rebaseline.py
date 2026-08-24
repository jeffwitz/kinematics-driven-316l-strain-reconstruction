#!/usr/bin/env python3
"""Qualify nested FEMU path refinement after the corrected Dirichlet baseline."""

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
    _path_search_config,
)
from scripts.qualify_srix_femu_fixed_path_gate import _fixed_path_trajectory
from scripts.qualify_srix_regm_transfer_noise import _WrapFreeTransfer
from scripts.qualify_srix_regm_twin import _orientation_map, _theta_from_preset

SOURCE = ROOT / "validation/reference_data/srix_femu_common_path_gate_v17"
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
DEFAULT_OUTPUT = ROOT / "validation/reference_data/srix_femu_path_convergence_v3"
PIXEL_SIZE_MM = 1.84e-3
MAX_LOCAL_REPAIRS = 128
PARAMETER_NAMES = ("log(tau0)", "log(R)", "log(Q)", "log(b)")


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _mandatory_refine(path: list[LoadPathStep], *, pixels: int) -> list[LoadPathStep]:
    ends = [step.end_fraction for step in path]
    mids = [0.5 * (left + right) for left, right in itertools.pairwise([0.0, *ends])]
    return _common_path([*ends, *mids], pixels=pixels)


def _repair_base_path(
    path: list[LoadPathStep],
    *,
    pixels: int,
    library: str,
    threads: int,
    max_repairs: int,
) -> tuple[list[LoadPathStep], list[TwoStateIncrementFields], list[dict[str, Any]]]:
    repairs: list[dict[str, Any]] = []
    search_config = _path_search_config()

    def run(config: Any) -> list[TwoStateIncrementFields]:
        return _fixed_path_trajectory(
            theta=_theta_from_preset(),
            path=path,
            initial_displacement=None,
            pixels=pixels,
            library=library,
            threads=threads,
            config=config,
        )

    while True:
        try:
            # Search only with the fail-fast policy.  The strict oracle is
            # applied below after the partition no longer needs repairs.
            run(search_config)
            strict_fields = run(_oracle_config())
            return path, strict_fields, repairs
        except RuntimeError as error:
            text = str(error)
            marker = "increment "
            if marker not in text:
                raise
            increment = int(text.split(marker, 1)[1].split()[0])
            if not 1 <= increment <= len(path):
                raise RuntimeError(f"invalid failed increment in {text!r}") from error
            if len(repairs) >= max_repairs:
                raise RuntimeError(
                    f"local repair budget exhausted after {len(repairs)} repairs: {text}"
                ) from error
            failed = path[increment - 1]
            left, right = failed.start_fraction, failed.end_fraction
            width = right - left
            if width <= 1.0 / 65536.0:
                raise RuntimeError(f"minimum repair interval reached: {text}") from error
            midpoint = 0.5 * (left + right)
            repairs.append(
                {
                    "failed_increment": increment,
                    "start_fraction": left,
                    "end_fraction": right,
                    "inserted_fraction": midpoint,
                    "steps_before": len(path),
                    "failure": text,
                }
            )
            ends = [step.end_fraction for step in path]
            path = _common_path(
                [*ends[: increment - 1], midpoint, *ends[increment - 1 :]], pixels=pixels
            )
            print(
                f"  local repair {len(repairs)}: {left:.9f}--{right:.9f} -> "
                f"{midpoint:.9f} ({text})",
                flush=True,
            )


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
    fields: list[TwoStateIncrementFields], scored: tuple[int, ...], transfer: Any
) -> np.ndarray:
    return np.concatenate(
        [np.asarray(transfer.apply(fields[index - 1].displacement)).reshape(-1) for index in scored]
    )


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-30))


def _geometry_with_contrast(matrix: np.ndarray) -> dict[str, Any]:
    geometry = _geometry(matrix)
    vectors = np.asarray(geometry["right_singular_vectors"], dtype=np.float64)
    contrast = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float64)
    contrast /= np.linalg.norm(contrast)
    geometry["q_minus_b_contrast_alignment_v4"] = float(abs(np.dot(vectors[:, 3], contrast)))
    return geometry


def _compare(coarse: dict[str, Any], fine: dict[str, Any]) -> dict[str, Any]:
    errors: list[float] = []
    cosines: list[float] = []
    for index in range(4):
        left = coarse["jacobian"][:, index]
        right = fine["jacobian"][:, index]
        errors.append(_relative(left, right))
        cosines.append(float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))))
    angles = np.degrees(
        subspace_angles(coarse["jacobian"][:, :3], fine["jacobian"][:, :3])
    ).tolist()
    coarse_spectrum = np.asarray(coarse["geometry"]["normalized_singular_values"])
    fine_spectrum = np.asarray(fine["geometry"]["normalized_singular_values"])
    return {
        "coarse_steps": coarse["steps"],
        "fine_steps": fine["steps"],
        "forward_observed_relative_l2": _relative(
            coarse["forward_observed"], fine["forward_observed"]
        ),
        "column_relative_l2": errors,
        "column_cosines": cosines,
        "rank3_principal_angles_degrees": angles,
        "normalized_singular_values": {
            "coarse": coarse_spectrum.tolist(),
            "fine": fine_spectrum.tolist(),
            "absolute_change": np.abs(fine_spectrum - coarse_spectrum).tolist(),
            "relative_change_first_three": (
                np.abs(fine_spectrum[:3] - coarse_spectrum[:3]) / fine_spectrum[:3]
            ).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--library", default="build/mfront/src/libBehaviour.so")
    parser.add_argument("--max-local-repairs", type=int, default=MAX_LOCAL_REPAIRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_report = json.loads((SOURCE / "report.json").read_text())
    archived_targets = source_report["target_fractions_normalized_from_archived_indices"]
    source_fractions = np.asarray(np.load(SOURCE / "common_path.npz")["end_fractions"])
    # Two archived target fractions map to the same L0 endpoint.  Score the
    # physical fractions actually represented by L0; nested levels preserve
    # every L0 endpoint exactly.
    targets = list(
        dict.fromkeys(
            float(source_fractions[int(np.argmin(np.abs(source_fractions - target)))])
            for target in archived_targets
        )
    )
    transfer = _WrapFreeTransfer(DICSpectralTransfer.from_sinusoidal_csv(TRANSFER))
    levels: list[dict[str, Any]] = []
    path = _common_path(source_fractions.tolist(), pixels=args.pixels)
    for level_index in range(3):
        if level_index > 0:
            path = _mandatory_refine(path, pixels=args.pixels)
        label = f"L{level_index}"
        print(f"{label}: mandatory path with {len(path)} steps", flush=True)
        repair_started = time.perf_counter()
        path, fields, repairs = _repair_base_path(
            path,
            pixels=args.pixels,
            library=args.library,
            threads=args.threads,
            max_repairs=args.max_local_repairs,
        )
        print(
            f"{label}: converged with {len(path)} steps and {len(repairs)} local repairs",
            flush=True,
        )
        scored = _nearest_indices(fields, targets)
        direct, timing = _direct_jacobian(
            fields=fields,
            scored=scored,
            orientations=_orientation_map(args.pixels),
            theta=_theta_from_preset(),
            library=args.library,
            threads=args.threads,
            transfer=transfer,
            h=FD_STEP,
        )
        levels.append(
            {
                "label": label,
                "steps": len(path),
                "path": path,
                "fields": fields,
                "scored": scored,
                "forward_observed": _observed_forward(fields, scored, transfer),
                "jacobian": direct,
                "geometry": _geometry_with_contrast(direct),
                "timing": {
                    "repair_and_forward_seconds": time.perf_counter() - repair_started,
                    "direct_sensitivity": timing,
                },
                "mandatory_refinement_level": level_index,
                "local_repairs": repairs,
            }
        )
    comparisons = [_compare(left, right) for left, right in itertools.pairwise(levels)]
    primary = comparisons[-1]
    primary_claim = (
        primary["forward_observed_relative_l2"] < 5.0e-3
        and all(value < 2.0e-2 for value in primary["column_relative_l2"][:3])
        and all(value > 0.999 for value in primary["column_cosines"][:3])
        and max(primary["rank3_principal_angles_degrees"]) < 2.0
        and max(primary["normalized_singular_values"]["relative_change_first_three"]) < 0.05
    )
    report = {
        "schema_version": 1,
        "method": (
            "nested direct FEMU path-discretization convergence after corrected "
            "Dirichlet baseline"
        ),
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "pixels": args.pixels,
        "threads": args.threads,
        "source_common_path": str(SOURCE / "common_path.npz"),
        "fd_step_log": FD_STEP,
        "archived_target_fractions": archived_targets,
        "scored_target_fractions": targets,
        "levels": [
            {
                "label": level["label"],
                "nominal_refinement_level": level["mandatory_refinement_level"],
                "steps": level["steps"],
                "scored": list(level["scored"]),
                "actual_local_repair_count": len(level["local_repairs"]),
                "maximum_interval_width": max(
                    step.end_fraction - step.start_fraction for step in level["path"]
                ),
                "minimum_interval_width": min(
                    step.end_fraction - step.start_fraction for step in level["path"]
                ),
                "timing": level["timing"],
                "normalized_singular_values": level["geometry"]["normalized_singular_values"],
                "condition_number": level["geometry"]["condition_number"],
                "q_minus_b_contrast_alignment_v4": level["geometry"][
                    "q_minus_b_contrast_alignment_v4"
                ],
                "local_repairs": level["local_repairs"],
            }
            for level in levels
        ],
        "comparisons": comparisons,
        "primary_comparison": "L1_to_L2",
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
        arrays[f"forward_{level['label']}"] = level["forward_observed"]
        arrays[f"jacobian_{level['label']}"] = level["jacobian"]
        arrays[f"end_fractions_{level['label']}"] = np.asarray(
            [step.end_fraction for step in level["path"]]
        )
    np.savez_compressed(output / "path_convergence.npz", **arrays)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    labels = [level["label"] for level in levels]
    axes[0].plot(labels, [level["steps"] for level in levels], "o-")
    axes[0].set(xlabel="level", ylabel="actual path steps")
    spectra = [level["geometry"]["normalized_singular_values"] for level in levels]
    for index in range(4):
        axes[1].semilogy(
            labels,
            [spectrum[index] for spectrum in spectra],
            "o-",
            label=f"sigma{index + 1}",
        )
    axes[1].set(xlabel="level", ylabel="sigma / sigma1")
    axes[1].legend(fontsize=8)
    axes[2].semilogy(
        ["L0-L1", "L1-L2"],
        [comparison["forward_observed_relative_l2"] for comparison in comparisons],
        "o-",
    )
    axes[2].set(xlabel="comparison", ylabel="observed forward change")
    figure.savefig(output / "path_convergence.png", dpi=180)
    plt.close(figure)
    print(
        json.dumps({"claims": report["claims"], "comparisons": comparisons}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
