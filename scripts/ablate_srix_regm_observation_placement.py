#!/usr/bin/env python3
"""A/B/C/D ablation of observation placement in the SRIX-REGM surrogate."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from fem_inhouse.identification.dic_whitening import DICSpectralTransfer
from fem_inhouse.identification.srix_equilibrium_gap import (
    SrixEquilibriumGapProblem,
    SrixTheta4,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from scripts.compare_srix_regm_femu import _population, _statistics
from scripts.qualify_srix_regm_transfer_noise import (
    TRANSFER,
    _Identity,
    _operator,
    _WrapFreeTransfer,
)
from scripts.qualify_srix_regm_twin import (
    PIXEL_SIZE_MM,
    _material_factory,
    _theta_from_preset,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "validation/reference_data/srix_regm_twin_v1"
EXACT_RANKING = ROOT / "validation/reference_data/srix_regm_femu_ranking_v1/report.json"
OBSERVED_RANKING = ROOT / "validation/reference_data/srix_regm_femu_observed_ranking_v1/report.json"
OUTPUT = ROOT / "validation/reference_data/srix_regm_observation_placement_v1"


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _problem(
    history: np.ndarray,
    transfer: Any,
    *,
    orientations: np.ndarray,
    increments: np.ndarray,
    scored: tuple[int, ...],
    library: str,
    threads: int,
) -> SrixEquilibriumGapProblem:
    pixels = history.shape[1] - 1
    grid = StructuredGrid2D(
        pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels
    )
    return SrixEquilibriumGapProblem(
        operator=_operator(grid, orientations, transfer, _Identity()),
        displacement_history=history,
        state_indices=tuple(range(1, len(history))),
        scored_states=set(scored),
        material_factory=_material_factory(
            pixels=pixels, orientations=orientations, library=library, threads=threads
        ),
        time_increments=increments,
        debug=False,
    )


def _pseudo_trajectory_rms(evaluation: Any, node_shape: tuple[int, int]) -> float:
    dof = float(np.prod(node_shape) * 2)
    values = np.asarray(
        [
            state.pseudo_displacement_norm / np.sqrt(dof)
            for state in evaluation.states
            if state.scored
        ],
        dtype=np.float64,
    )
    return float(np.sqrt(np.mean(values**2)))


def _record(
    identifier: str,
    theta: SrixTheta4,
    evaluation: Any,
    femu_by_id: dict[str, float],
    femu_seconds_by_id: dict[str, float],
    node_shape: tuple[int, int],
) -> dict[str, Any]:
    return {
        "id": identifier,
        "theta": theta.as_runtime_overrides(),
        "status": "complete",
        "regm_rms_mm": evaluation.residual_rms,
        "regm_seconds": evaluation.timing.total_seconds,
        "femu_rms_mm": femu_by_id[identifier],
        "femu_seconds": femu_seconds_by_id[identifier],
        "pseudo_trajectory_rms_mm": _pseudo_trajectory_rms(evaluation, node_shape),
    }


def _plot(variants: dict[str, dict[str, Any]], output: Path) -> None:
    names = list(variants)
    correlations = [variants[name]["statistics"].get("spearman", np.nan) for name in names]
    bias = variants["C"]["truth_probe"]["pseudo_trajectory_rms_mm"]
    spread = variants["C"]["diagnostics"]["candidate_spread_rms_mm"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].bar(np.arange(len(names)), correlations, color="tab:blue")
    axes[0].axhline(0.80, color="tab:red", linestyle="--", linewidth=1)
    axes[0].set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Spearman(REGM, observed FEMU)")
    axes[0].set_title("Ranking after observation-placement ablation")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar((0, 1), (bias, spread), color=("tab:red", "tab:gray"))
    axes[1].set_xticks((0, 1), ("truth bias C", "candidate spread C"))
    axes[1].set_ylabel("pseudo-displacement RMS (mm)")
    axes[1].set_title(f"C truth bias / spread = {bias / max(spread, 1e-300):.2f}")
    axes[1].grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"srix_regm_observation_placement.{suffix}", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--library", default="build/mfront/src/libBehaviour.so")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_report = json.loads((SOURCE / "report.json").read_text())
    exact_report = json.loads(EXACT_RANKING.read_text())
    observed_report = json.loads(OBSERVED_RANKING.read_text())
    fields = np.load(SOURCE / "fields.npz", mmap_mode="r")
    raw = np.asarray(fields["displacement_history"], dtype=np.float64)
    orientations = np.asarray(fields["orientations_deg"], dtype=np.float64)
    increments = np.asarray(source_report["time_increments"], dtype=np.float64)
    scored = tuple(int(value) for value in source_report["states_scored"])
    spatial = DICSpectralTransfer.from_sinusoidal_csv(TRANSFER)
    affine = _WrapFreeTransfer(spatial)
    periodic = spatial
    transferred = np.asarray([affine.apply(state) for state in raw])
    population = {identifier: theta for identifier, theta, _ in _population()}
    exact_by_id = {item["id"]: item for item in exact_report["candidates"]}
    observed_by_id = {item["id"]: item for item in observed_report["candidates"]}
    femu = {
        identifier: float(item["levels"]["T1_transfer"]["femu_rms_mm"])
        for identifier, item in observed_by_id.items()
    }
    femu_seconds = {
        identifier: float(item["femu_seconds"])
        for identifier, item in observed_by_id.items()
    }
    identifiers = list(population)

    definitions: dict[str, tuple[np.ndarray, Any, str]] = {
        "B-periodic": (raw, periodic, "raw_input/periodic_score"),
        "B-affine": (raw, affine, "raw_input/affine_score"),
        "C": (transferred, _Identity(), "affine_input/identity_score"),
        "D-periodic": (transferred, periodic, "affine_input/periodic_score"),
        "D-affine": (transferred, affine, "affine_input/affine_score"),
    }
    variants: dict[str, dict[str, Any]] = {
        "A-archived": {
            "description": "raw_input/identity_score",
            "records": [
                {
                    "id": identifier,
                    "theta": population[identifier].as_runtime_overrides(),
                    "status": item["status"],
                    "regm_rms_mm": float(item["regm_rms_mm"]),
                    "regm_seconds": float(item["regm_seconds"]),
                    "femu_rms_mm": femu[identifier],
                    "femu_seconds": float(observed_by_id[identifier]["femu_seconds"]),
                }
                for identifier, item in exact_by_id.items()
            ],
        }
    }
    variants["A-archived"]["statistics"] = _statistics(variants["A-archived"]["records"])

    for name, (history, transfer, description) in definitions.items():
        problem = _problem(
            history,
            transfer,
            orientations=orientations,
            increments=increments,
            scored=scored,
            library=args.library,
            threads=args.threads,
        )
        records = []
        started_variant = time.perf_counter()
        for index, identifier in enumerate(identifiers, start=1):
            evaluation = problem.evaluate(population[identifier])
            records.append(
                _record(
                    identifier,
                    population[identifier],
                    evaluation,
                    femu,
                    femu_seconds,
                    history.shape[1:3],
                )
            )
            print(f"{name} [{index:02d}/{len(identifiers)}] {identifier}", flush=True)
        variant = {
            "description": description,
            "records": records,
            "statistics": _statistics(records),
            "seconds": time.perf_counter() - started_variant,
        }
        variants[name] = variant

    true_theta = _theta_from_preset()
    c_problem = _problem(
        transferred,
        _Identity(),
        orientations=orientations,
        increments=increments,
        scored=scored,
        library=args.library,
        threads=args.threads,
    )
    truth_c = c_problem.evaluate(true_theta)
    c_values = np.asarray(
        [
            [
                state.pseudo_displacement_norm / np.sqrt(np.prod(transferred.shape[1:3]) * 2)
                for state in c_problem.evaluate(population[identifier]).states
                if state.scored
            ]
            for identifier in identifiers
        ],
        dtype=np.float64,
    )
    truth_values = np.asarray(
        [
            state.pseudo_displacement_norm / np.sqrt(np.prod(transferred.shape[1:3]) * 2)
            for state in truth_c.states
            if state.scored
        ],
        dtype=np.float64,
    )
    variants["C"]["truth_probe"] = {
        "pseudo_trajectory_rms_mm": float(np.sqrt(np.mean(truth_values**2))),
        "residual_rms": truth_c.residual_rms,
    }
    variants["C"]["diagnostics"] = {
        "candidate_spread_rms_mm": float(np.sqrt(np.mean(np.std(c_values, axis=0) ** 2))),
        "bias_to_spread_ratio": float(
            np.sqrt(np.mean(truth_values**2))
            / max(float(np.sqrt(np.mean(np.std(c_values, axis=0) ** 2))), 1e-300)
        ),
    }

    report = {
        "schema_version": 1,
        "method": "SRIX-REGM observation-placement A/B/C/D ablation",
        "date": "2026-08-24",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "source_git_sha": source_report["git_sha"],
        "exact_ranking_source": str(EXACT_RANKING.relative_to(ROOT)),
        "observed_femu_target": str(OBSERVED_RANKING.relative_to(ROOT)),
        "transfer_csv": str(TRANSFER.relative_to(ROOT)),
        "input_history": "raw exact twin or affine-preserving transfer of raw exact twin",
        "femu_target_level": "T1_transfer",
        "variants": variants,
        "claims": {
            "new_mechanics_launched": False,
            "p43_identification_authorized": False,
            "truth_bias_probe_available": True,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _plot(variants, output)
    print(json.dumps({name: value["statistics"] for name, value in variants.items()}, indent=2))


if __name__ == "__main__":
    main()
