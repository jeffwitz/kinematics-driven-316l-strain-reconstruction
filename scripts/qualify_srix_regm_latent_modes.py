#!/usr/bin/env python3
"""Twin-only rank study for missing kinematic modes before SRIX replay."""

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
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from scripts.compare_srix_regm_femu import _population, _statistics
from scripts.qualify_srix_regm_transfer_noise import (
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
OBSERVED_RANKING = ROOT / "validation/reference_data/srix_regm_femu_observed_ranking_v1/report.json"
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
OUTPUT = ROOT / "validation/reference_data/srix_regm_latent_modes_v1"


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _pod_missing_modes(
    raw: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    difference = (raw - observed).reshape(raw.shape[0], -1)
    left, singular, right_transposed = np.linalg.svd(difference, full_matrices=False)
    return left, singular, right_transposed


def _rank_history(
    observed: np.ndarray,
    left: np.ndarray,
    singular: np.ndarray,
    right_transposed: np.ndarray,
    rank: int,
) -> np.ndarray:
    if rank < 0 or rank > len(singular):
        raise ValueError("rank is outside the available POD basis")
    reconstruction = (left[:, :rank] * singular[:rank]) @ right_transposed[:rank]
    return observed + reconstruction.reshape(observed.shape)


def _problem(
    history: np.ndarray,
    transfer: Any,
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


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    ranks = np.asarray([row["rank"] for row in rows])
    spearman = np.asarray([row["statistics"].get("spearman", np.nan) for row in rows])
    truth = np.asarray([row["truth_residual_rms"] for row in rows])
    energy = np.asarray([row["cumulative_energy"] for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    axes[0].plot(ranks, spearman, marker="o")
    axes[0].axhline(0.8, color="tab:red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("latent POD rank k")
    axes[0].set_ylabel("Spearman(REGM, observed FEMU)")
    axes[0].grid(alpha=0.25)
    axes[1].semilogy(ranks, truth, marker="o")
    axes[1].set_xlabel("latent POD rank k")
    axes[1].set_ylabel("truth REGM RMS (mm)")
    axes[1].grid(alpha=0.25)
    axes[2].plot(ranks, energy, marker="o")
    axes[2].set_xlabel("latent POD rank k")
    axes[2].set_ylabel("cumulative missing-field energy")
    axes[2].set_ylim(0, 1.02)
    axes[2].grid(alpha=0.25)
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"srix_regm_latent_modes.{suffix}", dpi=180)
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
    observed_report = json.loads(OBSERVED_RANKING.read_text())
    fields = np.load(SOURCE / "fields.npz", mmap_mode="r")
    raw = np.asarray(fields["displacement_history"], dtype=np.float64)
    orientations = np.asarray(fields["orientations_deg"], dtype=np.float64)
    increments = np.asarray(source_report["time_increments"], dtype=np.float64)
    scored = tuple(int(value) for value in source_report["states_scored"])
    transfer = _WrapFreeTransfer(DICSpectralTransfer.from_sinusoidal_csv(TRANSFER))
    observed = np.asarray([transfer.apply(state) for state in raw])
    left, singular, right_transposed = _pod_missing_modes(raw, observed)
    population = {identifier: theta for identifier, theta, _ in _population()}
    observed_by_id = {item["id"]: item for item in observed_report["candidates"]}
    femu = {
        identifier: float(item["levels"]["T1_transfer"]["femu_rms_mm"])
        for identifier, item in observed_by_id.items()
    }
    transfer_energy = np.cumsum(singular**2) / max(float(np.sum(singular**2)), 1e-300)
    ranks = tuple(sorted(set((0, 1, 2, 3, 4, 5, len(singular)))))
    rows: list[dict[str, Any]] = []
    for rank in ranks:
        history = _rank_history(observed, left, singular, right_transposed, rank)
        problem = _problem(
            history, transfer, orientations, increments, scored, args.library, args.threads
        )
        truth = problem.evaluate(_theta_from_preset())
        candidates = []
        started = time.perf_counter()
        for identifier, theta in population.items():
            evaluation = problem.evaluate(theta)
            candidates.append(
                {
                    "id": identifier,
                    "status": "complete",
                    "theta": theta.as_runtime_overrides(),
                    "regm_rms_mm": evaluation.residual_rms,
                    "regm_seconds": evaluation.timing.total_seconds,
                    "femu_rms_mm": femu[identifier],
                    "femu_seconds": float(observed_by_id[identifier]["femu_seconds"]),
                }
            )
        rows.append(
            {
                "rank": rank,
                "cumulative_energy": 0.0 if rank == 0 else float(transfer_energy[rank - 1]),
                "missing_history_rms_mm": float(np.sqrt(np.mean((raw - history) ** 2))),
                "truth_residual_rms": truth.residual_rms,
                "statistics": _statistics(candidates),
                "seconds": time.perf_counter() - started,
                "candidates": candidates,
            }
        )
        print(
            f"rank={rank}  energy={rows[-1]['cumulative_energy']:.6f}  "
            f"truth={truth.residual_rms:.3e}  "
            f"spearman={rows[-1]['statistics'].get('spearman', float('nan')):.3f}",
            flush=True,
        )
    report = {
        "schema_version": 1,
        "method": "twin-only POD reconstruction of kinematic modes lost by DIC transfer",
        "date": "2026-08-24",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "source_git_sha": source_report["git_sha"],
        "observed_femu_target": str(OBSERVED_RANKING.relative_to(ROOT)),
        "transfer_csv": str(TRANSFER.relative_to(ROOT)),
        "singular_values": singular.tolist(),
        "ranks": list(ranks),
        "rows": rows,
        "claims": {"new_mechanics_launched": False, "p43_authorized": False},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output / "latent_modes.npz",
        left=left,
        singular_values=singular,
        right_transposed=right_transposed,
    )
    _plot(rows, output)


if __name__ == "__main__":
    main()
