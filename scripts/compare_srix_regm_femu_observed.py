#!/usr/bin/env python3
"""Compare REGM and FEMU rankings after the qualified DIC observation chain."""

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

from fem_inhouse.identification.dic_whitening import (
    DICSpectralTransfer,
    DICSpectralWhitener,
)
from fem_inhouse.identification.srix_equilibrium_gap import SrixEquilibriumGapProblem
from fem_inhouse.measurement import image_flow_to_canonical
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from scripts.compare_srix_regm_femu import (
    _atomic_json,
    _population,
    _statistics,
    _summarize_diagnostics,
)
from scripts.qualify_srix_regm_transfer_noise import (
    NOISE,
    TRANSFER,
    _Identity,
    _operator,
    _sample_noise,
    _WrapFreeTransfer,
)
from scripts.qualify_srix_regm_twin import (
    PIXEL_SIZE_MM,
    _generate_twin,
    _material_factory,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "validation/reference_data/srix_regm_twin_v1"
OUTPUT = ROOT / "validation/reference_data/srix_regm_femu_observed_ranking_v1"


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _problem(
    *,
    history: np.ndarray,
    time_increments: np.ndarray,
    scored_states: tuple[int, ...],
    orientations: np.ndarray,
    transfer: Any,
    whitener: Any,
    library: str,
    threads: int,
) -> SrixEquilibriumGapProblem:
    pixels = history.shape[1] - 1
    grid = StructuredGrid2D(
        pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels
    )
    return SrixEquilibriumGapProblem(
        operator=_operator(grid, orientations, transfer, whitener),
        displacement_history=history,
        state_indices=tuple(range(1, len(history))),
        scored_states=set(scored_states),
        material_factory=_material_factory(
            pixels=pixels,
            orientations=orientations,
            library=library,
            threads=threads,
        ),
        time_increments=time_increments,
        debug=False,
    )


def _observed_rms(
    candidate: np.ndarray,
    target: np.ndarray,
    whitener: Any,
) -> float:
    if candidate.shape != target.shape:
        raise ValueError("candidate and target observed fields disagree")
    residuals = np.asarray(
        [
            whitener.apply(candidate_state - target_state)
            for candidate_state, target_state in zip(candidate, target, strict=True)
        ]
    )
    return float(np.sqrt(np.mean(residuals[:, 1:-1, 1:-1, :] ** 2)))


def _level_rows(candidates: list[dict[str, Any]], level: str) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        values = candidate["levels"].get(level, {})
        row = {
            "id": candidate["id"],
            "status": candidate["status"],
            "regm_seconds": values.get("regm_seconds", np.nan),
            "femu_seconds": candidate.get("femu_seconds", np.nan),
        }
        if candidate["status"] == "complete":
            row.update(
                regm_rms_mm=values["regm_rms_mm"],
                femu_rms_mm=values["femu_rms_mm"],
            )
        rows.append(row)
    return rows


def _figures(candidates: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for axis, level in zip(axes, ("T1_transfer", "T2_transfer_noise"), strict=True):
        rows = [row for row in _level_rows(candidates, level) if row["status"] == "complete"]
        regm = np.asarray([row["regm_rms_mm"] for row in rows])
        femu = np.asarray([row["femu_rms_mm"] for row in rows])
        axis.scatter(regm, femu, color="tab:blue")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("observed REGM RMS")
        axis.set_ylabel("observed FEMU RMS")
        axis.set_title(level)
        axis.grid(True, which="both", alpha=0.25)
        for row, x_value, y_value in zip(rows, regm, femu, strict=True):
            axis.annotate(row["id"].replace("lhs_", ""), (x_value, y_value), fontsize=7)
    figure.suptitle("M8 SRIX ranking after the qualified DIC observation chain")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"srix_regm_femu_observed_ranking.{suffix}", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--library", default="build/mfront/src/libBehaviour.so")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "checkpoint.json"
    if any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(f"output is not empty: {args.output}; pass --resume")

    report0 = json.loads((SOURCE / "report.json").read_text())
    fields = np.load(SOURCE / "fields.npz", mmap_mode="r")
    raw = np.asarray(fields["displacement_history"], dtype=np.float64)
    orientations = np.asarray(fields["orientations_deg"], dtype=np.float64)
    increments = np.asarray(report0["time_increments"], dtype=np.float64)
    scored = tuple(int(value) for value in report0["states_scored"])
    spatial = DICSpectralTransfer.from_sinusoidal_csv(TRANSFER)
    transfer = _WrapFreeTransfer(spatial)
    transferred = np.asarray([transfer.apply(state) for state in raw])
    noise_pixels = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    noise_mm = image_flow_to_canonical(
        np.asarray(noise_pixels[:512, :512]), pixel_size_mm=PIXEL_SIZE_MM
    )
    support = np.ones((*raw.shape[1:3], 2), dtype=np.float64)
    support[[0, -1], :, :] = 0.0
    support[:, [0, -1], :] = 0.0
    whitener = DICSpectralWhitener.from_stationary_noise_field(
        noise_mm,
        target_shape=raw.shape[1:3],
        sample_count=256,
        seed=42,
        remove_spatial_mean=False,
        support_mask=support,
    )
    noisy = transferred + _sample_noise(noise_mm, increments, scored, raw.shape[1:3])
    problems = {
        "T1_transfer": _problem(
            history=transferred,
            time_increments=increments,
            scored_states=scored,
            orientations=orientations,
            transfer=transfer,
            whitener=_Identity(),
            library=args.library,
            threads=args.threads,
        ),
        "T2_transfer_noise": _problem(
            history=noisy,
            time_increments=increments,
            scored_states=scored,
            orientations=orientations,
            transfer=transfer,
            whitener=whitener,
            library=args.library,
            threads=args.threads,
        ),
    }
    target_scored = {
        "T1_transfer": transferred[np.asarray(scored)],
        "T2_transfer_noise": noisy[np.asarray(scored)],
    }

    existing: dict[str, dict[str, Any]] = {}
    if args.resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        existing = {item["id"]: item for item in checkpoint.get("candidates", [])}

    candidates: list[dict[str, Any]] = []
    for index, (candidate_id, theta, log_offset) in enumerate(_population(), start=1):
        if candidate_id in existing and existing[candidate_id].get("status") == "complete":
            candidates.append(existing[candidate_id])
            print(f"[{index:02d}/20] {candidate_id}: reused", flush=True)
            continue
        level_values: dict[str, dict[str, float]] = {}
        for name, problem in problems.items():
            started = time.perf_counter()
            evaluation = problem.evaluate(theta)
            level_values[name] = {
                "regm_rms_mm": evaluation.residual_rms,
                "regm_seconds": time.perf_counter() - started,
            }
        record: dict[str, Any] = {
            "id": candidate_id,
            "theta": theta.as_runtime_overrides(),
            "log_offset_from_truth": log_offset.tolist(),
            "levels": level_values,
        }
        try:
            history, _, candidate_states, diagnostics, femu_seconds = _generate_twin(
                pixels=8, library=args.library, threads=args.threads, theta=theta
            )
            candidate_macro = history[np.asarray(candidate_states)]
            candidate_observed = np.asarray(
                [transfer.apply(state) for state in candidate_macro]
            )
            level_values["T1_transfer"]["femu_rms_mm"] = _observed_rms(
                candidate_observed, target_scored["T1_transfer"], _Identity()
            )
            level_values["T2_transfer_noise"]["femu_rms_mm"] = _observed_rms(
                candidate_observed, target_scored["T2_transfer_noise"], whitener
            )
            record.update(
                status="complete",
                femu_seconds=femu_seconds,
                forward=_summarize_diagnostics(diagnostics),
            )
        except Exception as error:
            record.update(
                status="failed", error_type=type(error).__name__, error=str(error)
            )
        candidates.append(record)
        _atomic_json(
            checkpoint_path,
            {"schema_version": 1, "candidates": candidates},
        )
        print(f"[{index:02d}/20] {candidate_id}: {record['status']}", flush=True)

    statistics = {
        level: _statistics(_level_rows(candidates, level))
        for level in ("T1_transfer", "T2_transfer_noise")
    }
    overall_gate = all(values.get("gate_passed", False) for values in statistics.values())
    report = {
        "schema_version": 1,
        "method": "observed-space SRIX-REGM versus full-FEMU ranking",
        "date": "2026-08-23",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "source": str(SOURCE.relative_to(ROOT)),
        "transfer_csv": str(TRANSFER.relative_to(ROOT)),
        "noise_source": str(NOISE.relative_to(ROOT)),
        "threads": args.threads,
        "candidates": candidates,
        "statistics": statistics,
        "claims": {
            "both_observed_ranking_gates_passed": overall_gate,
            "p43_regm_identification_authorized": overall_gate,
        },
    }
    _atomic_json(args.output / "report.json", report)
    _figures(candidates, args.output)
    print(json.dumps({"statistics": statistics, "overall_gate": overall_gate}, indent=2))


if __name__ == "__main__":
    main()
