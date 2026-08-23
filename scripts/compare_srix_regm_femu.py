#!/usr/bin/env python3
"""Compare SRIX-REGM and full-FEMU rankings on the preregistered M8 twin."""

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
from scipy.stats import pearsonr, spearmanr

from fem_inhouse.identification.srix_equilibrium_gap import SrixTheta4
from scripts.qualify_srix_regm_twin import (
    _generate_twin,
    _problem,
    _theta_from_preset,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "validation/reference_data/srix_regm_twin_v1"
OUTPUT = ROOT / "validation/reference_data/srix_regm_femu_ranking_v1"
LOG_OFFSETS = np.asarray(
    (
        (-0.106535, +0.195647, +0.282258, +0.166001),
        (-0.268867, +0.266724, +0.229828, +0.207584),
        (-0.003049, +0.146794, -0.241087, -0.138832),
        (-0.048460, -0.099961, -0.045376, -0.018534),
        (+0.183792, +0.116277, +0.160055, +0.030313),
        (-0.178428, -0.023892, +0.266417, +0.100787),
        (+0.252891, -0.049726, -0.122052, +0.198030),
        (-0.030582, +0.290063, -0.050611, -0.095495),
        (-0.224258, -0.112246, -0.083937, -0.164764),
        (+0.079777, -0.277424, -0.224168, -0.212370),
        (-0.250035, +0.033006, +0.093018, -0.264816),
        (-0.169448, -0.207373, +0.036584, -0.291297),
        (+0.128381, +0.219309, +0.128248, +0.123126),
        (+0.226227, +0.066613, -0.168320, -0.007200),
        (+0.271877, -0.001076, -0.007494, +0.061936),
        (+0.058258, -0.253252, +0.196774, +0.297952),
        (-0.130702, +0.089323, +0.077862, -0.060408),
        (+0.028116, -0.172289, -0.199220, -0.191474),
        (+0.144311, -0.180116, -0.295437, +0.264668),
    ),
    dtype=np.float64,
)


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _population() -> list[tuple[str, SrixTheta4, np.ndarray]]:
    truth = _theta_from_preset()
    initial = SrixTheta4(
        tau0_mpa=1.25 * truth.tau0_mpa,
        r_mpa=0.80 * truth.r_mpa,
        q_mpa=1.30 * truth.q_mpa,
        b=0.75 * truth.b,
    )
    result = [("initial", initial, initial.log_coordinates() - truth.log_coordinates())]
    result.extend(
        (f"lhs_{index:02d}", SrixTheta4.from_log_coordinates(truth.log_coordinates() + row), row)
        for index, row in enumerate(LOG_OFFSETS, start=1)
    )
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _summarize_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    attempts = diagnostics.get("load_step_attempts", [])
    accepted = [item for item in attempts if item.get("accepted", False)]
    return {
        "accepted_steps": len(accepted),
        "attempts": len(attempts),
        "cutbacks": diagnostics.get("cutbacks"),
        "linear_solves": diagnostics.get("linear_solves"),
        "newton_iterations": int(sum(diagnostics.get("iterations_per_increment", []))),
        "maximum_plane_stress_residual_mpa": diagnostics.get(
            "maximum_plane_stress_residual_mpa"
        ),
        "timings": diagnostics.get("timings", {}),
    }


def _femu_rms(
    candidate: np.ndarray,
    candidate_states: tuple[int, ...],
    target: np.ndarray,
    target_states: tuple[int, ...],
) -> float:
    candidate_scored = candidate[np.asarray(candidate_states), 1:-1, 1:-1, :]
    target_scored = target[np.asarray(target_states), 1:-1, 1:-1, :]
    if candidate_scored.shape != target_scored.shape:
        raise ValueError("candidate and target macro-endpoint supports disagree")
    return float(np.sqrt(np.mean((candidate_scored - target_scored) ** 2)))


def _statistics(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in candidates if item.get("status") == "complete"]
    regm = np.asarray([item["regm_rms_mm"] for item in valid], dtype=np.float64)
    femu = np.asarray([item["femu_rms_mm"] for item in valid], dtype=np.float64)
    if len(valid) < 2 or np.any(regm <= 0.0) or np.any(femu <= 0.0):
        return {"valid_candidates": len(valid), "interpretable": False}
    pearson = pearsonr(np.log(regm), np.log(femu))
    spearman = spearmanr(regm, femu)
    best_regm = np.argsort(regm)[:5]
    best_femu = np.argsort(femu)[:5]
    overlap = sorted(
        {valid[index]["id"] for index in best_regm}
        & {valid[index]["id"] for index in best_femu}
    )
    regm_seconds = np.asarray([item["regm_seconds"] for item in valid])
    femu_seconds = np.asarray([item["femu_seconds"] for item in valid])
    gate = {
        "minimum_valid_candidates": len(valid) >= 15,
        "spearman": float(spearman.statistic) >= 0.80,
        "log_pearson": float(pearson.statistic) >= 0.70,
        "best_five_overlap": len(overlap) >= 3,
    }
    return {
        "valid_candidates": len(valid),
        "interpretable": len(valid) >= 15,
        "spearman": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
        "log_pearson": float(pearson.statistic),
        "log_pearson_pvalue": float(pearson.pvalue),
        "best_five_regm": [valid[index]["id"] for index in best_regm],
        "best_five_femu": [valid[index]["id"] for index in best_femu],
        "best_five_overlap": overlap,
        "best_five_overlap_count": len(overlap),
        "median_regm_seconds": float(np.median(regm_seconds)),
        "median_femu_seconds": float(np.median(femu_seconds)),
        "median_speedup": float(np.median(femu_seconds) / np.median(regm_seconds)),
        "gate_components": gate,
        "gate_passed": all(gate.values()),
    }


def _figures(candidates: list[dict[str, Any]], output: Path) -> None:
    valid = [item for item in candidates if item.get("status") == "complete"]
    regm = np.asarray([item["regm_rms_mm"] for item in valid])
    femu = np.asarray([item["femu_rms_mm"] for item in valid])
    order_regm = np.argsort(np.argsort(regm)) + 1
    order_femu = np.argsort(np.argsort(femu)) + 1
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    axes[0].scatter(regm, femu, color="tab:blue")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("SRIX-REGM RMS (mm)")
    axes[0].set_ylabel("Full-FEMU displacement RMS (mm)")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].scatter(order_regm, order_femu, color="tab:orange")
    axes[1].plot((1, len(valid)), (1, len(valid)), "k--", linewidth=1)
    axes[1].set_xlabel("REGM rank")
    axes[1].set_ylabel("FEMU rank")
    axes[1].grid(True, alpha=0.25)
    for item, x_value, y_value in zip(valid, regm, femu, strict=True):
        axes[0].annotate(item["id"].replace("lhs_", ""), (x_value, y_value), fontsize=7)
    figure.suptitle("Preregistered M8 SRIX parameter-ranking gate")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"srix_regm_femu_ranking.{suffix}", dpi=180)
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

    source_report = json.loads((SOURCE / "report.json").read_text())
    fields = np.load(SOURCE / "fields.npz", mmap_mode="r")
    target = np.asarray(fields["displacement_history"], dtype=np.float64)
    time_increments = np.asarray(source_report["time_increments"], dtype=np.float64)
    target_states = tuple(int(value) for value in source_report["states_scored"])
    problem = _problem(
        pixels=8,
        displacement_history=target,
        time_increments=time_increments,
        scored_states=target_states,
        library=args.library,
        threads=args.threads,
        debug=False,
    )

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
        regm_started = time.perf_counter()
        regm = problem.evaluate(theta)
        regm_seconds = time.perf_counter() - regm_started
        record: dict[str, Any] = {
            "id": candidate_id,
            "theta": theta.as_runtime_overrides(),
            "log_offset_from_truth": log_offset.tolist(),
            "regm_rms_mm": regm.residual_rms,
            "regm_seconds": regm_seconds,
            "regm_timing": {
                "material_seconds": regm.timing.material_seconds,
                "reconditioner_seconds": regm.timing.reconditioner_seconds,
                "observation_seconds": regm.timing.observation_seconds,
                "total_seconds": regm.timing.total_seconds,
            },
        }
        try:
            history, _, candidate_states, diagnostics, femu_seconds = _generate_twin(
                pixels=8, library=args.library, threads=args.threads, theta=theta
            )
            record.update(
                status="complete",
                femu_rms_mm=_femu_rms(history, candidate_states, target, target_states),
                femu_seconds=femu_seconds,
                forward=_summarize_diagnostics(diagnostics),
            )
        except Exception as error:
            record.update(
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
            )
        candidates.append(record)
        _atomic_json(
            checkpoint_path,
            {
                "schema_version": 1,
                "source_git_sha": source_report["git_sha"],
                "candidates": candidates,
            },
        )
        status = record["status"]
        femu_text = f" FEMU={record['femu_rms_mm']:.3e}" if status == "complete" else ""
        print(
            f"[{index:02d}/20] {candidate_id}: REGM={regm.residual_rms:.3e}{femu_text} {status}",
            flush=True,
        )

    statistics = _statistics(candidates)
    report = {
        "schema_version": 1,
        "method": "SRIX-REGM versus complete adaptive FEMU ranking",
        "date": "2026-08-23",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_git_sha": source_report["git_sha"],
        "threads": args.threads,
        "population": {
            "off_truth_candidates": 20,
            "latin_hypercube_seed": 20260824,
            "latin_hypercube_log_half_width": 0.30,
            "truth_control_excluded_from_correlations": True,
        },
        "truth_control": {
            "theta": _theta_from_preset().as_runtime_overrides(),
            "regm_rms_mm": source_report["evaluations"]["truth"]["residual_rms"],
            "femu_rms_mm": 0.0,
        },
        "objectives": {
            "regm": "RMS reconditioned pseudo-displacement over scored states",
            "femu": "RMS interior nodal displacement difference at eight macro endpoints",
        },
        "candidates": candidates,
        "statistics": statistics,
        "claims": {
            "ranking_gate_passed": statistics.get("gate_passed", False),
            "p43_authorized": statistics.get("gate_passed", False),
        },
    }
    _atomic_json(args.output / "report.json", report)
    _figures(candidates, args.output)
    print(json.dumps(statistics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
