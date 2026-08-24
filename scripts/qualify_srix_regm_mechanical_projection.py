#!/usr/bin/env python3
"""Test a damped mechanical projection before the causal SRIX replay.

This is a twin-only experiment.  It never launches a nonlinear forward solve
and never uses the projected history to identify parameters.
"""

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
from fem_inhouse.identification.srix_equilibrium_gap import SrixEquilibriumGapProblem
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from scripts.compare_srix_regm_femu import _population, _statistics
from scripts.qualify_srix_regm_transfer_noise import _Identity, _WrapFreeTransfer
from scripts.qualify_srix_regm_twin import (
    PIXEL_SIZE_MM,
    _material_factory,
    _point_elasticity,
    _theta_from_preset,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "validation/reference_data/srix_regm_twin_v1"
TARGET = ROOT / "validation/reference_data/srix_regm_femu_observed_ranking_v1/report.json"
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
DEFAULT_OUTPUT = ROOT / "validation/reference_data/srix_regm_mechanical_projection_v1"


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _operator(
    grid: StructuredGrid2D,
    orientations: np.ndarray,
    transfer: Any,
) -> TensorPlasticObservabilityOperator:
    return TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
        point_elasticity=_point_elasticity(orientations),
        transfer=transfer,
        whitener=_Identity(),
    )


def _problem(
    history: np.ndarray,
    orientations: np.ndarray,
    transfer: Any,
    increments: np.ndarray,
    scored: tuple[int, ...],
    library: str,
    threads: int,
    *,
    debug: bool,
) -> SrixEquilibriumGapProblem:
    pixels = history.shape[1] - 1
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    return SrixEquilibriumGapProblem(
        operator=_operator(grid, orientations, transfer),
        displacement_history=history,
        state_indices=tuple(range(1, len(history))),
        scored_states=set(scored),
        material_factory=_material_factory(
            pixels=pixels, orientations=orientations, library=library, threads=threads
        ),
        time_increments=increments,
        debug=debug,
    )


def _apply_corrections(
    history: np.ndarray, corrections: list[np.ndarray], damping: float
) -> np.ndarray:
    result = np.asarray(history, dtype=np.float64).copy()
    if len(corrections) != result.shape[0] - 1:
        raise ValueError("one correction is required per replayed increment")
    for index, correction in enumerate(corrections, start=1):
        value = np.asarray(correction, dtype=np.float64)
        if value.shape != result.shape[1:]:
            raise ValueError("correction shape does not match the displacement history")
        result[index] += damping * value
    return result


def _project_once(
    history: np.ndarray,
    orientations: np.ndarray,
    transfer: Any,
    increments: np.ndarray,
    scored: tuple[int, ...],
    library: str,
    threads: int,
    damping: float,
) -> tuple[np.ndarray, dict[str, float]]:
    problem = _problem(
        history,
        orientations,
        transfer,
        increments,
        scored,
        library,
        threads,
        debug=True,
    )
    evaluation = problem.evaluate(_theta_from_preset())
    corrections = [
        state.pseudo_displacement
        for state in evaluation.states
    ]
    if any(correction is None for correction in corrections):
        raise RuntimeError("debug replay did not archive projection corrections")
    projected = _apply_corrections(
        history,
        [
            np.asarray(correction, dtype=np.float64)
            for correction in corrections
            if correction is not None
        ],
        damping,
    )
    return projected, {
        "reference_residual_rms": float(evaluation.residual_rms),
        "reference_pseudo_displacement_rms": float(
            np.sqrt(np.mean([state.pseudo_displacement_norm**2 for state in evaluation.states]))
        ),
        "seconds": float(evaluation.timing.total_seconds),
    }


def _variant(
    name: str,
    observed: np.ndarray,
    passes: int,
    damping: float,
    orientations: np.ndarray,
    transfer: Any,
    increments: np.ndarray,
    scored: tuple[int, ...],
    library: str,
    threads: int,
    truth_history: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    history = observed.copy()
    diagnostics: list[dict[str, float]] = []
    for _ in range(passes):
        history, diagnostic = _project_once(
            history,
            orientations,
            transfer,
            increments,
            scored,
            library,
            threads,
            damping,
        )
        diagnostics.append(diagnostic)
    diagnostics.append(
        {
            "history_error_rms": float(np.sqrt(np.mean((history - truth_history) ** 2))),
            "observed_distance_rms": float(np.sqrt(np.mean((history - observed) ** 2))),
        }
    )
    return history, diagnostics


def _rank(
    history: np.ndarray,
    orientations: np.ndarray,
    transfer: Any,
    increments: np.ndarray,
    scored: tuple[int, ...],
    library: str,
    threads: int,
    femu_by_id: dict[str, float],
    observed_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], float]:
    problem = _problem(
        history,
        orientations,
        transfer,
        increments,
        scored,
        library,
        threads,
        debug=False,
    )
    candidates: list[dict[str, Any]] = []
    started = time.perf_counter()
    for identifier, theta, _ in _population():
        evaluation = problem.evaluate(theta)
        candidates.append(
            {
                "id": identifier,
                "status": "complete",
                "theta": theta.as_runtime_overrides(),
                "regm_rms_mm": float(evaluation.residual_rms),
                "regm_seconds": float(evaluation.timing.total_seconds),
                "femu_rms_mm": femu_by_id[identifier],
                "femu_seconds": float(observed_by_id[identifier]["femu_seconds"]),
            }
        )
    return {
        "statistics": _statistics(candidates),
        "candidates": candidates,
        "seconds": float(time.perf_counter() - started),
    }, float(problem.evaluate(_theta_from_preset()).residual_rms)


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [row["name"] for row in rows]
    spearman = [row["statistics"].get("spearman", np.nan) for row in rows]
    truth = [row["truth_regm_rms"] for row in rows]
    history_error = [row["diagnostics"][-1]["history_error_rms"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].bar(labels, spearman, color="tab:blue")
    axes[0].axhline(0.8, color="tab:red", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Spearman(REGM, observed FEMU)")
    axes[1].semilogy(labels, truth, marker="o", color="tab:orange")
    axes[1].set_ylabel("truth REGM RMS (mm)")
    axes[2].semilogy(labels, history_error, marker="o", color="tab:green")
    axes[2].set_ylabel("projected history error RMS (mm)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=0.25)
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"srix_regm_mechanical_projection.{suffix}", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--library", default="build/mfront/src/libBehaviour.so")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_report = json.loads((SOURCE / "report.json").read_text())
    target_report = json.loads(TARGET.read_text())
    fields = np.load(SOURCE / "fields.npz", mmap_mode="r")
    truth_history = np.asarray(fields["displacement_history"], dtype=np.float64)
    orientations = np.asarray(fields["orientations_deg"], dtype=np.float64)
    increments = np.asarray(source_report["time_increments"], dtype=np.float64)
    scored = tuple(int(value) for value in source_report["states_scored"])
    transfer = _WrapFreeTransfer(DICSpectralTransfer.from_sinusoidal_csv(TRANSFER))
    observed = np.asarray([transfer.apply(state) for state in truth_history])
    target_by_id = {item["id"]: item for item in target_report["candidates"]}
    femu_by_id = {
        identifier: float(item["levels"]["T1_transfer"]["femu_rms_mm"])
        for identifier, item in target_by_id.items()
    }
    variants = (("observed", 0, 0.0), ("proj025x1", 1, 0.25), ("proj050x1", 1, 0.50),
                ("proj100x1", 1, 1.0), ("proj050x2", 2, 0.50), ("proj100x2", 2, 1.0))
    rows: list[dict[str, Any]] = []
    for name, passes, damping in variants:
        if passes == 0:
            history, diagnostics = observed, []
        else:
            history, diagnostics = _variant(
                name,
                observed,
                passes,
                damping,
                orientations,
                transfer,
                increments,
                scored,
                args.library,
                args.threads,
                truth_history,
            )
        ranking, truth_regm_rms = _rank(
            history,
            orientations,
            transfer,
            increments,
            scored,
            args.library,
            args.threads,
            femu_by_id,
            target_by_id,
        )
        if passes == 0:
            diagnostics = [{
                "history_error_rms": float(np.sqrt(np.mean((history - truth_history) ** 2))),
                "observed_distance_rms": 0.0,
            }]
        rows.append({
            "name": name,
            "passes": passes,
            "damping": damping,
            "truth_regm_rms": truth_regm_rms,
            "diagnostics": diagnostics,
            **ranking,
        })
        print(
            f"{name}: spearman={rows[-1]['statistics'].get('spearman', float('nan')):.3f} "
            f"truth={truth_regm_rms:.3e}",
            flush=True,
        )
    report = {
        "schema_version": 1,
        "method": "twin-only damped mechanical projection before SRIX replay",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "source_twin": str(SOURCE.relative_to(ROOT)),
        "target_observed_femu": str(TARGET.relative_to(ROOT)),
        "transfer_csv": str(TRANSFER.relative_to(ROOT)),
        "projection_reference": "SRIX preset; one causal replay per projection pass",
        "rows": rows,
        "claims": {"new_mechanics_launched": False, "p43_authorized": False},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _plot(rows, output)


if __name__ == "__main__":
    main()
