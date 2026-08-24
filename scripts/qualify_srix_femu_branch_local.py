#!/usr/bin/env python3
"""Diagnose the local SRIX continuation branch near load fraction 0.237."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_two_state import (
    TwoStateIncrementFields,
    solve_two_state_dirichlet_plane_stress,
)
from fem_inhouse.spectral2d.step_control import LoadPathStep
from scripts.qualify_srix_femu_common_path_gate import _common_path
from scripts.qualify_srix_femu_direct_sensitivity import ROOT, _oracle_config
from scripts.qualify_srix_regm_twin import (
    PIXEL_SIZE_MM,
    _material_factory,
    _orientation_map,
    _theta_from_preset,
)

SOURCE = ROOT / "validation/reference_data/srix_femu_common_path_gate_v9/common_path.npz"
DEFAULT_OUTPUT = ROOT / "validation/reference_data/srix_femu_branch_local_v1"
F0 = 0.234375
F2 = 0.23828125
ALPHAS = (0.25, 0.4, 0.5, 0.6, 0.75)


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _local_path(base_path: list[LoadPathStep], alpha: float, pixels: int) -> list[LoadPathStep]:
    values = [
        step.end_fraction
        for step in base_path
        if step.end_fraction <= F0 + 1.0e-14 or step.end_fraction >= F2 - 1.0e-14
    ]
    values.extend((F0, F0 + alpha * (F2 - F0), F2))
    return _common_path(sorted(set(values)), pixels=pixels)


def _run(
    *,
    path: list[LoadPathStep],
    pixels: int,
    library: str,
    threads: int,
    callback: Callable[[LoadPathStep, np.ndarray], np.ndarray] | None = None,
) -> tuple[list[TwoStateIncrementFields], dict[str, Any]]:
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    material = _material_factory(
        pixels=pixels,
        orientations=_orientation_map(pixels),
        library=library,
        threads=threads,
    )(_theta_from_preset().as_runtime_overrides())
    boundary_history = np.stack(
        [np.zeros_like(path[0].boundary), *[step.boundary for step in path]]
    )
    fields: list[TwoStateIncrementFields] = []

    def observe(value: TwoStateIncrementFields) -> None:
        fields.append(value)

    result = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=material,
        boundary_displacement_history=boundary_history,
        config=_oracle_config(),
        load_path_override=path,
        initial_displacement=np.zeros_like(path[0].boundary),
        initial_guess_callback=callback,
        increment_observer=observe,
    )
    attempts = [
        {
            "accepted": attempt.accepted,
            "start_fraction": attempt.load_fraction_start,
            "end_fraction": attempt.load_fraction_end,
            "failure_reason": attempt.failure_reason,
            "newton_iterations": attempt.newton_iterations,
            "krylov_outer_callbacks": attempt.krylov_outer_callbacks,
            "minimum_line_search_factor": attempt.minimum_line_search_factor,
            "material_evaluations": attempt.material_evaluations,
            "material_seconds": attempt.material_seconds,
            "material_condensation_seconds": attempt.material_condensation_seconds,
        }
        for attempt in result.diagnostics.load_step_attempts
    ]
    return fields, {"attempts": attempts, "accepted_increments": len(fields)}


def _endpoint(fields: list[TwoStateIncrementFields], fraction: float) -> TwoStateIncrementFields:
    return min(fields, key=lambda field: abs(field.end_fraction - fraction))


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-30))


def _endpoint_spread(endpoints: list[dict[str, Any]]) -> dict[str, float]:
    """Range of endpoint discrepancies across converged local partitions."""
    keys = (
        "displacement_relative_to_coarse",
        "stress_relative_to_coarse",
        "plastic_strain_relative_to_coarse",
    )
    return {
        key: float(max(item[key] for item in endpoints) - min(item[key] for item in endpoints))
        for key in keys
    }


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
    base_path = _common_path(np.load(SOURCE)["end_fractions"].tolist(), pixels=args.pixels)
    all_runs: list[dict[str, Any]] = []

    coarse_fields, coarse_diag = _run(
        path=base_path,
        pixels=args.pixels,
        library=args.library,
        threads=args.threads,
    )
    coarse_endpoint = _endpoint(coarse_fields, F2)
    f0_endpoint = _endpoint(coarse_fields, F0)

    for alpha in ALPHAS:
        path = _local_path(base_path, alpha, args.pixels)
        try:
            fields, diagnostics = _run(
                path=path,
                pixels=args.pixels,
                library=args.library,
                threads=args.threads,
            )
            endpoint = _endpoint(fields, F2)
            all_runs.append(
                {
                    "kind": "partition",
                    "alpha": alpha,
                    "steps": len(path),
                    "status": "converged",
                    "endpoint": {
                        "fraction": endpoint.end_fraction,
                        "displacement_relative_to_coarse": _relative(
                            endpoint.displacement, coarse_endpoint.displacement
                        ),
                        "stress_relative_to_coarse": _relative(
                            endpoint.stress_in_plane_mpa, coarse_endpoint.stress_in_plane_mpa
                        ),
                        "plastic_strain_relative_to_coarse": _relative(
                            endpoint.plastic_strain_tensor, coarse_endpoint.plastic_strain_tensor
                        ),
                    },
                    "diagnostics": diagnostics,
                }
            )
        except RuntimeError as error:
            all_runs.append(
                {
                    "kind": "partition",
                    "alpha": alpha,
                    "steps": len(path),
                    "status": "failed",
                    "error": str(error),
                }
            )

    midpoint_path = _local_path(base_path, 0.5, args.pixels)
    midpoint = F0 + 0.5 * (F2 - F0)

    def extrapolated(path_item: LoadPathStep, current: np.ndarray) -> np.ndarray:
        if abs(path_item.start_fraction - midpoint) < 1.0e-14:
            return current + (F2 - midpoint) / (midpoint - F0) * (
                current - f0_endpoint.displacement
            )
        return current

    def coarse_guess(path_item: LoadPathStep, current: np.ndarray) -> np.ndarray:
        if abs(path_item.start_fraction - midpoint) < 1.0e-14:
            return coarse_endpoint.displacement
        return current

    for name, callback in (("extrapolated", extrapolated), ("coarse_endpoint_guess", coarse_guess)):
        try:
            fields, diagnostics = _run(
                path=midpoint_path,
                pixels=args.pixels,
                library=args.library,
                threads=args.threads,
                callback=callback,
            )
            endpoint = _endpoint(fields, F2)
            all_runs.append(
                {
                    "kind": "predictor",
                    "name": name,
                    "status": "converged",
                    "endpoint": {
                        "displacement_relative_to_coarse": _relative(
                            endpoint.displacement, coarse_endpoint.displacement
                        ),
                        "stress_relative_to_coarse": _relative(
                            endpoint.stress_in_plane_mpa, coarse_endpoint.stress_in_plane_mpa
                        ),
                        "plastic_strain_relative_to_coarse": _relative(
                            endpoint.plastic_strain_tensor, coarse_endpoint.plastic_strain_tensor
                        ),
                    },
                    "diagnostics": diagnostics,
                }
            )
        except RuntimeError as error:
            all_runs.append(
                {"kind": "predictor", "name": name, "status": "failed", "error": str(error)}
            )

    converged_partitions = [
        run["endpoint"] for run in all_runs
        if run["kind"] == "partition" and run["status"] == "converged"
    ]
    endpoint_spread = _endpoint_spread(converged_partitions) if converged_partitions else {}

    report = {
        "schema_version": 1,
        "method": "local SRIX branch continuation diagnostic",
        "git_sha": run_sha,
        "dirty": run_dirty,
        "machine": platform.node(),
        "parent_interval": [F0, F2],
        "alphas": list(ALPHAS),
        "coarse": {
            "steps": len(base_path),
            "diagnostics": coarse_diag,
            "endpoint_fraction": coarse_endpoint.end_fraction,
        },
        "runs": all_runs,
        "endpoint_spread_across_converged_partitions": endpoint_spread,
        "claims": {
            "branch_diagnostic_complete": True,
            "numerical_continuation_issue_demonstrated": False,
            "constitutive_branch_ambiguity_demonstrated": False,
            "path_convergence_authorized": False,
            "identification_authorized": False,
            "p43_authorized": False,
        },
        "limitations": [
            "The current TwoStateIncrementFields observer exposes stress, strain and plastic "
            "strain tensor, but not raw SRIX g/p/a arrays.",
            "The predictor comparison tests global initial guesses; it does not copy "
            "constitutive state from the coarse run.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if converged_partitions:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for key, label in (
            ("displacement_relative_to_coarse", "displacement"),
            ("stress_relative_to_coarse", "stress"),
            ("plastic_strain_relative_to_coarse", "plastic strain"),
        ):
            ax.plot(
                [run["alpha"] for run in all_runs if run["kind"] == "partition"],
                [run["endpoint"][key] for run in all_runs if run["kind"] == "partition"],
                marker="o",
                label=label,
            )
        ax.set_xlabel("local midpoint fraction alpha")
        ax.set_ylabel("relative difference at f = 0.23828125 vs coarse path")
        ax.set_title("Local SRIX continuation: endpoint sensitivity to partition")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "endpoint_partition_sensitivity.png", dpi=160)
        plt.close(fig)
    print(json.dumps({"claims": report["claims"], "runs": all_runs}, sort_keys=True))


if __name__ == "__main__":
    main()
