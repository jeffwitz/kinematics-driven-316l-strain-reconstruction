#!/usr/bin/env python3
"""Locate where the 57/114-step SRIX histories first diverge."""

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

from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_two_state import (
    TwoStateIncrementFields,
    solve_two_state_dirichlet_plane_stress,
)
from fem_inhouse.spectral2d.step_control import LoadPathStep
from scripts.qualify_srix_femu_common_path_gate import _common_path
from scripts.qualify_srix_femu_direct_sensitivity import ROOT, _oracle_config
from scripts.qualify_srix_femu_path_convergence import _refine_path
from scripts.qualify_srix_regm_twin import (
    PIXEL_SIZE_MM,
    _material_factory,
    _orientation_map,
    _theta_from_preset,
)

SOURCE = ROOT / "validation/reference_data/srix_femu_common_path_gate_v9/common_path.npz"
DEFAULT_OUTPUT = ROOT / "validation/reference_data/srix_femu_branch_causal_v1"
PREFIXES = (8, 16, 24, 32, 40, 48, 57)


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _prefix_refined_path(base: list[LoadPathStep], count: int, pixels: int) -> list[LoadPathStep]:
    if count < 0 or count > len(base):
        raise ValueError("prefix refinement count outside base path")
    refined: list[LoadPathStep] = []
    previous = np.zeros_like(base[0].boundary)
    for index, step in enumerate(base, start=1):
        pieces = 2 if index <= count else 1
        for part in range(1, pieces + 1):
            left = (part - 1) / pieces
            right = part / pieces
            refined.append(
                LoadPathStep(
                    index=len(refined) + 1,
                    start_fraction=(
                        step.start_fraction + left * (step.end_fraction - step.start_fraction)
                    ),
                    end_fraction=(
                        step.start_fraction + right * (step.end_fraction - step.start_fraction)
                    ),
                    boundary=previous + right * (step.boundary - previous),
                    time_increment=step.time_increment / pieces,
                )
            )
        previous = np.asarray(step.boundary, dtype=np.float64)
    return _common_path([step.end_fraction for step in refined], pixels=pixels)


def _run(
    *, path: list[LoadPathStep], pixels: int, library: str, threads: int
) -> tuple[list[TwoStateIncrementFields], dict[str, Any], str | None]:
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    material = _material_factory(
        pixels=pixels,
        orientations=_orientation_map(pixels),
        library=library,
        threads=threads,
    )(_theta_from_preset().as_runtime_overrides())
    history = np.stack([np.zeros_like(path[0].boundary), *[step.boundary for step in path]])
    fields: list[TwoStateIncrementFields] = []

    try:
        result = solve_two_state_dirichlet_plane_stress(
            grid=grid,
            material=material,
            boundary_displacement_history=history,
            config=_oracle_config(),
            load_path_override=path,
            initial_displacement=None,
            increment_observer=fields.append,
        )
        error = None
        attempts = result.diagnostics.load_step_attempts
    except RuntimeError as exc:
        error = str(exc)
        attempts = (
            material.last_diagnostics.load_step_attempts
            if hasattr(material, "last_diagnostics")
            else ()
        )
    diagnostics = {
        "accepted_increments": len(fields),
        "attempts": [
            {
                "start_fraction": item.load_fraction_start,
                "end_fraction": item.load_fraction_end,
                "accepted": item.accepted,
                "failure_reason": item.failure_reason,
                "newton_iterations": item.newton_iterations,
                "krylov_outer_callbacks": item.krylov_outer_callbacks,
                "minimum_line_search_factor": item.minimum_line_search_factor,
                "material_evaluations": item.material_evaluations,
                "material_seconds": item.material_seconds,
                "material_condensation_seconds": item.material_condensation_seconds,
            }
            for item in attempts
        ],
    }
    return fields, diagnostics, error


def _field_at(
    fields: list[TwoStateIncrementFields], fraction: float
) -> TwoStateIncrementFields | None:
    for field in fields:
        if abs(field.end_fraction - fraction) <= 1.0e-14:
            return field
    return None


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-30))


def _max_difference(left: np.ndarray, right: np.ndarray) -> tuple[float, tuple[int, ...]]:
    delta = np.abs(left - right)
    index = tuple(int(value) for value in np.unravel_index(np.argmax(delta), delta.shape))
    return float(delta[index]), index


def _state_comparison(
    coarse: TwoStateIncrementFields, refined: TwoStateIncrementFields
) -> dict[str, Any]:
    result: dict[str, Any] = {"fraction": coarse.end_fraction}
    for name, left, right in (
        ("displacement", coarse.displacement, refined.displacement),
        ("stress", coarse.stress_in_plane_mpa, refined.stress_in_plane_mpa),
        ("plastic_strain", coarse.plastic_strain_tensor, refined.plastic_strain_tensor),
        ("elastic_strain", coarse.elastic_strain_tensor, refined.elastic_strain_tensor),
    ):
        if left is not None and right is not None:
            result[f"{name}_relative_l2"] = _relative(right, left)
            maximum, index = _max_difference(right, left)
            result[f"{name}_max_abs"] = maximum
            result[f"{name}_max_index"] = index
    for name in ("plastic_slip", "equivalent_plastic_slip", "back_strain"):
        left = coarse.observables.get(name)
        right = refined.observables.get(name)
        if left is None or right is None:
            continue
        result[f"{name}_relative_l2"] = _relative(right, left)
        maximum, index = _max_difference(right, left)
        result[f"{name}_max_abs"] = maximum
        result[f"{name}_max_index"] = index
        if name in ("plastic_slip", "equivalent_plastic_slip"):
            left_active = np.abs(left) > 1.0e-12
            right_active = np.abs(right) > 1.0e-12
            result[f"{name}_activation_switches"] = int(
                np.count_nonzero(left_active != right_active)
            )
    return result


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
    base = _common_path(np.load(SOURCE)["end_fractions"].tolist(), pixels=args.pixels)
    base_fields, base_diag, base_error = _run(
        path=base, pixels=args.pixels, library=args.library, threads=args.threads
    )
    refined_path = _refine_path(base)
    refined_fields, refined_diag, refined_error = _run(
        path=refined_path, pixels=args.pixels, library=args.library, threads=args.threads
    )
    common = []
    common_base_fields: list[TwoStateIncrementFields] = []
    common_refined_fields: list[TwoStateIncrementFields] = []
    for field in base_fields:
        counterpart = _field_at(refined_fields, field.end_fraction)
        if counterpart is not None:
            common.append(_state_comparison(field, counterpart))
            common_base_fields.append(field)
            common_refined_fields.append(counterpart)
    prefixes: list[dict[str, Any]] = []
    for count in PREFIXES:
        path = _prefix_refined_path(base, count, args.pixels)
        fields, diagnostics, error = _run(
            path=path, pixels=args.pixels, library=args.library, threads=args.threads
        )
        prefixes.append(
            {
                "refined_base_intervals": count,
                "path_steps": len(path),
                "accepted_increments": len(fields),
                "status": "converged" if error is None else "failed",
                "error": error,
                "last_accepted_fraction": fields[-1].end_fraction if fields else None,
                "diagnostics": diagnostics,
            }
        )
    report = {
        "schema_version": 1,
        "method": "causal 57/114 SRIX history divergence diagnostic",
        "git_sha": run_sha,
        "dirty": run_dirty,
        "machine": platform.node(),
        "base_steps": len(base),
        "refined_steps": len(refined_path),
        "base_error": base_error,
        "refined_error": refined_error,
        "base_diagnostics": base_diag,
        "refined_diagnostics": refined_diag,
        "common_endpoint_comparisons": common,
        "prefix_refinement": prefixes,
        "available_observables": sorted(base_fields[-1].observables) if base_fields else [],
        "claims": {
            "branch_diagnostic_complete": True,
            "accumulated_incremental_drift": False,
            "active_set_transition": False,
            "active_set_transition_candidate": bool(
                any(item.get("plastic_slip_activation_switches", 0) for item in common)
            ),
            "global_solver_basin_or_conditioning": False,
            "unresolved": True,
            "identification_authorized": False,
            "p43_authorized": False,
        },
        "limitations": [
            "The 114 path is compared only through its accepted prefix before the Newton failure.",
            "The current observer does not expose per-Newton local Cbb condition "
            "numbers or raw MGIS substep traces.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if common_base_fields:
        arrays: dict[str, np.ndarray] = {
            "fractions": np.asarray([item.end_fraction for item in common_base_fields]),
            "base_displacement": np.stack([item.displacement for item in common_base_fields]),
            "refined_displacement": np.stack(
                [item.displacement for item in common_refined_fields]
            ),
            "base_stress": np.stack(
                [item.stress_in_plane_mpa for item in common_base_fields]
            ),
            "refined_stress": np.stack(
                [item.stress_in_plane_mpa for item in common_refined_fields]
            ),
        }
        for name in ("plastic_strain_tensor", "elastic_strain_tensor"):
            base_values = [getattr(item, name) for item in common_base_fields]
            refined_values = [getattr(item, name) for item in common_refined_fields]
            if all(value is not None for value in (*base_values, *refined_values)):
                arrays[f"base_{name}"] = np.stack(base_values)
                arrays[f"refined_{name}"] = np.stack(refined_values)
        for name in sorted(set().union(*(item.observables for item in common_base_fields))):
            base_values = [item.observables[name] for item in common_base_fields]
            refined_values = [item.observables[name] for item in common_refined_fields]
            arrays[f"base_observable_{name}"] = np.stack(base_values)
            arrays[f"refined_observable_{name}"] = np.stack(refined_values)
        np.savez_compressed(output / "common_endpoint_states.npz", **arrays)
    if common:
        fractions = [item["fraction"] for item in common]
        slip = [item.get("plastic_slip_relative_l2", np.nan) for item in common]
        stress = [item.get("stress_relative_l2", np.nan) for item in common]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.semilogy(fractions, stress, label="stress")
        if np.isfinite(slip).any():
            ax.semilogy(fractions, slip, label="plastic slip g")
        ax.set_xlabel("common coarse endpoint fraction")
        ax.set_ylabel("relative difference: 114 vs 57")
        ax.set_title("Causal divergence before the 114-step failure")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "causal_divergence.png", dpi=160)
        plt.close(fig)
    print(json.dumps({"claims": report["claims"], "prefix_refinement": prefixes}, sort_keys=True))


if __name__ == "__main__":
    main()
