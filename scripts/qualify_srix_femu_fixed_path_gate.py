#!/usr/bin/env python3
"""Compare direct FEMU sensitivities with a fixed-accepted-path FD oracle."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import subspace_angles

from fem_inhouse.identification.dic_whitening import DICSpectralTransfer
from fem_inhouse.identification.srix_equilibrium_gap import SrixTheta4
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_two_state import (
    TwoStateIncrementFields,
    solve_two_state_dirichlet_plane_stress,
)
from fem_inhouse.spectral2d.step_control import LoadPathStep
from scripts.qualify_srix_femu_direct_sensitivity import (
    FD_STEP,
    ROOT,
    _direct_jacobian,
    _geometry,
    _reference_config,
    _reference_trajectory,
)
from scripts.qualify_srix_regm_information_geometry import _plot
from scripts.qualify_srix_regm_transfer_noise import _WrapFreeTransfer
from scripts.qualify_srix_regm_twin import (
    PIXEL_SIZE_MM,
    _material_factory,
    _orientation_map,
    _theta_from_preset,
)

SOURCE = ROOT / "validation/reference_data/srix_regm_information_geometry_v1"
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
DEFAULT_OUTPUT = ROOT / "validation/reference_data/srix_femu_fixed_path_gate_v1"


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _path_from_fields(
    base_fields: list[TwoStateIncrementFields],
    *,
    refinement: int = 1,
) -> list[LoadPathStep]:
    if refinement < 1:
        raise ValueError("path refinement must be positive")
    previous_boundary = np.zeros_like(base_fields[0].boundary)
    path: list[LoadPathStep] = []
    path_index = 0
    for field in base_fields:
        for substep in range(1, refinement + 1):
            start = (substep - 1) / refinement
            end = substep / refinement
            boundary = previous_boundary + end * (field.boundary - previous_boundary)
            path_index += 1
            path.append(
                LoadPathStep(
                    index=path_index,
                    start_fraction=field.start_fraction + start * (
                        field.end_fraction - field.start_fraction
                    ),
                    end_fraction=field.start_fraction + end * (
                        field.end_fraction - field.start_fraction
                    ),
                    boundary=boundary,
                    time_increment=field.time_increment / refinement,
                )
            )
        previous_boundary = np.asarray(field.boundary)
    return path


def _fixed_path_trajectory(
    *,
    theta: SrixTheta4,
    path: list[LoadPathStep],
    initial_displacement: np.ndarray,
    pixels: int,
    library: str,
    threads: int,
    config: Any | None = None,
) -> list[TwoStateIncrementFields]:
    orientations = _orientation_map(pixels)
    material = _material_factory(
        pixels=pixels, orientations=orientations, library=library, threads=threads
    )(theta.as_runtime_overrides())
    zero = np.zeros_like(path[0].boundary)
    boundary_history = np.stack([zero, *[field.boundary for field in path]])
    collected: list[TwoStateIncrementFields] = []

    def observe(value: TwoStateIncrementFields) -> None:
        collected.append(value)

    # Keep the base accepted path fixed, but allow the perturbed constitutive
    # solve enough local/global Newton iterations to converge on that same
    # interval.  The path is the object being frozen; the iteration cap is not
    # part of the physical discrete load path and must not turn the FD oracle
    # into a failure oracle.
    if config is None:
        config = replace(
            _reference_config(),
            adaptive_stepping_enabled=False,
            maximum_newton_iterations=80,
            maximum_line_search_reductions=20,
        )
    solve_two_state_dirichlet_plane_stress(
        grid=StructuredGrid2D(
            pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels
        ),
        material=material,
        boundary_displacement_history=boundary_history,
        config=config,
        load_path_override=path,
        initial_displacement=np.asarray(initial_displacement),
        increment_observer=observe,
    )
    if len(collected) != len(path):
        raise RuntimeError("fixed-path forward did not preserve the base increment count")
    return collected


def _fixed_path_forward(
    *,
    theta: SrixTheta4,
    base_fields: list[TwoStateIncrementFields],
    pixels: int,
    library: str,
    threads: int,
    path: list[LoadPathStep] | None = None,
) -> list[np.ndarray]:
    selected_path = path if path is not None else _path_from_fields(base_fields)
    fields = _fixed_path_trajectory(
        theta=theta,
        path=selected_path,
        initial_displacement=np.asarray(base_fields[0].displacement),
        pixels=pixels,
        library=library,
        threads=threads,
    )
    return [np.asarray(field.displacement).copy() for field in fields]


def _fixed_path_fd(
    *,
    base_fields: list[TwoStateIncrementFields],
    scored: tuple[int, ...],
    pixels: int,
    library: str,
    threads: int,
    transfer: Any,
    h: float,
    path: list[LoadPathStep] | None = None,
) -> np.ndarray:
    eta = _theta_from_preset().log_coordinates()
    columns = []
    for index in range(4):
        plus = eta.copy()
        minus = eta.copy()
        plus[index] += h
        minus[index] -= h
        plus_history = _fixed_path_forward(
            theta=SrixTheta4.from_log_coordinates(plus),
            base_fields=base_fields,
            pixels=pixels,
            library=library,
            threads=threads,
            path=path,
        )
        minus_history = _fixed_path_forward(
            theta=SrixTheta4.from_log_coordinates(minus),
            base_fields=base_fields,
            pixels=pixels,
            library=library,
            threads=threads,
            path=path,
        )
        plus_values = np.concatenate(
            [np.asarray(transfer.apply(plus_history[state - 1])).reshape(-1) for state in scored]
        )
        minus_values = np.concatenate(
            [np.asarray(transfer.apply(minus_history[state - 1])).reshape(-1) for state in scored]
        )
        columns.append((plus_values - minus_values) / (2.0 * h))
    return np.column_stack(columns)


def _comparison(direct: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    errors = []
    cosines = []
    for index in range(4):
        left = direct[:, index]
        right = target[:, index]
        errors.append(float(np.linalg.norm(left - right) / np.linalg.norm(right)))
        cosines.append(float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))))
    geometry = _geometry(direct)
    geometry["cumulative"] = []
    target_geometry = _geometry(target)
    target_geometry["cumulative"] = []
    angles = {
        str(count): np.degrees(
            subspace_angles(
                np.asarray(geometry["right_singular_vectors"])[:, :count],
                np.asarray(target_geometry["right_singular_vectors"])[:, :count],
            )
        ).tolist()
        for count in (1, 2, 3)
    }
    return {
        "relative_column_l2_errors": errors,
        "column_cosines": cosines,
        "geometry": geometry,
        "target_geometry": target_geometry,
        "principal_angles_degrees": angles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--library", default="build/mfront/src/libBehaviour.so")
    parser.add_argument("--fd-step", type=float, default=FD_STEP)
    parser.add_argument("--path-refinement", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    adaptive_fields, forward_diagnostics, forward_seconds = _reference_trajectory(
        pixels=args.pixels,
        library=args.library,
        threads=args.threads,
        theta=_theta_from_preset(),
    )
    adaptive_scored = tuple(
        int(value)
        for value in json.loads((SOURCE / "report.json").read_text())["states_scored"]
    )
    path = _path_from_fields(adaptive_fields, refinement=args.path_refinement)
    if args.path_refinement == 1:
        base_fields = adaptive_fields
    else:
        grid = StructuredGrid2D(
            args.pixels,
            args.pixels,
            PIXEL_SIZE_MM * args.pixels,
            PIXEL_SIZE_MM * args.pixels,
        )
        base_fields = _fixed_path_trajectory(
            theta=_theta_from_preset(),
            path=path,
            initial_displacement=np.zeros((*grid.node_shape, 2)),
            pixels=args.pixels,
            library=args.library,
            threads=args.threads,
        )
    target_fractions = [adaptive_fields[index - 1].end_fraction for index in adaptive_scored]
    scored = tuple(
        int(np.argmin([abs(field.end_fraction - target) for field in base_fields])) + 1
        for target in target_fractions
    )
    transfer = _WrapFreeTransfer(DICSpectralTransfer.from_sinusoidal_csv(TRANSFER))
    orientations = _orientation_map(args.pixels)
    direct, direct_timing = _direct_jacobian(
        fields=base_fields,
        scored=scored,
        orientations=orientations,
        theta=_theta_from_preset(),
        library=args.library,
        threads=args.threads,
        transfer=transfer,
        h=args.fd_step,
    )
    fixed_fd_started = time.perf_counter()
    fixed_fd = _fixed_path_fd(
        base_fields=base_fields,
        scored=scored,
        pixels=args.pixels,
        library=args.library,
        threads=args.threads,
        transfer=transfer,
        h=args.fd_step,
    )
    fixed_fd_seconds = time.perf_counter() - fixed_fd_started
    archived = np.asarray(np.load(SOURCE / "jacobians.npz")["FEMU_observed"])
    report = {
        "schema_version": 1,
        "method": "direct FEMU matrix-free sensitivity versus fixed accepted-path FD",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "states_scored": list(scored),
        "fd_step_log": args.fd_step,
        "accepted_increments": len(base_fields),
        "path_refinement": args.path_refinement,
        "timing_seconds": {
            "base_forward": forward_seconds,
            "direct_sensitivity": direct_timing["elapsed_seconds"],
            "fixed_path_fd": fixed_fd_seconds,
            "total": time.perf_counter() - started,
        },
        "forward_diagnostics": forward_diagnostics,
        "comparison_direct_vs_fixed_path_fd": _comparison(direct, fixed_fd),
        "comparison_direct_vs_archived_adaptive_fd": _comparison(direct, archived),
        "claims": {
            "adaptive_archived_fd_is_same_path": False,
            "direct_femu_qualified": False,
            "p43_authorized": False,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output / "jacobians.npz",
        FEMU_direct=direct,
        FEMU_fixed_path_fd=fixed_fd,
        FEMU_archived_adaptive_fd=archived,
    )
    _plot(
        {
            "FEMU_direct": report["comparison_direct_vs_fixed_path_fd"]["geometry"],
            "FEMU_fixed_path_FD": report["comparison_direct_vs_fixed_path_fd"]["target_geometry"],
            "FEMU_archived_FD": report[
                "comparison_direct_vs_archived_adaptive_fd"
            ]["target_geometry"],
        },
        output,
    )
    print(json.dumps(report["comparison_direct_vs_fixed_path_fd"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
