#!/usr/bin/env python3
"""Qualify direct FEMU sensitivities on a synchronized adaptive path."""

from __future__ import annotations

import argparse
import json
import platform
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from fem_inhouse.identification.dic_whitening import DICSpectralTransfer
from fem_inhouse.identification.srix_equilibrium_gap import SrixTheta4
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_two_state import TwoStateIncrementFields
from fem_inhouse.spectral2d.step_control import LoadPathStep
from scripts.qualify_srix_femu_direct_sensitivity import (
    FD_STEP,
    ROOT,
    _direct_jacobian,
    _reference_trajectory,
)
from scripts.qualify_srix_femu_fixed_path_gate import (
    _comparison,
    _fixed_path_fd,
    _fixed_path_trajectory,
)
from scripts.qualify_srix_regm_transfer_noise import _WrapFreeTransfer
from scripts.qualify_srix_regm_twin import (
    PIXEL_SIZE_MM,
    _boundary_history,
    _orientation_map,
    _theta_from_preset,
)

SOURCE = ROOT / "validation/reference_data/srix_regm_information_geometry_v1"
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
DEFAULT_OUTPUT = ROOT / "validation/reference_data/srix_femu_common_path_gate_v1"


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _variants(theta: SrixTheta4) -> list[tuple[str, SrixTheta4]]:
    eta = theta.log_coordinates()
    result = [("base", theta)]
    names = (
        "tau0_plus",
        "tau0_minus",
        "R_plus",
        "R_minus",
        "Q_plus",
        "Q_minus",
        "b_plus",
        "b_minus",
    )
    for index in range(4):
        for sign, name in ((1.0, names[2 * index]), (-1.0, names[2 * index + 1])):
            perturbed = eta.copy()
            perturbed[index] += sign * FD_STEP
            result.append((name, SrixTheta4.from_log_coordinates(perturbed)))
    return result


def _common_path(
    fractions: list[float],
    *,
    pixels: int,
) -> list[LoadPathStep]:
    values = np.asarray(sorted({0.0, 1.0, *fractions}), dtype=np.float64)
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    history = _boundary_history(grid)
    anchors = np.linspace(0.0, 1.0, history.shape[0])
    flat = history.reshape(history.shape[0], -1)
    boundaries = np.column_stack(
        [np.interp(values, anchors, flat[:, column]) for column in range(flat.shape[1])]
    ).reshape(len(values), *grid.node_shape, 2)
    return [
        LoadPathStep(
            index=index,
            start_fraction=float(values[index - 1]),
            end_fraction=float(values[index]),
            boundary=boundaries[index].copy(),
            time_increment=float(values[index] - values[index - 1]),
        )
        for index in range(1, len(values))
    ]


def _failure_increment(error: BaseException) -> int | None:
    match = re.search(r"increment (\d+)", str(error))
    return None if match is None else int(match.group(1))


def _adaptive_with_timeout(
    *,
    theta: SrixTheta4,
    pixels: int,
    library: str,
    threads: int,
    timeout_seconds: float,
) -> tuple[list[TwoStateIncrementFields], dict[str, Any], float]:
    if timeout_seconds <= 0.0:
        return _reference_trajectory(
            pixels=pixels, library=library, threads=threads, theta=theta
        )

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"adaptive trajectory exceeded {timeout_seconds:g} seconds")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return _reference_trajectory(
            pixels=pixels, library=library, threads=threads, theta=theta
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _synchronise(
    *,
    adaptive: dict[str, list[TwoStateIncrementFields]],
    variants: list[tuple[str, SrixTheta4]],
    pixels: int,
    library: str,
    threads: int,
    max_bisections: int,
) -> tuple[list[LoadPathStep], list[TwoStateIncrementFields], dict[str, Any]]:
    fractions = [
        fraction
        for fields in adaptive.values()
        for fraction in [0.0, *[f.end_fraction for f in fields]]
    ]
    path = _common_path(fractions, pixels=pixels)
    history: list[dict[str, Any]] = []
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    for iteration in range(max_bisections + 1):
        base_fields: list[TwoStateIncrementFields] | None = None
        failure: dict[str, Any] | None = None
        for name, theta in variants:
            try:
                fields = _fixed_path_trajectory(
                    theta=theta,
                    path=path,
                    initial_displacement=(
                        np.zeros((*grid.node_shape, 2))
                        if base_fields is None
                        else np.asarray(base_fields[0].displacement)
                    ),
                    pixels=pixels,
                    library=library,
                    threads=threads,
                )
                if name == "base":
                    base_fields = fields
            except RuntimeError as error:
                failure = {
                    "iteration": iteration,
                    "direction": name,
                    "failure": str(error),
                    "failed_increment": _failure_increment(error),
                    "path_steps": len(path),
                }
                break
        if failure is None:
            assert base_fields is not None
            return path, base_fields, {"status": "converged", "bisections": history}
        failed = failure["failed_increment"]
        if failed is None or not 1 <= failed <= len(path):
            return path, [], {"status": "blocked", "bisections": [*history, failure]}
        left = path[failed - 1].start_fraction
        right = path[failed - 1].end_fraction
        midpoint = 0.5 * (left + right)
        if midpoint <= left or midpoint >= right:
            return path, [], {"status": "blocked", "bisections": [*history, failure]}
        history.append(
            {
                **failure,
                "start_fraction": left,
                "end_fraction": right,
                "inserted_fraction": midpoint,
            }
        )
        ends = [step.end_fraction for step in path]
        path = _common_path(
            [*ends[: failed - 1], midpoint, *ends[failed - 1:]], pixels=pixels
        )
    return path, [], {"status": "blocked", "bisections": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--library", default="build/mfront/src/libBehaviour.so")
    parser.add_argument("--max-bisections", type=int, default=24)
    parser.add_argument(
        "--adaptive-timeout",
        type=float,
        default=600.0,
        help="maximum seconds per adaptive trajectory; zero disables the limit",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    theta = _theta_from_preset()
    adaptive: dict[str, list[TwoStateIncrementFields]] = {}
    adaptive_diagnostics: dict[str, Any] = {}
    for name, direction in _variants(theta):
        print(f"adaptive path: {name}", flush=True)
        try:
            fields, diagnostics, _ = _adaptive_with_timeout(
                theta=direction,
                pixels=args.pixels,
                library=args.library,
                threads=args.threads,
                timeout_seconds=args.adaptive_timeout,
            )
        except TimeoutError as error:
            report = {
                "schema_version": 1,
                "method": "direct FEMU sensitivity versus synchronized common-path FD",
                "git_sha": _git("rev-parse HEAD"),
                "dirty": bool(_git("status --porcelain")),
                "machine": platform.node(),
                "fd_step_log": FD_STEP,
                "status": "blocked_adaptive_trajectory_timeout",
                "completed_directions": sorted(adaptive),
                "timed_out_direction": name,
                "timeout_seconds": args.adaptive_timeout,
                "error": str(error),
                "claims": {
                    "common_path_fd_available": False,
                    "direct_femu_qualified": False,
                    "p43_authorized": False,
                },
            }
            (output / "report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps(report, sort_keys=True), flush=True)
            return
        adaptive[name] = fields
        adaptive_diagnostics[name] = {
            "accepted_increments": len(fields),
            "end_fractions": [field.end_fraction for field in fields],
            "solver": diagnostics["solver"],
        }
    common, base_fields, sync = _synchronise(
        adaptive=adaptive,
        variants=_variants(theta),
        pixels=args.pixels,
        library=args.library,
        threads=args.threads,
        max_bisections=args.max_bisections,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "direct FEMU sensitivity versus synchronized common-path FD",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "fd_step_log": FD_STEP,
        "adaptive_paths": adaptive_diagnostics,
        "common_path": {
            "status": sync["status"],
            "steps": len(common),
            "end_fractions": [step.end_fraction for step in common],
            "bisections": sync["bisections"],
        },
        "claims": {
            "common_path_fd_available": False,
            "direct_femu_qualified": False,
            "p43_authorized": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    np.savez_compressed(
        output / "common_path.npz",
        end_fractions=np.asarray([step.end_fraction for step in common]),
        boundaries=np.asarray([step.boundary for step in common]),
    )
    if sync["status"] != "converged":
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    scored_source = json.loads((SOURCE / "report.json").read_text())["states_scored"]
    targets = [adaptive["base"][int(index) - 1].end_fraction for index in scored_source]
    scored = tuple(
        int(np.argmin([abs(field.end_fraction - target) for field in base_fields])) + 1
        for target in targets
    )
    transfer = _WrapFreeTransfer(DICSpectralTransfer.from_sinusoidal_csv(TRANSFER))
    direct, direct_timing = _direct_jacobian(
        fields=base_fields,
        scored=scored,
        orientations=_orientation_map(args.pixels),
        theta=theta,
        library=args.library,
        threads=args.threads,
        transfer=transfer,
        h=FD_STEP,
    )
    fixed_fd = _fixed_path_fd(
        base_fields=base_fields,
        scored=scored,
        pixels=args.pixels,
        library=args.library,
        threads=args.threads,
        transfer=transfer,
        h=FD_STEP,
        path=common,
    )
    comparison = _comparison(direct, fixed_fd)
    report.update(
        {
            "states_scored": list(scored),
            "direct_timing": direct_timing,
            "comparison_direct_vs_common_fd": comparison,
            "claims": {
                "common_path_fd_available": True,
                "direct_femu_qualified": all(
                    value < 0.02 for value in comparison["relative_column_l2_errors"]
                )
                and all(value > 0.999 for value in comparison["column_cosines"]),
                "p43_authorized": False,
            },
        }
    )
    np.savez_compressed(output / "jacobians.npz", FEMU_direct=direct, FEMU_common_path_fd=fixed_fd)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(comparison, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
