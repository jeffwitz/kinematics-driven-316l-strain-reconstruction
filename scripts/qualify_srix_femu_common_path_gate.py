#!/usr/bin/env python3
"""Qualify direct FEMU sensitivities on a synchronized adaptive path."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
    _oracle_config,
    _path_search_config,
    _reference_trajectory,
    _seed_config,
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
CACHE_ROOT = ROOT / "validation/reference_data/srix_femu_common_path_cache"
SEED_SKIP_DEFAULT = {"b_minus"}
MAX_LOCAL_BISECTIONS = 10
MIN_COMMON_INTERVAL = 1.0 / 65536.0


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_metadata(
    *,
    pixels: int,
    library: str,
    threads: int,
    git_sha: str | None = None,
    dirty: bool | None = None,
) -> dict[str, Any]:
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    history = np.asarray(_boundary_history(grid), dtype=np.float64)
    return {
        "schema_version": 1,
        "fixed_path_initialization_contract": 2,
        "git_sha": _git("rev-parse HEAD") if git_sha is None else git_sha,
        "dirty": bool(_git("status --porcelain")) if dirty is None else dirty,
        "pixels": pixels,
        "threads": threads,
        "library": library,
        "library_sha256": _sha256_file(ROOT / library),
        "boundary_sha256": hashlib.sha256(history.tobytes()).hexdigest(),
        "boundary_shape": list(history.shape),
        "fd_step_log": FD_STEP,
        "seed_config": {
            "relative_equilibrium_tolerance": 1.0e-5,
            "maximum_newton_iterations": 12,
            "verify_final_state": False,
            "minimum_increment_fraction": 1.0 / 1024.0,
            "increment_growth_factor": 2.0,
            "maximum_cutbacks_per_step": 3,
            "line_search_difficult_threshold": 0.25,
        },
    }


def _cache_path(name: str) -> Path:
    return CACHE_ROOT / f"{name}.json"


def _load_cached_fractions(
    name: str,
    *,
    metadata: dict[str, Any],
    theta: SrixTheta4,
) -> tuple[list[float] | None, str]:
    path = _cache_path(name)
    if not path.exists():
        return None, "cache_missing"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return None, f"cache_invalid:{error}"
    expected = dict(metadata)
    expected["theta_log"] = theta.log_coordinates().tolist()
    for key, value in expected.items():
        if payload.get(key) != value:
            return None, f"cache_mismatch:{key}"
    fractions = payload.get("end_fractions")
    if not isinstance(fractions, list) or not fractions:
        return None, "cache_invalid:end_fractions"
    values = [float(value) for value in fractions]
    if any(not 0.0 < value <= 1.0 for value in values) or any(
        right <= left for left, right in itertools.pairwise(values)
    ) or values[-1] != 1.0:
        return None, "cache_invalid:fractions"
    return values, "cache_hit"


def _write_cached_fractions(
    name: str,
    fractions: list[float],
    *,
    metadata: dict[str, Any],
    theta: SrixTheta4,
    diagnostics: dict[str, Any],
) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload.update(
        {
            "name": name,
            "theta_log": theta.log_coordinates().tolist(),
            "accepted_increments": len(fractions),
            "end_fractions": fractions,
            "timing": diagnostics.get("elapsed_seconds"),
            "solver": diagnostics.get("solver", {}),
        }
    )
    _cache_path(name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
    config: Any,
) -> tuple[list[TwoStateIncrementFields], dict[str, Any], float]:
    if timeout_seconds <= 0.0:
        return _reference_trajectory(
            pixels=pixels, library=library, threads=threads, theta=theta, config=config
        )

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"adaptive trajectory exceeded {timeout_seconds:g} seconds")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return _reference_trajectory(
            pixels=pixels, library=library, threads=threads, theta=theta, config=config
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _fixed_path_with_timeout(
    *,
    theta: SrixTheta4,
    path: list[LoadPathStep],
    pixels: int,
    library: str,
    threads: int,
    config: Any,
    timeout_seconds: float,
) -> list[TwoStateIncrementFields]:
    """Run one exploratory fixed-path variant with a hard wall-clock guard."""

    if timeout_seconds <= 0.0:
        return _fixed_path_trajectory(
            theta=theta,
            path=path,
            initial_displacement=None,
            pixels=pixels,
            library=library,
            threads=threads,
            config=config,
        )

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"path-search variant exceeded {timeout_seconds:g} seconds")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return _fixed_path_trajectory(
            theta=theta,
            path=path,
            initial_displacement=None,
            pixels=pixels,
            library=library,
            threads=threads,
            config=config,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _synchronise(
    *,
    adaptive: dict[str, list[float]],
    variants: list[tuple[str, SrixTheta4]],
    pixels: int,
    library: str,
    threads: int,
    max_bisections: int,
    max_local_bisections: int = MAX_LOCAL_BISECTIONS,
    min_interval: float = MIN_COMMON_INTERVAL,
    path_search_timeout: float = 120.0,
) -> tuple[list[LoadPathStep], list[TwoStateIncrementFields], dict[str, Any]]:
    fractions = [
        fraction
        for ends in adaptive.values()
        for fraction in [0.0, *ends]
    ]
    path = _common_path(fractions, pixels=pixels)
    history: list[dict[str, Any]] = []
    search_config = _path_search_config()
    search_order = [
        name
        for name in (
            "b_minus",
            "b_plus",
            "Q_minus",
            "Q_plus",
            "R_minus",
            "R_plus",
            "tau0_minus",
            "tau0_plus",
            "base",
        )
        if name in dict(variants)
    ]
    by_name = dict(variants)
    bisections_by_interval: dict[tuple[float, float], int] = {}
    total_bisections = 0

    def bisect(failure: dict[str, Any]) -> bool:
        nonlocal path, total_bisections
        failed = failure["failed_increment"]
        if failed is None or not 1 <= failed <= len(path):
            return False
        left = path[failed - 1].start_fraction
        right = path[failed - 1].end_fraction
        width = right - left
        key = (left, right)
        count = bisections_by_interval.get(key, 0)
        if (
            count >= max_local_bisections
            or width <= min_interval
            or total_bisections >= max_bisections
        ):
            return False
        midpoint = 0.5 * (left + right)
        bisections_by_interval[key] = count + 1
        total_bisections += 1
        history.append(
            {
                **failure,
                "start_fraction": left,
                "end_fraction": right,
                "inserted_fraction": midpoint,
                "local_bisections": count + 1,
            }
        )
        ends = [step.end_fraction for step in path]
        path = _common_path(
            [*ends[: failed - 1], midpoint, *ends[failed - 1:]], pixels=pixels
        )
        return True

    # Search one difficult variant at a time.  This avoids restarting all
    # nine trajectories after every exploratory bisection.
    for name in search_order:
        theta = by_name[name]
        deadline = (
            None
            if path_search_timeout <= 0.0
            else time.perf_counter() + path_search_timeout
        )
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                return path, [], {
                    "status": "blocked_path_search_timeout",
                    "bisections": history,
                    "failed_variant": name,
                    "timeout_seconds": path_search_timeout,
                }
            try:
                fields = _fixed_path_with_timeout(
                    theta=theta,
                    path=path,
                    pixels=pixels,
                    library=library,
                    threads=threads,
                    config=search_config,
                    timeout_seconds=(
                        path_search_timeout
                        if deadline is None
                        else max(0.0, deadline - time.perf_counter())
                    ),
                )
            except (RuntimeError, TimeoutError) as error:
                failure = {
                    "phase": "path_search",
                    "direction": name,
                    "failure": str(error),
                    "failed_increment": _failure_increment(error),
                    "path_steps": len(path),
                }
                if not bisect(failure):
                    return path, [], {
                        "status": "blocked_path_search",
                        "bisections": [*history, failure],
                        "failed_variant": name,
                    }
                continue
            break

    # The exploratory path is now replayed once with the strict oracle policy.
    # A strict failure sends only that variant back through the fail-fast search.
    oracle_config = _oracle_config()
    while True:
        strict_failure: dict[str, Any] | None = None
        strict_base: list[TwoStateIncrementFields] | None = None
        for name, theta in variants:
            try:
                fields = _fixed_path_trajectory(
                    theta=theta,
                    path=path,
                    initial_displacement=None,
                    pixels=pixels,
                    library=library,
                    threads=threads,
                    config=oracle_config,
                )
                if name == "base":
                    strict_base = fields
            except RuntimeError as error:
                strict_failure = {
                    "phase": "oracle",
                    "direction": name,
                    "failure": str(error),
                    "failed_increment": _failure_increment(error),
                    "path_steps": len(path),
                }
                break
        if strict_failure is None:
            assert strict_base is not None
            return path, strict_base, {
                "status": "converged",
                "bisections": history,
                "search_order": search_order,
            }
        if not bisect(strict_failure):
            return path, [], {
                "status": "blocked_oracle",
                "bisections": [*history, strict_failure],
                "failed_variant": strict_failure["direction"],
            }
        # Qualify the new interval for the failing variant before another
        # strict replay.  This keeps the expensive oracle as a final check.
        name = str(strict_failure["direction"])
        deadline = (
            None
            if path_search_timeout <= 0.0
            else time.perf_counter() + path_search_timeout
        )
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                return path, [], {
                    "status": "blocked_path_search_timeout",
                    "bisections": history,
                    "failed_variant": name,
                    "timeout_seconds": path_search_timeout,
                }
            try:
                _fixed_path_with_timeout(
                    theta=by_name[name],
                    path=path,
                    pixels=pixels,
                    library=library,
                    threads=threads,
                    config=search_config,
                    timeout_seconds=(
                        path_search_timeout
                        if deadline is None
                        else max(0.0, deadline - time.perf_counter())
                    ),
                )
            except (RuntimeError, TimeoutError) as error:
                failure = {
                    "phase": "path_search_after_oracle",
                    "direction": name,
                    "failure": str(error),
                    "failed_increment": _failure_increment(error),
                    "path_steps": len(path),
                }
                if not bisect(failure):
                    return path, [], {
                        "status": "blocked_path_search",
                        "bisections": [*history, failure],
                        "failed_variant": name,
                    }
                continue
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--library", default="build/mfront/src/libBehaviour.so")
    parser.add_argument("--max-bisections", type=int, default=64)
    parser.add_argument(
        "--seed-timeout",
        type=float,
        default=60.0,
        help="maximum seconds per seed trajectory; zero disables the limit",
    )
    parser.add_argument(
        "--path-search-timeout",
        type=float,
        default=120.0,
        help="maximum seconds per fail-fast path-search variant; zero disables the limit",
    )
    parser.add_argument(
        "--include-b-minus-seed",
        action="store_true",
        help="also attempt the known-expensive b_minus adaptive seed",
    )
    parser.add_argument(
        "--reuse-common-path",
        type=Path,
        default=None,
        help="reuse a previously qualified common_path.npz and skip path search",
    )
    parser.add_argument(
        "--proposal-path",
        type=Path,
        default=None,
        help="use fractions from an unqualified historical path only as a search proposal",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_git_sha = _git("rev-parse HEAD")
    run_dirty = bool(_git("status --porcelain"))
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    theta = _theta_from_preset()
    adaptive: dict[str, list[float]] = {}
    adaptive_diagnostics: dict[str, Any] = {}
    seed_metadata = _seed_metadata(
        pixels=args.pixels,
        library=args.library,
        threads=args.threads,
        git_sha=run_git_sha,
        dirty=run_dirty,
    )
    skipped = set() if args.include_b_minus_seed else set(SEED_SKIP_DEFAULT)
    if args.proposal_path is not None:
        proposal_file = args.proposal_path
        if not proposal_file.is_absolute():
            proposal_file = ROOT / proposal_file
        proposal_fractions = np.asarray(np.load(proposal_file)["end_fractions"], dtype=float)
        adaptive["proposal"] = proposal_fractions.tolist()
        adaptive_diagnostics["proposal"] = {
            "status": "unqualified_proposal",
            "source": str(proposal_file),
            "accepted_increments": int(proposal_fractions.size),
        }
        skipped = {name for name, _ in _variants(theta)}
        print(f"using unqualified proposal path: {proposal_file}", flush=True)
    for name, direction in _variants(theta):
        if args.proposal_path is not None:
            continue
        if name in skipped:
            adaptive_diagnostics[name] = {"status": "seed_skipped", "reason": "known_expensive"}
            print(f"seed path: {name} skipped", flush=True)
            continue
        cached, cache_status = _load_cached_fractions(
            name, metadata=seed_metadata, theta=direction
        )
        if cached is not None:
            adaptive[name] = cached
            adaptive_diagnostics[name] = {
                "status": "cache_hit",
                "accepted_increments": len(cached),
                "end_fractions": cached,
                "cache_status": cache_status,
            }
            print(f"seed path: {name} cache hit", flush=True)
            continue
        print(f"seed path: {name} ({cache_status})", flush=True)
        try:
            fields, diagnostics, _ = _adaptive_with_timeout(
                theta=direction,
                pixels=args.pixels,
                library=args.library,
                threads=args.threads,
                timeout_seconds=args.seed_timeout,
                config=_seed_config(),
            )
        except TimeoutError as error:
            adaptive_diagnostics[name] = {
                "status": "seed_timeout",
                "timeout_seconds": args.seed_timeout,
                "error": str(error),
            }
            print(f"seed path: {name} unavailable ({error})", flush=True)
            continue
        fractions = [float(field.end_fraction) for field in fields]
        adaptive[name] = fractions
        _write_cached_fractions(
            name,
            fractions,
            metadata=seed_metadata,
            theta=direction,
            diagnostics=diagnostics,
        )
        adaptive_diagnostics[name] = {
            "status": "computed",
            "accepted_increments": len(fields),
            "end_fractions": fractions,
            "solver": diagnostics["solver"],
        }
    if "base" not in adaptive and args.proposal_path is None:
        raise SystemExit("base seed path is required to define scored endpoints")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    (CACHE_ROOT / "manifest.json").write_text(
        json.dumps(
            {
                **seed_metadata,
                "seed_skipped_directions": sorted(skipped),
                "seed_status": adaptive_diagnostics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if args.reuse_common_path is not None:
        path_file = args.reuse_common_path
        if not path_file.is_absolute():
            path_file = ROOT / path_file
        fractions = np.asarray(np.load(path_file)["end_fractions"], dtype=np.float64)
        common = _common_path(fractions.tolist(), pixels=args.pixels)
        base_fields = _fixed_path_trajectory(
            theta=theta,
            path=common,
            initial_displacement=None,
            pixels=args.pixels,
            library=args.library,
            threads=args.threads,
            config=_oracle_config(),
        )
        sync = {
            "status": "converged",
            "bisections": [],
            "reused_from": str(path_file),
        }
    else:
        common, base_fields, sync = _synchronise(
            adaptive=adaptive,
            variants=_variants(theta),
            pixels=args.pixels,
            library=args.library,
            threads=args.threads,
            max_bisections=args.max_bisections,
            path_search_timeout=args.path_search_timeout,
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "direct FEMU sensitivity versus synchronized common-path FD",
        "git_sha": run_git_sha,
        "dirty": run_dirty,
        "machine": platform.node(),
        "fd_step_log": FD_STEP,
        "policies": {
            "seed": "_seed_config",
            "path_search": "_path_search_config",
            "oracle": "_oracle_config",
            "seed_timeout_seconds": args.seed_timeout,
            "path_search_timeout_seconds": args.path_search_timeout,
            "seed_skipped_directions": sorted(skipped),
        },
        "adaptive_paths": adaptive_diagnostics,
        "common_path": {
            "status": sync["status"],
            "steps": len(common),
            "end_fractions": [step.end_fraction for step in common],
            "bisections": sync["bisections"],
            **({"reused_from": sync["reused_from"]} if "reused_from" in sync else {}),
        },
        "claims": {
            "common_path_fd_available": False,
            "direct_femu_qualified": False,
            "p43_authorized": False,
        },
        "supersedes": [
            "validation/reference_data/srix_femu_common_path_gate_v9",
            "validation/reference_data/srix_femu_path_convergence_v2",
        ],
        "initialization_contract": 2,
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
    # The historical audit stores only accepted-step indices, not their
    # fractions.  The common-path gate must not index the shorter seed path by
    # those old counts.  Use the documented normalized positions as a
    # provenance-preserving fallback; direct and FD then score identical
    # common-path endpoints.
    target_fractions = [
        float(index) / float(max(scored_source)) for index in scored_source
    ]
    scored = tuple(
        dict.fromkeys(
            int(np.argmin([abs(field.end_fraction - target) for field in base_fields])) + 1
            for target in target_fractions
        )
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
            "target_fractions_normalized_from_archived_indices": target_fractions,
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
