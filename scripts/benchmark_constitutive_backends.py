#!/usr/bin/env python3
"""Benchmark Python and MFront constitutive updates on identical point batches."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, process_time

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from fem_inhouse.core.constitutive import (
    PLANE_STRESS_VON_MISES_METRIC,
    consistent_tangent,
    make_hardening,
    return_mapping,
)
from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.core.mfront import MFrontMaterialPointBatch

DEFAULT_LIBRARY = Path("build/mfront/src/libBehaviour.so")
DEFAULT_SOURCE = Path("mfront/PixelLudwikJ2Plasticity.mfront")


@dataclass(frozen=True, slots=True)
class Workload:
    yield_stress_mpa: NDArray
    hardening_coefficient_mpa: NDArray
    hardening_exponent: NDArray
    terminal_strain: NDArray
    increments: int


@dataclass(frozen=True, slots=True)
class BackendState:
    stress_mpa: NDArray
    plastic_strain: NDArray
    equivalent_plastic_strain: NDArray
    tangent_mpa: NDArray


@dataclass(frozen=True, slots=True)
class Timing:
    setup_wall_seconds: float
    integration_wall_seconds: float
    integration_cpu_seconds: float
    state: BackendState


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return None
    for line in cpuinfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.split(":", maxsplit=1)[1].strip()
    return None


def _make_workload(point_count: int, increments: int) -> Workload:
    indices = np.arange(point_count, dtype=float)
    yield_stress = 240.0 + 20.0 * np.sin(indices * 0.001)
    hardening_coefficient = 380.0 + 60.0 * np.cos(indices * 0.0007)
    exponent = np.full(point_count, 0.245)
    amplitude = 0.85 + 0.30 * ((indices % 101.0) / 100.0)
    terminal_strain = np.column_stack(
        (
            0.010 * amplitude,
            -0.001 * amplitude,
            0.003 * amplitude,
        )
    )
    return Workload(
        yield_stress_mpa=yield_stress,
        hardening_coefficient_mpa=hardening_coefficient,
        hardening_exponent=exponent,
        terminal_strain=terminal_strain,
        increments=increments,
    )


def _run_python(workload: Workload) -> Timing:
    setup_started = perf_counter()
    elasticity = plane_stress_elasticity(205_000.0, 0.3)
    constitutive_metric = elasticity @ PLANE_STRESS_VON_MISES_METRIC
    cm11 = float(constitutive_metric[0, 0])
    cm12 = float(constitutive_metric[0, 1])
    cm33 = float(constitutive_metric[2, 2])
    hardening, hardening_derivative = make_hardening(0.245, "tabular")
    point_count = len(workload.yield_stress_mpa)
    plastic_strain = np.zeros((point_count, 3))
    equivalent_plastic_strain = np.zeros(point_count)
    stress = np.zeros((point_count, 3))
    tangent = np.repeat(elasticity[None, :, :], point_count, axis=0)
    setup_wall_seconds = perf_counter() - setup_started

    integration_wall_started = perf_counter()
    integration_cpu_started = process_time()
    for increment in range(1, workload.increments + 1):
        total_strain = workload.terminal_strain * (increment / workload.increments)
        trial_stress = (total_strain - plastic_strain) @ elasticity.T
        stress, plastic_increment, equivalent_increment = return_mapping(
            trial_stress,
            equivalent_plastic_strain,
            workload.yield_stress_mpa,
            workload.hardening_coefficient_mpa,
            hardening,
            cm11,
            cm12,
            cm33,
        )
        plastic_indices = np.flatnonzero(equivalent_increment > 0)
        tangent = np.repeat(elasticity[None, :, :], point_count, axis=0)
        if plastic_indices.size:
            tangent[plastic_indices] = consistent_tangent(
                stress[plastic_indices],
                equivalent_increment[plastic_indices],
                equivalent_plastic_strain[plastic_indices],
                workload.yield_stress_mpa[plastic_indices],
                workload.hardening_coefficient_mpa[plastic_indices],
                hardening,
                hardening_derivative,
                elasticity,
                cm11,
                cm12,
                cm33,
            )
        plastic_strain += plastic_increment
        equivalent_plastic_strain += equivalent_increment
    integration_cpu_seconds = process_time() - integration_cpu_started
    integration_wall_seconds = perf_counter() - integration_wall_started
    return Timing(
        setup_wall_seconds=setup_wall_seconds,
        integration_wall_seconds=integration_wall_seconds,
        integration_cpu_seconds=integration_cpu_seconds,
        state=BackendState(
            stress_mpa=stress,
            plastic_strain=plastic_strain,
            equivalent_plastic_strain=equivalent_plastic_strain,
            tangent_mpa=tangent,
        ),
    )


def _run_mfront(
    workload: Workload,
    library: Path,
    *,
    thread_count: int,
) -> Timing:
    setup_started = perf_counter()
    bridge = MFrontMaterialPointBatch(
        library,
        workload.yield_stress_mpa,
        workload.hardening_coefficient_mpa,
        workload.hardening_exponent,
        thread_count=thread_count,
    )
    setup_wall_seconds = perf_counter() - setup_started
    integration_wall_started = perf_counter()
    integration_cpu_started = process_time()
    result = None
    for increment in range(1, workload.increments + 1):
        result = bridge.evaluate(
            workload.terminal_strain * (increment / workload.increments),
            time_increment=1.0 / workload.increments,
            consistent_tangent=True,
            commit=True,
        )
    integration_cpu_seconds = process_time() - integration_cpu_started
    integration_wall_seconds = perf_counter() - integration_wall_started
    assert result is not None
    assert result.consistent_tangent_mpa is not None
    return Timing(
        setup_wall_seconds=setup_wall_seconds,
        integration_wall_seconds=integration_wall_seconds,
        integration_cpu_seconds=integration_cpu_seconds,
        state=BackendState(
            stress_mpa=result.stress_mpa,
            plastic_strain=result.plastic_strain,
            equivalent_plastic_strain=result.equivalent_plastic_strain,
            tangent_mpa=result.consistent_tangent_mpa,
        ),
    )


def _warm_up(library: Path, thread_count: int) -> None:
    workload = _make_workload(2_000, 4)
    _run_python(workload)
    _run_mfront(workload, library, thread_count=1)
    _run_mfront(workload, library, thread_count=thread_count)


def _state_summary(state: BackendState) -> dict[str, float]:
    return {
        "stress_sum_mpa": float(state.stress_mpa.sum()),
        "plastic_strain_sum": float(state.plastic_strain.sum()),
        "peeq_sum": float(state.equivalent_plastic_strain.sum()),
        "tangent_sum_mpa": float(state.tangent_mpa.sum()),
    }


def _timing_record(timing: Timing) -> dict[str, float | dict[str, float]]:
    return {
        "setup_wall_seconds": timing.setup_wall_seconds,
        "integration_wall_seconds": timing.integration_wall_seconds,
        "integration_cpu_seconds": timing.integration_cpu_seconds,
        "state_checksums": _state_summary(timing.state),
    }


def _median(records: list[dict[str, float | dict[str, float]]], key: str) -> float:
    return float(np.median([float(record[key]) for record in records]))


def _plot(report: dict[str, object], output: Path) -> None:
    summaries = report["summary"]
    assert isinstance(summaries, dict)
    labels = ["Python", "MFront série", "MFront parallèle"]
    keys = ["python", "mfront_serial", "mfront_parallel"]
    times = [float(summaries[key]["median_wall_seconds"]) for key in keys]
    throughputs = [
        float(summaries[key]["million_point_updates_per_second"]) for key in keys
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, times)
    axes[0].set_ylabel("Temps médian [s]")
    axes[0].set_title("20 intégrations avec tangente")
    axes[1].bar(labels, throughputs)
    axes[1].set_ylabel("Millions de points x incréments / s")
    axes[1].set_title("Débit constitutif")
    for axis in axes:
        axis.grid(axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    temporary = output.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=160)
    plt.close(figure)
    temporary.replace(output)


def benchmark(
    library: Path,
    source: Path,
    output_directory: Path,
    *,
    point_count: int,
    increments: int,
    repeats: int,
    thread_count: int,
) -> dict[str, object]:
    if point_count < 1:
        raise ValueError("point_count must be positive")
    if increments < 2:
        raise ValueError("increments must be at least 2")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if thread_count < 2:
        raise ValueError("thread_count must be at least 2 for the parallel case")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    workload = _make_workload(point_count, increments)
    _warm_up(library, thread_count)
    records: dict[str, list[dict[str, float | dict[str, float]]]] = {
        "python": [],
        "mfront_serial": [],
        "mfront_parallel": [],
    }
    first_states: dict[str, BackendState] = {}
    orders = [
        ("python", "mfront_serial", "mfront_parallel"),
        ("mfront_parallel", "mfront_serial", "python"),
    ]
    for repetition in range(repeats):
        for backend in orders[repetition % len(orders)]:
            gc.collect()
            if backend == "python":
                timing = _run_python(workload)
            elif backend == "mfront_serial":
                timing = _run_mfront(workload, library, thread_count=1)
            else:
                timing = _run_mfront(
                    workload,
                    library,
                    thread_count=thread_count,
                )
            records[backend].append(_timing_record(timing))
            first_states.setdefault(backend, timing.state)

    point_updates = point_count * increments
    summary: dict[str, dict[str, float]] = {}
    for backend, backend_records in records.items():
        median_wall = _median(backend_records, "integration_wall_seconds")
        summary[backend] = {
            "median_wall_seconds": median_wall,
            "minimum_wall_seconds": min(
                float(record["integration_wall_seconds"]) for record in backend_records
            ),
            "maximum_wall_seconds": max(
                float(record["integration_wall_seconds"]) for record in backend_records
            ),
            "median_cpu_seconds": _median(
                backend_records,
                "integration_cpu_seconds",
            ),
            "million_point_updates_per_second": point_updates / median_wall / 1e6,
            "median_setup_wall_seconds": _median(
                backend_records,
                "setup_wall_seconds",
            ),
        }

    python_time = summary["python"]["median_wall_seconds"]
    serial_time = summary["mfront_serial"]["median_wall_seconds"]
    parallel_time = summary["mfront_parallel"]["median_wall_seconds"]
    serial_state = first_states["mfront_serial"]
    parallel_state = first_states["mfront_parallel"]
    report: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "parallel_mgis_threads": thread_count,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
        },
        "inputs": {
            "mfront_library": _portable_path(library),
            "mfront_library_sha256": _sha256(library),
            "mfront_source": _portable_path(source),
            "mfront_source_sha256": _sha256(source),
            "point_count": point_count,
            "increments": increments,
            "repeats": repeats,
            "point_updates_per_repeat": point_updates,
            "consistent_tangent": True,
            "python_hardening": "1000-point tabular",
            "mfront_hardening": "regularised analytical",
        },
        "method": {
            "warm_up_points": 2_000,
            "warm_up_increments": 4,
            "timed_region": (
                "constitutive integration, state update, tangent computation, "
                "and Python/MGIS result conversion; setup reported separately"
            ),
            "repeat_order": [list(order) for order in orders],
        },
        "raw_repetitions": records,
        "summary": summary,
        "ratios": {
            "mfront_serial_slowdown_vs_python": serial_time / python_time,
            "mfront_parallel_speedup_vs_serial": serial_time / parallel_time,
            "mfront_parallel_speedup_vs_python": python_time / parallel_time,
        },
        "parity": {
            "mfront_parallel_vs_serial_max_stress_mpa": float(
                np.max(np.abs(parallel_state.stress_mpa - serial_state.stress_mpa))
            ),
            "mfront_parallel_vs_serial_max_peeq": float(
                np.max(
                    np.abs(
                        parallel_state.equivalent_plastic_strain
                        - serial_state.equivalent_plastic_strain
                    )
                )
            ),
            "python_vs_mfront_relative_stress_l2": float(
                np.linalg.norm(
                    first_states["python"].stress_mpa - serial_state.stress_mpa
                )
                / np.linalg.norm(serial_state.stress_mpa)
            ),
        },
        "artifacts": {
            "states": "final_states.npz",
            "plot": "timings.png",
        },
    }

    states_path = output_directory / "final_states.npz"
    temporary_states = states_path.with_suffix(".tmp")
    sample_indices = np.linspace(
        0,
        point_count - 1,
        min(point_count, 4_096),
        dtype=int,
    )
    with temporary_states.open("wb") as stream:
        np.savez_compressed(
            stream,
            yield_stress_mpa=workload.yield_stress_mpa,
            hardening_coefficient_mpa=workload.hardening_coefficient_mpa,
            hardening_exponent=workload.hardening_exponent,
            terminal_strain=workload.terminal_strain,
            python_stress_mpa=first_states["python"].stress_mpa,
            python_plastic_strain=first_states["python"].plastic_strain,
            python_peeq=first_states["python"].equivalent_plastic_strain,
            mfront_stress_mpa=serial_state.stress_mpa,
            mfront_plastic_strain=serial_state.plastic_strain,
            mfront_peeq=serial_state.equivalent_plastic_strain,
            tangent_sample_indices=sample_indices,
            python_tangent_sample_mpa=first_states["python"].tangent_mpa[sample_indices],
            mfront_tangent_sample_mpa=serial_state.tangent_mpa[sample_indices],
        )
    temporary_states.replace(states_path)

    report_path = output_directory / "report.json"
    temporary_report = report_path.with_suffix(".tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(report_path)
    _plot(report, output_directory / "timings.png")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=200_000)
    parser.add_argument("--increments", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    arguments = parser.parse_args()
    report = benchmark(
        arguments.library,
        arguments.source,
        arguments.output,
        point_count=arguments.points,
        increments=arguments.increments,
        repeats=arguments.repeats,
        thread_count=arguments.threads,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(arguments.output / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
