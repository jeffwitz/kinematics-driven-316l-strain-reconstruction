#!/usr/bin/env python3
"""Benchmark the three plane-stress constitutive paths in isolated FEM processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from fem_inhouse.config import CaseStudyConfig, MaterialConfig, MeshConfig, SolverConfig
from fem_inhouse.solver import run_case_study

BACKENDS = (
    "python",
    "mfront-native-plane-stress",
    "mfront-3d-condensed-plane-stress",
)
FIELD_NAMES = (
    "U",
    "S",
    "E",
    "PE",
    "PEEQ",
    "RF",
    "S_3D",
    "E_3D",
    "EE_3D",
    "PE_3D",
    "PLANE_STRESS_RESIDUAL_MPA",
    "S33_RESIDUAL_MPA",
)
COMPARISON_FIELDS = (
    "U",
    "S",
    "E",
    "PE",
    "PEEQ",
    "RF",
    "S_3D",
    "E_3D",
    "EE_3D",
    "PE_3D",
)
RELATIVE_LINF_LIMITS = {
    "U": 1e-6,
    "S": 5e-4,
    "E": 5e-4,
    "PE": 1e-3,
    "PEEQ": 1e-3,
    "RF": 5e-4,
    "S_3D": 5e-4,
    "E_3D": 5e-4,
    "EE_3D": 5e-4,
    "PE_3D": 1e-3,
}
INPUT_NAMES = (
    "displacement_x_mm",
    "displacement_y_mm",
    "yield_stress_mpa",
    "hardening_coefficient_mpa",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return None
    for line in cpuinfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def _load_inputs(
    directory: Path,
    *,
    mmap_mode: Literal["r"] | None = None,
) -> dict[str, np.ndarray]:
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode=mmap_mode) for name in INPUT_NAMES}
    element_shape = arrays["yield_stress_mpa"].shape
    if len(element_shape) != 2:
        raise ValueError("material maps must be two-dimensional")
    if arrays["hardening_coefficient_mpa"].shape != element_shape:
        raise ValueError("material maps do not share the same shape")
    nodal_shape = (element_shape[0] + 1, element_shape[1] + 1)
    if arrays["displacement_x_mm"].shape != nodal_shape:
        raise ValueError("displacement_x_mm is incompatible with the material maps")
    if arrays["displacement_y_mm"].shape != nodal_shape:
        raise ValueError("displacement_y_mm is incompatible with the material maps")
    return arrays


def _prepare_crop(source: Path, destination: Path, nx: int, ny: int) -> dict[str, Any]:
    arrays = _load_inputs(source, mmap_mode="r")
    source_nx, source_ny = arrays["yield_stress_mpa"].shape
    if nx < 1 or ny < 1 or nx > source_nx or ny > source_ny:
        raise ValueError("requested crop does not fit inside the prepared input")
    x0 = (source_nx - nx) // 2
    y0 = (source_ny - ny) // 2
    x1 = x0 + nx
    y1 = y0 + ny
    destination.mkdir(parents=True, exist_ok=False)
    slices = {
        "displacement_x_mm": (slice(x0, x1 + 1), slice(y0, y1 + 1)),
        "displacement_y_mm": (slice(x0, x1 + 1), slice(y0, y1 + 1)),
        "yield_stress_mpa": (slice(x0, x1), slice(y0, y1)),
        "hardening_coefficient_mpa": (slice(x0, x1), slice(y0, y1)),
    }
    files: dict[str, dict[str, Any]] = {}
    for name, field_slice in slices.items():
        output = destination / f"{name}.npy"
        np.save(output, np.asarray(arrays[name][field_slice]))
        files[name] = {
            "filename": output.name,
            "sha256": _sha256(output),
            "shape": list(np.load(output, mmap_mode="r").shape),
        }
    source_manifest = source / "manifest.json"
    manifest = {
        "schema_version": 1,
        "source_directory": str(source),
        "source_manifest_sha256": (_sha256(source_manifest) if source_manifest.is_file() else None),
        "source_element_shape": [source_nx, source_ny],
        "crop_element_bounds": {"x": [x0, x1], "y": [y0, y1]},
        "crop_element_shape": [nx, ny],
        "files": files,
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def _result_arrays(result: Any) -> dict[str, np.ndarray]:
    required = {
        "U": result.displacement_mm,
        "S": result.stress_mpa,
        "E": result.total_strain,
        "PE": result.plastic_strain,
        "PEEQ": result.equivalent_plastic_strain,
        "RF": result.reaction_force,
        "S_3D": result.stress_tensor_mpa,
        "E_3D": result.total_strain_tensor,
        "EE_3D": result.elastic_strain_tensor,
        "PE_3D": result.plastic_strain_tensor,
        "PLANE_STRESS_RESIDUAL_MPA": result.plane_stress_residual_vector_mpa,
        "S33_RESIDUAL_MPA": result.plane_stress_residual_mpa,
    }
    missing = [name for name, values in required.items() if values is None]
    if missing:
        raise RuntimeError(f"solver result is missing required fields: {missing}")
    return {name: np.asarray(values) for name, values in required.items()}


def _run_worker(args: argparse.Namespace) -> int:
    inputs = _load_inputs(args.input)
    nx, ny = inputs["yield_stress_mpa"].shape
    config = CaseStudyConfig(
        mesh=MeshConfig(nx=nx, ny=ny),
        material=MaterialConfig(),
        solver=SolverConfig(
            increments=args.increments,
            max_newton_iterations=args.max_newton_iterations,
            residual_tolerance=args.residual_tolerance,
            hardening_mode="ludwik",
            constitutive_backend=args.backend,
            mfront_library=str(args.library),
            mfront_threads=args.mfront_threads,
        ),
    )
    result = run_case_study(
        config,
        displacement_x_mm=inputs["displacement_x_mm"],
        displacement_y_mm=inputs["displacement_y_mm"],
        yield_stress_mpa=inputs["yield_stress_mpa"],
        hardening_coefficient_mpa=inputs["hardening_coefficient_mpa"],
    )
    fields = _result_arrays(result)
    args.worker_output.mkdir(parents=True, exist_ok=True)
    fields_path = args.worker_output / "fields.npz"
    temporary_fields = fields_path.with_suffix(".npz.tmp")
    with temporary_fields.open("wb") as stream:
        np.savez_compressed(stream, **fields)  # type: ignore[arg-type]
    temporary_fields.replace(fields_path)
    diagnostics = asdict(result.diagnostics) if result.diagnostics is not None else None
    worker_report = {
        "backend": args.backend,
        "repeat": args.repeat,
        "configuration": asdict(config),
        "diagnostics": diagnostics,
        "fields": {
            "filename": fields_path.name,
            "sha256": _sha256(fields_path),
            "names": list(fields),
        },
    }
    _write_json(args.worker_output / "worker.json", worker_report)
    return 0


def _parse_resource_usage(path: Path) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    integer_keys = {"maximum_resident_set_kib", "exit_status"}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        values[key] = int(value) if key in integer_keys else float(value)
    return values


def _field_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    difference = prediction - reference
    tiny = np.finfo(float).tiny
    return {
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "relative_linf": float(
            np.max(np.abs(difference)) / max(float(np.max(np.abs(reference))), tiny)
        ),
        "relative_l2": float(
            np.linalg.norm(difference) / max(float(np.linalg.norm(reference)), tiny)
        ),
    }


def _median(records: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(record[key]) for record in records))


def _summarize(
    records: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for backend, backend_records in records.items():
        wall_values = [float(record["resource"]["wall_seconds"]) for record in backend_records]
        solver_values = [
            float(record["worker"]["diagnostics"]["elapsed_seconds"]) for record in backend_records
        ]
        constitutive_values = [
            float(record["worker"]["diagnostics"]["constitutive_seconds"])
            for record in backend_records
        ]
        rss_values = [
            float(record["resource"]["maximum_resident_set_kib"]) for record in backend_records
        ]
        summary[backend] = {
            "median_process_wall_seconds": float(statistics.median(wall_values)),
            "minimum_process_wall_seconds": min(wall_values),
            "maximum_process_wall_seconds": max(wall_values),
            "median_solver_seconds": float(statistics.median(solver_values)),
            "median_constitutive_seconds": float(statistics.median(constitutive_values)),
            "median_peak_rss_kib": float(statistics.median(rss_values)),
            "median_peak_rss_mib": float(statistics.median(rss_values)) / 1024.0,
            "maximum_peak_rss_kib": max(rss_values),
            "maximum_peak_rss_mib": max(rss_values) / 1024.0,
            "median_newton_iterations": _median(
                [
                    {"value": record["worker"]["diagnostics"]["total_newton_iterations"]}
                    for record in backend_records
                ],
                "value",
            ),
        }
    return summary


def _comparison(output: Path) -> tuple[dict[str, Any], dict[str, float]]:
    first_fields: dict[str, Any] = {}
    for backend in BACKENDS:
        first_run = output / "runs" / f"repeat-00-{backend}" / "fields.npz"
        first_fields[backend] = np.load(first_run)
    reference = first_fields["mfront-native-plane-stress"]
    comparison: dict[str, Any] = {}
    residuals: dict[str, float] = {}
    try:
        for backend in BACKENDS:
            residuals[backend] = float(
                np.max(np.abs(first_fields[backend]["PLANE_STRESS_RESIDUAL_MPA"]))
            )
            if backend == "mfront-native-plane-stress":
                continue
            comparison[backend] = {
                name: _field_metrics(reference[name], first_fields[backend][name])
                for name in COMPARISON_FIELDS
            }
    finally:
        for fields in first_fields.values():
            fields.close()
    return comparison, residuals


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("build/mfront/src/libBehaviour.so"),
    )
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--increments", type=int, default=20)
    parser.add_argument("--max-newton-iterations", type=int, default=25)
    parser.add_argument("--residual-tolerance", type=float, default=1e-7)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mfront-threads", type=int, default=2)
    parser.add_argument("--linear-threads", type=int, default=2)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


def _run_benchmark(args: argparse.Namespace) -> int:
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.mfront_threads < 1 or args.linear_threads < 1:
        raise ValueError("thread counts must be positive")
    if not args.library.is_file():
        raise FileNotFoundError(f"MFront behaviour library not found: {args.library}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty campaign: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    crop_manifest = _prepare_crop(args.input, args.output / "input", args.nx, args.ny)
    runs_directory = args.output / "runs"
    runs_directory.mkdir()
    benchmark_config = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "backends": list(BACKENDS),
        "repeats": args.repeats,
        "increments": args.increments,
        "max_newton_iterations": args.max_newton_iterations,
        "residual_tolerance": args.residual_tolerance,
        "mfront_threads": args.mfront_threads,
        "linear_threads": args.linear_threads,
        "mfront_library": {
            "path": str(args.library),
            "sha256": _sha256(args.library),
        },
        "input": crop_manifest,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
        },
    }
    _write_json(args.output / "benchmark-config.json", benchmark_config)

    orders = (
        BACKENDS,
        tuple(reversed(BACKENDS)),
        (BACKENDS[1], BACKENDS[2], BACKENDS[0]),
    )
    records: dict[str, list[dict[str, Any]]] = {backend: [] for backend in BACKENDS}
    time_binary = Path("/usr/bin/time")
    if not time_binary.is_file():
        raise FileNotFoundError("/usr/bin/time is required for peak-RSS measurement")
    for repeat in range(args.repeats):
        for backend in orders[repeat % len(orders)]:
            run_directory = runs_directory / f"repeat-{repeat:02d}-{backend}"
            run_directory.mkdir()
            resource_path = run_directory / "resource-usage.txt"
            command = [
                str(time_binary),
                "-f",
                (
                    "wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\n"
                    "maximum_resident_set_kib=%M\nexit_status=%x"
                ),
                "-o",
                str(resource_path),
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--input",
                str(args.output / "input"),
                "--output",
                str(args.output),
                "--library",
                str(args.library.resolve()),
                "--increments",
                str(args.increments),
                "--max-newton-iterations",
                str(args.max_newton_iterations),
                "--residual-tolerance",
                str(args.residual_tolerance),
                "--mfront-threads",
                str(args.mfront_threads),
                "--backend",
                backend,
                "--repeat",
                str(repeat),
                "--worker-output",
                str(run_directory),
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    "OMP_NUM_THREADS": str(args.linear_threads),
                    "MKL_NUM_THREADS": str(args.linear_threads),
                    "OPENBLAS_NUM_THREADS": "1",
                }
            )
            with (
                (run_directory / "stdout.log").open("w", encoding="utf-8") as stdout,
                (run_directory / "stderr.log").open("w", encoding="utf-8") as stderr,
            ):
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=stdout,
                    stderr=stderr,
                    env=environment,
                )
            resource = _parse_resource_usage(resource_path)
            worker_path = run_directory / "worker.json"
            record = {
                "backend": backend,
                "repeat": repeat,
                "returncode": completed.returncode,
                "resource": resource,
                "worker": (
                    json.loads(worker_path.read_text(encoding="utf-8"))
                    if worker_path.is_file()
                    else None
                ),
            }
            records[backend].append(record)
            _write_json(args.output / "runs.json", records)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"benchmark worker failed for {backend}, repeat {repeat}; "
                    f"see {run_directory / 'stderr.log'}"
                )

    summary = _summarize(records)
    comparison, residuals = _comparison(args.output)
    reference = summary["mfront-native-plane-stress"]
    ratios = {
        backend: {
            "process_wall_vs_native": (
                values["median_process_wall_seconds"] / reference["median_process_wall_seconds"]
            ),
            "solver_time_vs_native": (
                values["median_solver_seconds"] / reference["median_solver_seconds"]
            ),
            "constitutive_time_vs_native": (
                values["median_constitutive_seconds"] / reference["median_constitutive_seconds"]
            ),
            "peak_rss_vs_native": (
                values["median_peak_rss_kib"] / reference["median_peak_rss_kib"]
            ),
        }
        for backend, values in summary.items()
    }
    parity_passed = all(
        metrics["relative_linf"] <= RELATIVE_LINF_LIMITS[field]
        for backend_metrics in comparison.values()
        for field, metrics in backend_metrics.items()
    )
    convergence_passed = all(
        int(record["worker"]["diagnostics"]["cutbacks"]) == 0
        and int(record["worker"]["diagnostics"]["local_plane_stress_failures"]) == 0
        for backend_records in records.values()
        for record in backend_records
    )
    report = {
        **benchmark_config,
        "schema_version": 2,
        "orders": [list(order) for order in orders],
        "raw_runs": records,
        "summary": summary,
        "ratios": ratios,
        "comparison_against_mfront_native": comparison,
        "relative_linf_limits": RELATIVE_LINF_LIMITS,
        "maximum_plane_stress_residual_mpa": residuals,
        "checks": {
            "parity_passed": parity_passed,
            "convergence_passed": convergence_passed,
        },
        "passed": parity_passed and convergence_passed,
    }
    _write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        if args.backend is None or args.worker_output is None:
            raise ValueError("worker mode requires --backend and --worker-output")
        return _run_worker(args)
    return _run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
