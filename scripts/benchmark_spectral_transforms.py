"""Benchmark full-Dirichlet transform and B0-preconditioner backends."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import scipy

from fem_inhouse.spectral2d.green import B0Green2D, ReferenceOperatorSymbols
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.transform_factory import create_full_dirichlet_dsti_plan
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig


def _stats(values: list[float]) -> dict[str, float]:
    median = statistics.median(values)
    return {
        "minimum": min(values),
        "median": median,
        "mean": statistics.fmean(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "mad": statistics.median([abs(value - median) for value in values]),
        "p95": float(np.percentile(values, 95)),
    }


def _measure(function, warmups: int, samples: int, applications: int) -> dict[str, float]:
    for _ in range(warmups):
        for _ in range(applications):
            function()
    values = []
    for _ in range(samples):
        started = time.perf_counter()
        for _ in range(applications):
            function()
        values.append((time.perf_counter() - started) / applications)
    return _stats(values)


def _current_loop(field: np.ndarray) -> None:
    from scipy.fft import dstn, idstn

    for component in range(2):
        transformed = dstn(field[..., component], type=1, norm="ortho", axes=(0, 1), workers=1)
        idstn(transformed, type=1, norm="ortho", axes=(0, 1), workers=1)


def _pair(plan, field: np.ndarray, spectral: np.ndarray, physical: np.ndarray) -> None:
    plan.forward_into(field, spectral)
    plan.inverse_into(spectral, physical)


def _preconditioner(
    plan,
    green,
    field: np.ndarray,
    spectral: np.ndarray,
    green_output: np.ndarray,
    physical: np.ndarray,
) -> None:
    plan.forward_into(field, spectral)
    green.apply_into(spectral, green_output)
    plan.inverse_into(green_output, physical)


def _variant_specs(
    backends: list[str], threads: list[int]
) -> list[tuple[str, SpectralTransformConfig]]:
    specs: list[tuple[str, SpectralTransformConfig]] = []
    if "scipy" in backends:
        specs.extend(
            [
                ("scipy-current-loop", SpectralTransformConfig(backend="scipy", workers=1)),
                ("scipy-batched-workers-1", SpectralTransformConfig(backend="scipy", workers=1)),
                ("scipy-batched-workers-2", SpectralTransformConfig(backend="scipy", workers=2)),
            ]
        )
    if "fftw" in backends:
        for effort in ("estimate", "measure", "patient"):
            for worker_count in threads:
                specs.append(
                    (
                        f"fftw-{effort}-{worker_count}",
                        SpectralTransformConfig(
                            backend="fftw",
                            workers=worker_count,
                            fftw_planner_effort=effort,
                        ),
                    )
                )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meshes", nargs="+", type=int, default=[12, 24, 48, 96, 192])
    parser.add_argument("--backends", nargs="+", choices=("scipy", "fftw"), default=["scipy"])
    parser.add_argument("--fftw-threads", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--applications", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if "fftw" in arguments.backends and importlib.util.find_spec("pyfftw") is None:
        raise SystemExit(
            "FFTW benchmark requested, but pyFFTW is not installed. "
            "Install the project with the 'fftw' optional dependency."
        )
    records = []
    for mesh in arguments.meshes:
        grid = StructuredGrid2D(mesh, mesh, 1.0, 1.0)
        field = np.random.default_rng(mesh).normal(size=(*grid.interior_shape, 2))
        symbols = ReferenceOperatorSymbols(
            np.ones(grid.interior_shape),
            np.ones(grid.interior_shape),
            np.ones(grid.interior_shape),
        )
        for name, config in _variant_specs(arguments.backends, arguments.fftw_threads):
            started = time.perf_counter()
            plan = create_full_dirichlet_dsti_plan(grid, config)
            planning_seconds = time.perf_counter() - started
            spectral = np.empty_like(field)
            physical = np.empty_like(field)
            green = B0Green2D(symbols, lambda_0=2.0, mu_0=3.0)
            green_output = np.empty_like(field)

            pair = partial(_pair, plan, field, spectral, physical)
            preconditioner = partial(
                _preconditioner, plan, green, field, spectral, green_output, physical
            )

            forward_inverse = _measure(
                pair, arguments.warmups, arguments.samples, arguments.applications
            )
            preconditioner_time = _measure(
                preconditioner, arguments.warmups, arguments.samples, arguments.applications
            )
            if name == "scipy-current-loop":
                forward_inverse = _measure(
                    lambda field=field: _current_loop(field),
                    arguments.warmups,
                    arguments.samples,
                    arguments.applications,
                )
            records.append(
                {
                    "mesh": mesh,
                    "interior_shape": list(field.shape),
                    "variant": name,
                    "transform": asdict(plan.diagnostics),
                    "planning_seconds_wall": planning_seconds,
                    "forward_inverse_seconds": forward_inverse,
                    "preconditioner_seconds": preconditioner_time,
                }
            )
    report = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyfftw": (
                __import__("pyfftw").__version__
                if importlib.util.find_spec("pyfftw") is not None
                else None
            ),
        },
        "method": {
            "warmups": arguments.warmups,
            "samples": arguments.samples,
            "applications": arguments.applications,
        },
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
