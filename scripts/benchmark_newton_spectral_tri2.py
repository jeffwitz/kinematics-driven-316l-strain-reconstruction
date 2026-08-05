"""Compare CPS4 Newton with the traditional two-state TRI2 spectral solver."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import fields
from pathlib import Path

from fem_inhouse.spectral2d.transforms import SpectralTransformConfig

try:
    from scripts.qualify_ebi_state_sharing import solve_two_state
    from scripts.qualify_ebi_tet_against_cps4 import solve_cps4
    from scripts.qualify_spectral2d_against_newton import build_case
except ModuleNotFoundError:  # Direct script execution.
    from qualify_ebi_state_sharing import solve_two_state  # type: ignore[no-redef]
    from qualify_ebi_tet_against_cps4 import solve_cps4  # type: ignore[no-redef]
    from qualify_spectral2d_against_newton import build_case  # type: ignore[no-redef]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def _run_variant(
    name: str,
    case: dict,
    library: str,
    increments: int,
    tolerance: float,
    repeats: int,
) -> dict:
    samples = []
    for repeat in range(repeats):
        started = time.perf_counter()
        if name == "cps4_newton":
            result = solve_cps4(case, library, increments, tolerance)
            spectral_timings = {}
            solver_timings = {
                field.name: getattr(result.diagnostics, field.name)
                for field in fields(result.diagnostics)
                if field.name.endswith("_seconds")
            }
        else:
            backend = "scipy" if name == "tri2_scipy" else "fftw"
            result, _ = solve_two_state(
                case,
                library,
                increments,
                tolerance,
                1.0,
                transform=SpectralTransformConfig(
                    backend=backend,
                    workers=1,
                    fftw_planner_effort="estimate",
                    fftw_use_wisdom=False,
                ),
            )
            spectral_timings = {
                key: value
                for key, value in result.diagnostics.timings.items()
                if key.endswith("_seconds")
            }
            solver_timings = {}
        elapsed = time.perf_counter() - started
        if name == "cps4_newton":
            iterations = int(result.diagnostics.total_newton_iterations)
            material_evaluations = None
            linear_factorizations = int(result.diagnostics.pardiso_factorization_calls)
        else:
            iterations = sum(result.diagnostics.iterations_per_increment)
            material_evaluations = int(result.diagnostics.timings["material_evaluations"])
            linear_factorizations = None
        samples.append(
            {
                "repeat": repeat,
                "elapsed_seconds": elapsed,
                "iterations": iterations,
                "material_evaluations": material_evaluations,
                "linear_factorizations": linear_factorizations,
                "solver_timings": solver_timings,
                "spectral_timings": spectral_timings,
            }
        )
    result = {
        "name": name,
        "repeats": samples,
        "elapsed_seconds": _summary([sample["elapsed_seconds"] for sample in samples]),
        "iterations": (
            _summary(
                [
                    float(sample["iterations"])
                    for sample in samples
                    if sample["iterations"] is not None
                ]
            )
            if samples[0]["iterations"] is not None
            else None
        ),
        "material_evaluations": (
            _summary([float(sample["material_evaluations"]) for sample in samples])
            if samples[0]["material_evaluations"] is not None
            else None
        ),
    }
    if samples[0]["solver_timings"]:
        keys = samples[0]["solver_timings"]
        result["solver_timings"] = {
            key: _summary([float(sample["solver_timings"][key]) for sample in samples])
            for key in keys
        }
    if samples[0]["spectral_timings"]:
        keys = samples[0]["spectral_timings"]
        result["spectral_timings"] = {
            key: _summary([float(sample["spectral_timings"][key]) for sample in samples])
            for key in keys
        }
    if samples[0]["linear_factorizations"] is not None:
        result["linear_factorizations"] = _summary(
            [float(sample["linear_factorizations"]) for sample in samples]
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=int, default=12)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("cps4_newton", "tri2_scipy", "tri2_fftw"),
        default=("tri2_fftw", "cps4_newton", "tri2_scipy"),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        raise SystemExit("--repeats must be positive")
    case = build_case(arguments.mesh)
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", "build/mfront/src/libBehaviour.so")
    report = {
        "status": "completed_newton_spectral_tri2_timing",
        "mesh": arguments.mesh,
        "increments": arguments.increments,
        "tolerance": arguments.tolerance,
        "repeats": arguments.repeats,
        "variants": [
            _run_variant(
                variant,
                case,
                library,
                arguments.increments,
                arguments.tolerance,
                arguments.repeats,
            )
            for variant in arguments.variants
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
