"""Benchmark the nonsymmetric TRI2 two-state crystal solver."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from fem_inhouse.spectral2d.transforms import SpectralTransformConfig

try:
    from scripts.qualify_ebi_state_sharing import solve_two_state
    from scripts.qualify_spectral2d_against_newton import build_case
except ModuleNotFoundError:  # Direct script execution.
    from qualify_ebi_state_sharing import solve_two_state  # type: ignore[no-redef]
    from qualify_spectral2d_against_newton import build_case  # type: ignore[no-redef]


def run_case(
    mesh: int,
    behaviour: str,
    method: str,
    threads: int,
    increments: int,
    tolerance: float,
    load_scale: float,
    linear_mode: str,
    reference_update_mode: str,
) -> dict[str, object]:
    case = build_case(mesh)
    case["displacement_x_mm"] = load_scale * case["displacement_x_mm"]
    case["displacement_y_mm"] = load_scale * case["displacement_y_mm"]
    started = time.perf_counter()
    try:
        result, elapsed = solve_two_state(
            case,
            os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", "build/mfront/src/libBehaviour.so"),
            increments,
            tolerance,
            1.0,
            transform=SpectralTransformConfig(
                backend="scipy",
                workers=1,
                fftw_planner_effort="estimate",
                fftw_use_wisdom=False,
            ),
            behaviour_id=behaviour,
            mfront_threads=threads,
            krylov_method=method,
            krylov_recycling=method != "gmres",
            local_condition_check_mode="on_failure",
            linear_mode=linear_mode,
            reference_update_mode=reference_update_mode,
        )
    except Exception as error:
        return {
            "status": "failed",
            "mesh": mesh,
            "behaviour": behaviour,
            "krylov_method": method,
            "mfront_threads": threads,
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.perf_counter() - started,
        }
    diagnostics = result.diagnostics
    return {
        "status": "completed",
        "mesh": mesh,
        "behaviour": behaviour,
        "krylov_method": method,
        "mfront_threads": threads,
        "elapsed_seconds": elapsed,
        "newton_iterations": sum(diagnostics.iterations_per_increment),
        "iterations_per_increment": list(diagnostics.iterations_per_increment),
        "krylov_iterations": int(diagnostics.timings["gmres_iterations"]),
        "final_residual": diagnostics.verification_residual,
        "timings": diagnostics.timings,
        "provenance": diagnostics.provenance,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meshes", nargs="+", type=int, default=[12, 24])
    parser.add_argument(
        "--behaviours",
        nargs="+",
        choices=("fcc_forest_rubin_srix", "fcc_meric_cailletaud"),
        default=["fcc_forest_rubin_srix", "fcc_meric_cailletaud"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("gmres", "lgmres", "gcrotmk"),
        default=["gmres", "lgmres"],
    )
    parser.add_argument("--mfront-threads", nargs="+", type=int, default=[1])
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--load-scale", type=float, default=1.0)
    parser.add_argument("--linear-mode", choices=("fixed", "eisenstat_walker"), default="fixed")
    parser.add_argument(
        "--reference-update",
        choices=("initial", "per_increment", "per_newton"),
        default="initial",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = [
        run_case(
            mesh,
            behaviour,
            method,
            threads,
            arguments.increments,
            arguments.tolerance,
            arguments.load_scale,
            arguments.linear_mode,
            arguments.reference_update,
        )
        for mesh in arguments.meshes
        for behaviour in arguments.behaviours
        for method in arguments.methods
        for threads in arguments.mfront_threads
    ]
    report = {
        "status": "completed_crystal_tet2_sweep",
        "meshes": arguments.meshes,
        "behaviours": arguments.behaviours,
        "records": records,
        "load_scale": arguments.load_scale,
        "linear_mode": arguments.linear_mode,
        "reference_update": arguments.reference_update,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
