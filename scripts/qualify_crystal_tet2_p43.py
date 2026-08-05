"""Run the condensed 3D crystal TRI2 solver on the registered P43 crop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch
from fem_inhouse.spectral2d import EBISpectralSolverConfig
from fem_inhouse.spectral2d.newton_two_state import solve_two_state_dirichlet_plane_stress
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig

try:
    from scripts.benchmark_tri2_j2_krylov import DEFAULT_CROP, _load_case
except ModuleNotFoundError:  # Direct script execution.
    from benchmark_tri2_j2_krylov import DEFAULT_CROP, _load_case  # type: ignore[no-redef]


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _git_head() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            or None
        )
    except OSError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=DEFAULT_CROP)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument(
        "--behaviour",
        choices=("fcc_forest_rubin_srix", "fcc_meric_cailletaud"),
        default="fcc_forest_rubin_srix",
    )
    parser.add_argument("--mfront-threads", type=int, default=4)
    parser.add_argument(
        "--krylov-method",
        choices=("gmres", "lgmres", "gcrotmk"),
        default="lgmres",
    )
    parser.add_argument("--krylov-recycling", action="store_true", default=True)
    parser.add_argument(
        "--linear-mode",
        choices=("fixed", "eisenstat_walker"),
        default="eisenstat_walker",
    )
    parser.add_argument(
        "--reference-update",
        choices=("initial", "per_increment", "per_newton"),
        default="initial",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    crop = tuple(arguments.crop_nodes)
    mesh = crop[1] - crop[0]
    if mesh != crop[3] - crop[2]:
        raise SystemExit("P43 crop must be square")
    grid, history, yield_stress, coefficient, _ = _load_case(mesh, crop)
    material = create_plane_stress_material_batch(
        "mfront-3d-condensed-plane-stress",
        np.repeat(yield_stress, 2),
        np.repeat(coefficient, 2),
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1_000,
        first_positive_plastic_strain=1.0e-6,
        mfront_library=os.environ.get(
            "MFRONT_BEHAVIOUR_LIBRARY", "build/mfront/src/libBehaviour.so"
        ),
        mfront_threads=arguments.mfront_threads,
        mfront_behaviour_id=arguments.behaviour,
        local_plane_stress_options={"local_condition_check_mode": "on_failure"},
        constitutive_options={
            "crystal_orientation": {
                "mode": "homogeneous",
                "euler_bunge_deg": [35.0, 20.0, 15.0],
            }
        },
    )
    started = time.perf_counter()
    result = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=material,
        boundary_displacement_history=history[: arguments.increments + 1],
        config=EBISpectralSolverConfig(
            relative_equilibrium_tolerance=arguments.tolerance,
            linear_tolerance_mode=arguments.linear_mode,
            reference_update_mode=arguments.reference_update,
            krylov_method=arguments.krylov_method,
            krylov_recycling=arguments.krylov_recycling,
            transform=SpectralTransformConfig(
                backend="fftw",
                workers=1,
                fftw_planner_effort="measure",
                fftw_planning_time_limit_s=2.0,
                fftw_use_wisdom=False,
            ),
        ),
    )
    elapsed = time.perf_counter() - started
    fields = {
        "displacement": result.displacement,
        "stress_in_plane_mpa": result.stress_in_plane_mpa,
        "reaction_forces": result.reaction_forces,
        "accumulated_slip": result.observables["accumulated_slip"],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    field_path = arguments.output.with_suffix(".fields.npz")
    np.savez_compressed(field_path, **fields)
    diagnostics = result.diagnostics
    report = {
        "status": "completed_crystal_tet2_p43",
        "crop_nodes": list(crop),
        "mesh": [mesh, mesh],
        "increments": arguments.increments,
        "tolerance": arguments.tolerance,
        "behaviour": arguments.behaviour,
        "mfront_threads": arguments.mfront_threads,
        "krylov_method": arguments.krylov_method,
        "linear_mode": arguments.linear_mode,
        "reference_update": arguments.reference_update,
        "elapsed_seconds": elapsed,
        "newton_iterations": sum(diagnostics.iterations_per_increment),
        "iterations_per_increment": list(diagnostics.iterations_per_increment),
        "krylov_iterations": int(diagnostics.timings["gmres_iterations"]),
        "krylov_outer_callbacks": int(
            diagnostics.timings["krylov_outer_callbacks"]
        ),
        "jacobian_matvec_calls": int(
            diagnostics.timings["jacobian_matvec_calls"]
        ),
        "preconditioner_calls": int(diagnostics.timings["preconditioner_calls"]),
        "final_residual": diagnostics.verification_residual,
        "timings": diagnostics.timings,
        "provenance": diagnostics.provenance,
        "execution_commit": diagnostics.provenance.get("commit_sha"),
        "archive_commit": os.environ.get("ARCHIVE_COMMIT", _git_head()),
        "field_file": str(field_path),
        "field_sha256": {name: _hash(values) for name, values in fields.items()},
    }
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
