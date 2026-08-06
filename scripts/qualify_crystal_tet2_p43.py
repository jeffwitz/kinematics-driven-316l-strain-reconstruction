"""Run the condensed 3D crystal TRI2 solver on the registered P43 crop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import cast

import numpy as np

from fem_inhouse.core.crystal_parameter_pairs import (
    CrystalLaw,
    resolve_paired_crystal_parameters,
)
from fem_inhouse.core.mfront_crystal_structure import read_crystal_structure_fingerprint
from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch
from fem_inhouse.spectral2d import (
    AdaptiveStepConfig,
    EBISpectralSolverConfig,
    StepDoublingErrorConfig,
)
from fem_inhouse.spectral2d.newton_two_state import solve_two_state_dirichlet_plane_stress
from fem_inhouse.spectral2d.step_doubling import StepDoublingFailureError
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig

try:
    from scripts.benchmark_tri2_j2_krylov import DEFAULT_CROP, _load_case
except ModuleNotFoundError:  # Direct script execution.
    from benchmark_tri2_j2_krylov import (  # type: ignore[import-not-found,no-redef]
        DEFAULT_CROP,
        _load_case,
    )


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


def _git_worktree_state() -> dict[str, object]:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
        diff = subprocess.run(
            ["git", "diff", "--binary"],
            check=False,
            capture_output=True,
        )
        return {
            "working_tree_dirty_at_generation": bool(status.stdout.strip()),
            "working_tree_diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
        }
    except OSError:
        return {
            "working_tree_dirty_at_generation": None,
            "working_tree_diff_sha256": None,
        }


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
    parser.add_argument("--paired-parameter-set", required=True)
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
    parser.add_argument(
        "--no-final-verification",
        action="store_true",
        help="promote the accepted complete trial without an independent re-integration",
    )
    parser.add_argument("--adaptive-stepping", action="store_true")
    parser.add_argument("--adaptive-initial-step", type=float, default=0.25)
    parser.add_argument("--adaptive-min-step", type=float, default=1.0 / 256.0)
    parser.add_argument("--adaptive-max-step", type=float, default=0.5)
    parser.add_argument(
        "--adaptive-error-control",
        choices=("none", "step-doubling"),
        default="none",
    )
    parser.add_argument("--adaptive-error-stress-rtol", type=float, default=1.0e-3)
    parser.add_argument("--adaptive-error-slip-rtol", type=float, default=1.0e-3)
    parser.add_argument("--adaptive-error-reaction-rtol", type=float, default=1.0e-3)
    parser.add_argument("--adaptive-error-displacement-rtol", type=float, default=1.0e-5)
    parser.add_argument("--adaptive-error-linf-factor", type=float, default=5.0)
    parser.add_argument("--adaptive-error-safety-factor", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    step_doubling_enabled = arguments.adaptive_error_control == "step-doubling"
    adaptive_enabled = arguments.adaptive_stepping or step_doubling_enabled
    if adaptive_enabled and arguments.behaviour != "fcc_forest_rubin_srix":
        raise SystemExit(
            "adaptive stepping is currently restricted to the rate-independent SRIX law"
        )
    git_worktree = _git_worktree_state()
    crop = tuple(arguments.crop_nodes)
    mesh = crop[1] - crop[0]
    if mesh != crop[3] - crop[2]:
        raise SystemExit("P43 crop must be square")
    grid, _, yield_stress, coefficient, boundary = _load_case(mesh, crop)
    law: CrystalLaw = (
        "forest_rubin_srix"
        if arguments.behaviour == "fcc_forest_rubin_srix"
        else "meric_cailletaud"
    )
    _paired_overrides, crystal_manifest = resolve_paired_crystal_parameters(
        paired_parameter_set=arguments.paired_parameter_set,
        law=law,
    )
    repository_root = Path(__file__).resolve().parents[1]
    meric_source_fingerprint = read_crystal_structure_fingerprint(
        repository_root / "mfront/Fcc316LMericCailletaud.mfront"
    )
    srix_source_fingerprint = read_crystal_structure_fingerprint(
        repository_root / "mfront/Fcc316LForestRubinSrix.mfront"
    )
    if (
        meric_source_fingerprint.crystal_structure != srix_source_fingerprint.crystal_structure
        or meric_source_fingerprint.sliding_system != srix_source_fingerprint.sliding_system
        or meric_source_fingerprint.interaction_matrix
        != srix_source_fingerprint.interaction_matrix
    ):
        raise SystemExit("the two MFront crystal sources do not share the same structure contract")
    source_path = repository_root / "mfront" / (
        "Fcc316LForestRubinSrix.mfront"
        if law == "forest_rubin_srix"
        else "Fcc316LMericCailletaud.mfront"
    )
    source_fingerprint = read_crystal_structure_fingerprint(source_path)
    backbone_record = cast(dict[str, object], crystal_manifest["backbone"])
    if source_fingerprint.interaction_matrix != tuple(
        cast(list[float], backbone_record["interaction_matrix"])
    ):
        raise SystemExit("compiled MFront interaction structure does not match the paired backbone")
    history = np.stack(
        [fraction * boundary for fraction in np.linspace(0.0, 1.0, arguments.increments + 1)]
    )
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
            },
            "paired_parameter_set": arguments.paired_parameter_set,
        },
    )
    started = time.perf_counter()
    solve_config = EBISpectralSolverConfig(
            relative_equilibrium_tolerance=arguments.tolerance,
            linear_tolerance_mode=arguments.linear_mode,
            reference_update_mode=arguments.reference_update,
            verify_final_state=not arguments.no_final_verification,
            adaptive_stepping_enabled=adaptive_enabled,
            adaptive_step=AdaptiveStepConfig(
                initial_increment_fraction=arguments.adaptive_initial_step,
                minimum_increment_fraction=arguments.adaptive_min_step,
                maximum_increment_fraction=arguments.adaptive_max_step,
            ),
            step_doubling=StepDoublingErrorConfig(
                enabled=step_doubling_enabled,
                stress_relative_tolerance=arguments.adaptive_error_stress_rtol,
                reaction_relative_tolerance=arguments.adaptive_error_reaction_rtol,
                signed_slip_relative_tolerance=arguments.adaptive_error_slip_rtol,
                accumulated_slip_relative_tolerance=arguments.adaptive_error_slip_rtol,
                displacement_relative_tolerance=arguments.adaptive_error_displacement_rtol,
                linf_relative_tolerance_factor=arguments.adaptive_error_linf_factor,
                safety_factor=arguments.adaptive_error_safety_factor,
            ),
            krylov_method=arguments.krylov_method,
            krylov_recycling=arguments.krylov_recycling,
            transform=SpectralTransformConfig(
                backend="fftw",
                workers=1,
                fftw_planner_effort="measure",
                fftw_planning_time_limit_s=2.0,
                fftw_use_wisdom=False,
            ),
        )
    try:
        result = solve_two_state_dirichlet_plane_stress(
            grid=grid,
            material=material,
            boundary_displacement_history=history[: arguments.increments + 1],
            config=solve_config,
        )
    except StepDoublingFailureError as error:
        elapsed = time.perf_counter() - started
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        partial_report = {
            "status": "failed_step_doubling",
            "failure_reason": str(error),
            "crop_nodes": list(crop),
            "mesh": [mesh, mesh],
            "increments": arguments.increments,
            "behaviour": arguments.behaviour,
            "mfront_threads": arguments.mfront_threads,
            "adaptive_stepping_enabled": adaptive_enabled,
            "adaptive_error_control": arguments.adaptive_error_control,
            "adaptive_error_configuration": {
                "stress_relative_tolerance": arguments.adaptive_error_stress_rtol,
                "reaction_relative_tolerance": arguments.adaptive_error_reaction_rtol,
                "signed_slip_relative_tolerance": arguments.adaptive_error_slip_rtol,
                "accumulated_slip_relative_tolerance": arguments.adaptive_error_slip_rtol,
                "displacement_relative_tolerance": arguments.adaptive_error_displacement_rtol,
                "linf_relative_tolerance_factor": arguments.adaptive_error_linf_factor,
                "activity_threshold": 1.0e-8,
                "signed_slip_absolute_tolerance": 1.0e-8,
                "accumulated_slip_absolute_tolerance": 1.0e-8,
                "safety_factor": arguments.adaptive_error_safety_factor,
            },
            "adaptive_step_history": list(error.history),
            "elapsed_seconds": elapsed,
            "execution_commit": _git_head(),
            "paired_parameter_set": arguments.paired_parameter_set,
            "boundary_sha256": _hash(boundary),
            "provenance": _git_worktree_state(),
        }
        arguments.output.write_text(json.dumps(partial_report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(partial_report, indent=2, sort_keys=True))
        return 2
    elapsed = time.perf_counter() - started
    fields = {
        "displacement": result.displacement,
        "stress_in_plane_mpa": result.stress_in_plane_mpa,
        "reaction_forces": result.reaction_forces,
        "accumulated_slip": result.observables["accumulated_slip"],
    }
    for name in ("plastic_slip", "equivalent_plastic_slip"):
        values = result.observables.get(name)
        if values is not None:
            fields[name] = values
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    field_path = arguments.output.with_suffix(".fields.npz")
    np.savez_compressed(field_path, **fields)  # type: ignore[arg-type]
    diagnostics = result.diagnostics
    accepted_increment_count = (
        sum(
            bool(item.get("accepted", False))
            for item in diagnostics.adaptive_step_history
        )
        if step_doubling_enabled
        else len(diagnostics.iterations_per_increment)
    )
    report = {
        "status": "completed_crystal_tet2_p43",
        "crop_nodes": list(crop),
        "mesh": [mesh, mesh],
        "increments": arguments.increments,
        "requested_increments": arguments.increments,
        "accepted_increments": accepted_increment_count,
        "tolerance": arguments.tolerance,
        "behaviour": arguments.behaviour,
        "mfront_threads": arguments.mfront_threads,
        "krylov_method": arguments.krylov_method,
        "linear_mode": arguments.linear_mode,
        "reference_update": arguments.reference_update,
        "verify_final_state": not arguments.no_final_verification,
        "adaptive_stepping_enabled": adaptive_enabled,
        "adaptive_error_control": arguments.adaptive_error_control,
        "adaptive_error_configuration": {
            "stress_relative_tolerance": arguments.adaptive_error_stress_rtol,
            "reaction_relative_tolerance": arguments.adaptive_error_reaction_rtol,
            "signed_slip_relative_tolerance": arguments.adaptive_error_slip_rtol,
            "accumulated_slip_relative_tolerance": arguments.adaptive_error_slip_rtol,
            "displacement_relative_tolerance": arguments.adaptive_error_displacement_rtol,
            "linf_relative_tolerance_factor": arguments.adaptive_error_linf_factor,
            "activity_threshold": 1.0e-8,
            "signed_slip_absolute_tolerance": 1.0e-8,
            "accumulated_slip_absolute_tolerance": 1.0e-8,
            "safety_factor": arguments.adaptive_error_safety_factor,
        },
        "adaptive_step_history": list(diagnostics.adaptive_step_history),
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
        "git_worktree": git_worktree,
        "field_file": str(field_path),
        "field_sha256": {name: _hash(values) for name, values in fields.items()},
        "slip_observables": {
            name: {
                "shape": list(values.shape),
                "meaning": (
                    "signed plastic slip per FCC system"
                    if name == "plastic_slip"
                    else "accumulated equivalent slip per FCC system"
                ),
                "system_axis": -1,
                "triangle_axis": 2,
            }
            for name, values in fields.items()
            if name in {"plastic_slip", "equivalent_plastic_slip"}
        },
        "boundary_sha256": _hash(boundary),
        "units": "mm, MPa",
        "orientation": {
            "mode": "homogeneous",
            "euler_bunge_deg": [35.0, 20.0, 15.0],
            "sha256": _hash(np.asarray([35.0, 20.0, 15.0], dtype=np.float64)),
        },
        "crystal_material": {
            **crystal_manifest,
            "mfront_structure": {
                "crystal_structure": source_fingerprint.crystal_structure,
                "sliding_system": source_fingerprint.sliding_system,
                "interaction_matrix": list(source_fingerprint.interaction_matrix),
                "source_sha256": source_fingerprint.source_sha256,
                "structure_contract_sha256": source_fingerprint.structure_contract_sha256(),
                "meric_source_sha256": meric_source_fingerprint.source_sha256,
                "srix_source_sha256": srix_source_fingerprint.source_sha256,
            },
        },
    }
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
