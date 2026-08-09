"""Run the condensed 3D crystal TRI2 solver on the registered P43 crop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
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
from fem_inhouse.spectral2d.diagnostics import summarize_load_step_attempts
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


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_ebsd_orientation_crop(
    path: Path,
    crop: tuple[int, int, int, int],
) -> tuple[np.ndarray, dict[str, object]]:
    """Load co-registered Bunge angles from the CP HDF5 dataset."""

    import h5py

    x0, x1, y0, y1 = crop
    with h5py.File(path, "r") as handle:
        datasets = {
            name: np.asarray(handle[f"orientation/{name}"][x0:x1, y0:y1], dtype=np.float64)
            for name in ("phi1", "Phi", "phi2")
        }
        source_shape = tuple(handle["orientation/phi1"].shape)
        descriptions = {
            name: str(handle[f"orientation/{name}"].attrs.get("description", ""))
            for name in datasets
        }
    angles = np.stack((datasets["phi1"], datasets["Phi"], datasets["phi2"]), axis=-1)
    if angles.shape != (x1 - x0, y1 - y0, 3):
        raise ValueError(f"unexpected EBSD crop shape {angles.shape}")
    return angles, {
        "mode": "ebsd",
        "source_file": str(path),
        "source_sha256": _hash_file(path),
        "dataset_paths": ["orientation/phi1", "orientation/Phi", "orientation/phi2"],
        "source_shape": list(source_shape),
        "crop_nodes": list(crop),
        "angles_shape": list(angles.shape),
        "angles_sha256": _hash(angles),
        "descriptions": descriptions,
        "co_registration": (
            "CP_dataset orientation fields are co-registered with DIC displacement fields"
        ),
    }


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
    parser.add_argument(
        "--material-backend",
        choices=(
            "mfront-3d-condensed-plane-stress",
            "mfront-3d-condensed-plane-stress-halved",
            "mfront-native-generalised-plane-stress",
            "mfront-structural-plane-stress",
        ),
        default="mfront-3d-condensed-plane-stress",
        help="reference Python condensation, the reference forced through the "
        "uniform two-half sub-stepping path, or the experimental native MGIS "
        "closure",
    )
    parser.add_argument("--paired-parameter-set", required=True)
    parser.add_argument(
        "--ebsd-orientation-h5",
        type=Path,
        help="co-registered CP_dataset.h5 containing orientation/phi1, Phi, phi2",
    )
    parser.add_argument("--mfront-threads", type=int, default=4)
    parser.add_argument("--maximum-newton-iterations", type=int, default=40)
    parser.add_argument(
        "--local-transverse-predictor",
        choices=("committed", "tangent"),
        default="committed",
    )
    parser.add_argument(
        "--condensation-block-size",
        type=int,
        help="partition the MFront plane-stress condensation into independent blocks",
    )
    parser.add_argument(
        "--local-closure-tolerance",
        type=float,
        help="local plane-stress closure tolerance in MPa (default 1e-8 for the "
        "condensed reference); tightening it aligns the reference's local "
        "solution with the GPS's ~1e-14",
    )
    parser.add_argument(
        "--gps-shadow-tangent",
        action="store_true",
        help="for the native GPS backend, replace the Newton tangent by the "
        "reference Schur evaluated at the GPS's own converged state (one extra "
        "3D integration per evaluation); stress, state, closure and "
        "sub-stepping are untouched",
    )
    parser.add_argument(
        "--gps-composite-fd-tangent",
        action="store_true",
        help="for the native GPS backend, rebuild the tangent of the "
        "sub-stepped points by finite differences on the composite trajectory",
    )
    parser.add_argument(
        "--srix-smoothing-epsilon",
        type=float,
        default=None,
        help="experimental SRIX Charbonnier stress scale in MPa; zero preserves the historical law",
    )
    parser.add_argument(
        "--srix-smoothing-exponent",
        type=float,
        default=None,
        help="experimental SRIX generalized Charbonnier exponent",
    )
    parser.add_argument(
        "--gps-condensed-tangent",
        action="store_true",
        help="for the native GPS backend, enable the law's CondensedTangent "
        "parameter: the @TangentOperator returns the exact plane-stress Schur "
        "of the raw law computed inside the local Newton (no shadow, no second "
        "integration)",
    )
    parser.add_argument(
        "--progress-output",
        type=Path,
        help="JSONL file written after each increment/Newton event",
    )
    parser.add_argument(
        "--krylov-method",
        choices=("gmres", "lgmres", "gcrotmk"),
        default="lgmres",
    )
    parser.add_argument(
        "--krylov-recycling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gmres-rtol", type=float, default=1.0e-8)
    parser.add_argument("--gmres-restart", type=int, default=50)
    parser.add_argument("--lgmres-inner-m", type=int, default=30)
    parser.add_argument("--lgmres-outer-k", type=int, default=3)
    parser.add_argument("--gcrotmk-m", type=int, default=20)
    parser.add_argument("--gcrotmk-k", type=int, default=10)
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
    parser.add_argument(
        "--adaptive-step-control",
        choices=("none", "newton", "predictive"),
        default="none",
        help="fast adaptive controller: Newton budget or predictive slip indicator",
    )
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
    parser.add_argument("--adaptive-error-slip-atol", type=float, default=1.0e-6)
    parser.add_argument("--adaptive-error-slip-linf-cap", type=float, default=1.0e-6)
    parser.add_argument("--adaptive-error-activity-threshold", type=float, default=1.0e-6)
    parser.add_argument("--adaptive-error-reaction-rtol", type=float, default=1.0e-3)
    parser.add_argument("--adaptive-error-displacement-rtol", type=float, default=1.0e-5)
    parser.add_argument("--adaptive-error-linf-factor", type=float, default=5.0)
    parser.add_argument("--adaptive-error-reaction-linf-factor", type=float, default=10.0)
    parser.add_argument("--adaptive-error-reaction-linf-cap", type=float, default=5.0e-5)
    parser.add_argument("--adaptive-error-safety-factor", type=float, default=0.8)
    parser.add_argument("--adaptive-slip-rtol", type=float, default=5.0e-3)
    parser.add_argument("--adaptive-slip-atol", type=float, default=1.0e-6)
    parser.add_argument("--adaptive-slip-growth-threshold", type=float, default=0.25)
    parser.add_argument("--adaptive-slip-cutback-threshold", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    step_doubling_enabled = arguments.adaptive_error_control == "step-doubling"
    adaptive_enabled = (
        arguments.adaptive_stepping
        or arguments.adaptive_step_control != "none"
        or step_doubling_enabled
    )
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
    if arguments.ebsd_orientation_h5 is None:
        orientation_configuration: dict[str, object] = {
            "mode": "homogeneous",
            "euler_bunge_deg": [35.0, 20.0, 15.0],
        }
        orientation_manifest: dict[str, object] = {
            "mode": "homogeneous",
            "euler_bunge_deg": [35.0, 20.0, 15.0],
            "sha256": _hash(np.asarray([35.0, 20.0, 15.0], dtype=np.float64)),
        }
    else:
        orientation_angles, orientation_manifest = _load_ebsd_orientation_crop(
            arguments.ebsd_orientation_h5,
            crop,
        )
        orientation_configuration = {
            "mode": "ebsd",
            "euler_bunge_deg": orientation_angles,
        }
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
    backend = (
        "mfront-3d-condensed-plane-stress"
        if arguments.material_backend == "mfront-3d-condensed-plane-stress-halved"
        else arguments.material_backend
    )
    material = create_plane_stress_material_batch(
        backend,
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
        local_plane_stress_options={
            "local_condition_check_mode": "on_failure",
            "local_transverse_predictor": arguments.local_transverse_predictor,
            **(
                {"local_tolerance_mpa": arguments.local_closure_tolerance}
                if arguments.local_closure_tolerance is not None
                else {}
            ),
            **(
                {"condensation_block_size": arguments.condensation_block_size}
                if arguments.condensation_block_size is not None
                else {}
            ),
        },
        constitutive_options={
            "crystal_orientation": {
                **orientation_configuration,
            },
            "paired_parameter_set": arguments.paired_parameter_set,
            **(
                {"srix_smoothing_epsilon": arguments.srix_smoothing_epsilon}
                if arguments.srix_smoothing_epsilon is not None
                else {}
            ),
            **(
                {"srix_smoothing_exponent": arguments.srix_smoothing_exponent}
                if arguments.srix_smoothing_exponent is not None
                else {}
            ),
            **(
                {"gps_shadow_tangent": True}
                if arguments.gps_shadow_tangent
                else {}
            ),
            **(
                {"gps_composite_fd_tangent": True}
                if arguments.gps_composite_fd_tangent
                else {}
            ),
        },
    )
    if (
        arguments.gps_condensed_tangent
        and arguments.material_backend == "mfront-native-generalised-plane-stress"
    ):
        material._parameters["CondensedTangent"] = 1.0
        material._condensed_tangent = True
    started = time.perf_counter()
    progress_path = arguments.progress_output or arguments.output.with_suffix(".progress.jsonl")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("", encoding="utf-8")

    def progress_callback(event: dict[str, object]) -> None:
        record = {"elapsed_seconds": time.perf_counter() - started, **event}
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()

    solve_config = EBISpectralSolverConfig(
            relative_equilibrium_tolerance=arguments.tolerance,
            maximum_newton_iterations=arguments.maximum_newton_iterations,
            progress_callback=progress_callback,
            linear_tolerance_mode=arguments.linear_mode,
            reference_update_mode=arguments.reference_update,
            verify_final_state=not arguments.no_final_verification,
            adaptive_stepping_enabled=adaptive_enabled,
            adaptive_step=AdaptiveStepConfig(
                initial_increment_fraction=arguments.adaptive_initial_step,
                minimum_increment_fraction=arguments.adaptive_min_step,
                maximum_increment_fraction=arguments.adaptive_max_step,
                slip_error_control=(
                    "predictive"
                    if arguments.adaptive_step_control == "predictive"
                    else "disabled"
                ),
                slip_error_relative_tolerance=arguments.adaptive_slip_rtol,
                slip_error_absolute_tolerance=arguments.adaptive_slip_atol,
                slip_error_growth_threshold=arguments.adaptive_slip_growth_threshold,
                slip_error_cutback_threshold=arguments.adaptive_slip_cutback_threshold,
            ),
            step_doubling=StepDoublingErrorConfig(
                enabled=step_doubling_enabled,
                stress_relative_tolerance=arguments.adaptive_error_stress_rtol,
                reaction_relative_tolerance=arguments.adaptive_error_reaction_rtol,
                signed_slip_relative_tolerance=arguments.adaptive_error_slip_rtol,
                signed_slip_absolute_tolerance=arguments.adaptive_error_slip_atol,
                signed_slip_linf_absolute_cap=arguments.adaptive_error_slip_linf_cap,
                accumulated_slip_relative_tolerance=arguments.adaptive_error_slip_rtol,
                accumulated_slip_absolute_tolerance=arguments.adaptive_error_slip_atol,
                accumulated_slip_linf_absolute_cap=arguments.adaptive_error_slip_linf_cap,
                activity_threshold=arguments.adaptive_error_activity_threshold,
                displacement_relative_tolerance=arguments.adaptive_error_displacement_rtol,
                linf_relative_tolerance_factor=arguments.adaptive_error_linf_factor,
                reaction_linf_relative_tolerance_factor=(
                    arguments.adaptive_error_reaction_linf_factor
                ),
                reaction_linf_absolute_cap=arguments.adaptive_error_reaction_linf_cap,
                safety_factor=arguments.adaptive_error_safety_factor,
            ),
            krylov_method=arguments.krylov_method,
            krylov_recycling=arguments.krylov_recycling,
            gmres_relative_tolerance=arguments.gmres_rtol,
            gmres_restart=arguments.gmres_restart,
            lgmres_inner_m=arguments.lgmres_inner_m,
            lgmres_outer_k=arguments.lgmres_outer_k,
            gcrotmk_m=arguments.gcrotmk_m,
            gcrotmk_k=arguments.gcrotmk_k,
            transform=SpectralTransformConfig(
                backend="fftw",
                workers=1,
                fftw_planner_effort="measure",
                fftw_planning_time_limit_s=2.0,
                fftw_use_wisdom=False,
            ),
        )
    if arguments.material_backend == "mfront-3d-condensed-plane-stress-halved":
        # Decisive sub-stepping-path test: the reference forced through the
        # same uniform two-half path the GPS sub-stepping imposes on its
        # failing points, law and tangent untouched.
        from scripts.benchmark_substepping_path import UniformlyHalvedReference

        material = UniformlyHalvedReference(material)
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
            "maximum_newton_iterations": arguments.maximum_newton_iterations,
            "local_transverse_predictor": arguments.local_transverse_predictor,
            "progress_file": str(progress_path),
            "adaptive_stepping_enabled": adaptive_enabled,
            "adaptive_step_control": arguments.adaptive_step_control,
            "adaptive_error_control": arguments.adaptive_error_control,
            "adaptive_step_configuration": {
                "slip_error_control": (
                    "predictive"
                    if arguments.adaptive_step_control == "predictive"
                    else "disabled"
                ),
                "slip_error_relative_tolerance": arguments.adaptive_slip_rtol,
                "slip_error_absolute_tolerance": arguments.adaptive_slip_atol,
                "slip_error_growth_threshold": arguments.adaptive_slip_growth_threshold,
                "slip_error_cutback_threshold": arguments.adaptive_slip_cutback_threshold,
            },
            "adaptive_error_configuration": {
                "stress_relative_tolerance": arguments.adaptive_error_stress_rtol,
                "reaction_relative_tolerance": arguments.adaptive_error_reaction_rtol,
                "signed_slip_relative_tolerance": arguments.adaptive_error_slip_rtol,
                "accumulated_slip_relative_tolerance": arguments.adaptive_error_slip_rtol,
                "displacement_relative_tolerance": arguments.adaptive_error_displacement_rtol,
                "linf_relative_tolerance_factor": arguments.adaptive_error_linf_factor,
                "reaction_linf_relative_tolerance_factor": (
                    arguments.adaptive_error_reaction_linf_factor
                ),
                "reaction_linf_absolute_cap": arguments.adaptive_error_reaction_linf_cap,
                "activity_threshold": arguments.adaptive_error_activity_threshold,
                "signed_slip_absolute_tolerance": arguments.adaptive_error_slip_atol,
                "signed_slip_linf_absolute_cap": arguments.adaptive_error_slip_linf_cap,
                "accumulated_slip_absolute_tolerance": arguments.adaptive_error_slip_atol,
                "accumulated_slip_linf_absolute_cap": arguments.adaptive_error_slip_linf_cap,
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
    except Exception as error:
        progress_callback(
            {
                "event": "solver_exception",
                "exception_type": type(error).__name__,
                "message": str(error),
            }
        )
        raise
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
        "srix_smoothing_epsilon_mpa": arguments.srix_smoothing_epsilon,
        "srix_smoothing_exponent": arguments.srix_smoothing_exponent,
        "mfront_threads": arguments.mfront_threads,
        "maximum_newton_iterations": arguments.maximum_newton_iterations,
        "local_transverse_predictor": arguments.local_transverse_predictor,
        "condensation_block_size": arguments.condensation_block_size,
        "krylov_method": arguments.krylov_method,
        "krylov_recycling": arguments.krylov_recycling,
        "gmres_relative_tolerance": arguments.gmres_rtol,
        "gmres_restart": arguments.gmres_restart,
        "lgmres_inner_m": arguments.lgmres_inner_m,
        "lgmres_outer_k": arguments.lgmres_outer_k,
        "gcrotmk_m": arguments.gcrotmk_m,
        "gcrotmk_k": arguments.gcrotmk_k,
        "linear_mode": arguments.linear_mode,
        "reference_update": arguments.reference_update,
        "verify_final_state": not arguments.no_final_verification,
        "adaptive_stepping_enabled": adaptive_enabled,
        "adaptive_step_control": arguments.adaptive_step_control,
        "adaptive_error_control": arguments.adaptive_error_control,
        "adaptive_step_configuration": {
            "slip_error_control": (
                "predictive"
                if arguments.adaptive_step_control == "predictive"
                else "disabled"
            ),
            "slip_error_relative_tolerance": arguments.adaptive_slip_rtol,
            "slip_error_absolute_tolerance": arguments.adaptive_slip_atol,
            "slip_error_growth_threshold": arguments.adaptive_slip_growth_threshold,
            "slip_error_cutback_threshold": arguments.adaptive_slip_cutback_threshold,
        },
        "adaptive_error_configuration": {
            "stress_relative_tolerance": arguments.adaptive_error_stress_rtol,
            "reaction_relative_tolerance": arguments.adaptive_error_reaction_rtol,
            "signed_slip_relative_tolerance": arguments.adaptive_error_slip_rtol,
            "signed_slip_absolute_tolerance": arguments.adaptive_error_slip_atol,
            "signed_slip_linf_absolute_cap": arguments.adaptive_error_slip_linf_cap,
            "accumulated_slip_relative_tolerance": arguments.adaptive_error_slip_rtol,
            "accumulated_slip_absolute_tolerance": arguments.adaptive_error_slip_atol,
            "accumulated_slip_linf_absolute_cap": arguments.adaptive_error_slip_linf_cap,
            "activity_threshold": arguments.adaptive_error_activity_threshold,
            "displacement_relative_tolerance": arguments.adaptive_error_displacement_rtol,
            "linf_relative_tolerance_factor": arguments.adaptive_error_linf_factor,
            "reaction_linf_relative_tolerance_factor": (
                arguments.adaptive_error_reaction_linf_factor
            ),
            "reaction_linf_absolute_cap": arguments.adaptive_error_reaction_linf_cap,
            "safety_factor": arguments.adaptive_error_safety_factor,
        },
        "adaptive_step_history": list(diagnostics.adaptive_step_history),
        "linear_solves": [asdict(item) for item in diagnostics.linear_solves],
        "load_step_attempts": [
            asdict(item) for item in diagnostics.load_step_attempts
        ],
        "attempt_cost_summary": summarize_load_step_attempts(
            diagnostics.load_step_attempts
        ),
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
        "material_local_iteration_histogram": list(
            diagnostics.material_local_iteration_histogram
        ),
        "provenance": diagnostics.provenance,
        "execution_commit": diagnostics.provenance.get("commit_sha"),
        "archive_commit": os.environ.get("ARCHIVE_COMMIT", _git_head()),
        "progress_file": str(progress_path),
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
        "orientation": orientation_manifest,
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
