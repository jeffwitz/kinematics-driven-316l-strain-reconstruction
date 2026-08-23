#!/usr/bin/env python3
"""Qualify SRIX-REGM on an exact small digital twin before any P43 run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch
from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET, get_parameter_set
from fem_inhouse.identification.crystal_plane_stress_elasticity import (
    cubic_stiffness_from_engineering_constants,
    rotated_plane_stress_stiffness,
)
from fem_inhouse.identification.srix_equilibrium_gap import (
    SrixEquilibriumGapProblem,
    SrixTheta4,
)
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_ebi import EBISpectralSolverConfig
from fem_inhouse.spectral2d.newton_two_state import (
    TwoStateIncrementFields,
    solve_two_state_dirichlet_plane_stress,
)
from fem_inhouse.spectral2d.step_control import AdaptiveStepConfig

ROOT = Path(__file__).resolve().parents[1]
PIXEL_SIZE_MM = 0.00184
ORIENTATIONS_DEG = np.asarray(
    (
        (0.0, 0.0, 0.0),
        (35.0, 20.0, 15.0),
        (45.0, 54.7356, 0.0),
        (10.0, 65.0, 25.0),
    ),
    dtype=np.float64,
)
PATH_COMPONENTS = np.asarray(
    (
        (0.0, 0.0, 0.0),
        (0.0005, -0.0001, 0.0),
        (0.0010, -0.0002, 0.0),
        (0.0015, -0.0003, 0.0010),
        (0.0020, -0.0004, 0.0020),
        (0.0028, -0.0006, 0.0030),
        (0.0035, -0.0008, 0.0020),
        (0.0042, -0.0010, 0.0040),
        (0.0050, -0.0012, 0.0050),
    ),
    dtype=np.float64,
)
FD_STEPS = (1.0e-2, 3.0e-3, 1.0e-3)
SUBSTEPS_PER_SEGMENT = 4
SINGULAR_DIRECTION_LOG_STEP = 5.0e-2


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class _Identity:
    def apply(self, values: Any) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values: Any) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _orientation_map(pixels: int) -> np.ndarray:
    rows, columns = np.indices((pixels, pixels))
    indices = 2 * (rows >= pixels // 2) + (columns >= pixels // 2)
    return ORIENTATIONS_DEG[indices]


def _expanded_path() -> np.ndarray:
    values = [PATH_COMPONENTS[0]]
    for start, end in pairwise(PATH_COMPONENTS):
        values.extend(
            start + fraction * (end - start)
            for fraction in np.linspace(
                1.0 / SUBSTEPS_PER_SEGMENT, 1.0, SUBSTEPS_PER_SEGMENT
            )
        )
    return np.asarray(values, dtype=np.float64)


def _boundary_history(grid: StructuredGrid2D) -> np.ndarray:
    x = np.linspace(0.0, grid.length_x, grid.nx + 1)
    y = np.linspace(0.0, grid.length_y, grid.ny + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    fields = []
    for exx, eyy, gamma_xy in _expanded_path():
        fields.append(np.stack((exx * xx + gamma_xy * yy, eyy * yy), axis=-1))
    return np.asarray(fields, dtype=np.float64)


def _theta_from_preset() -> SrixTheta4:
    preset = get_parameter_set(DEFAULT_PARAMETER_SET)
    return SrixTheta4(
        tau0_mpa=preset.tau0_mpa,
        r_mpa=preset.overstress_modulus_mpa,
        q_mpa=preset.q_mpa,
        b=preset.b,
    )


def _material_factory(
    *, pixels: int, orientations: np.ndarray, library: str, threads: int
):
    point_count = 2 * pixels * pixels

    def create(overrides: dict[str, float]):
        return create_plane_stress_material_batch(
            "mfront-3d-condensed-plane-stress",
            np.ones(point_count),
            np.ones(point_count),
            0.245,
            young_modulus_mpa=205_000.0,
            poisson_ratio=0.30,
            hardening_mode="ludwik",
            plastic_strain_max=0.2,
            plastic_table_points=1_000,
            first_positive_plastic_strain=1.0e-6,
            mfront_library=library,
            mfront_threads=threads,
            mfront_behaviour_id="fcc_forest_rubin_srix",
            local_plane_stress_options={
                "local_condition_check_mode": "on_failure",
                "local_transverse_predictor": "tangent",
            },
            constitutive_options={
                "parameter_set": DEFAULT_PARAMETER_SET,
                "parameters": overrides,
                "crystal_orientation": {
                    "mode": "ebsd",
                    "euler_bunge_deg": orientations,
                },
            },
        )

    return create


def _point_elasticity(orientations: np.ndarray) -> np.ndarray:
    preset = get_parameter_set(DEFAULT_PARAMETER_SET)
    cubic = cubic_stiffness_from_engineering_constants(
        preset.elasticity.young_modulus_100_mpa,
        preset.elasticity.poisson_ratio_100,
        preset.elasticity.shear_modulus_mpa,
    )
    per_pixel = rotated_plane_stress_stiffness(cubic, orientations.reshape(-1, 3))
    return np.repeat(per_pixel, 2, axis=0)


def _operator(grid: StructuredGrid2D, orientations: np.ndarray):
    preset = get_parameter_set(DEFAULT_PARAMETER_SET)
    return TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=preset.elasticity.young_modulus_100_mpa,
        poisson_ratio=preset.elasticity.poisson_ratio_100,
        point_elasticity=_point_elasticity(orientations),
        transfer=_Identity(),
        whitener=_Identity(),
    )


def _generate_twin(
    *, pixels: int, library: str, threads: int, theta: SrixTheta4
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], dict[str, Any], float]:
    grid = StructuredGrid2D(
        pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels
    )
    orientations = _orientation_map(pixels)
    material = _material_factory(
        pixels=pixels,
        orientations=orientations,
        library=library,
        threads=threads,
    )(theta.as_runtime_overrides())
    boundary = _boundary_history(grid)
    displacements = [boundary[0].copy()]

    def observe(fields: TwoStateIncrementFields) -> None:
        displacements.append(np.asarray(fields.displacement, dtype=np.float64).copy())

    started = time.perf_counter()
    result = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=material,
        boundary_displacement_history=boundary,
        config=EBISpectralSolverConfig(
            relative_equilibrium_tolerance=1.0e-6,
            maximum_newton_iterations=25,
            verify_final_state=True,
            adaptive_stepping_enabled=True,
            adaptive_step=AdaptiveStepConfig(
                initial_increment_fraction=1.0 / (8 * SUBSTEPS_PER_SEGMENT),
                minimum_increment_fraction=1.0 / 4096.0,
                maximum_increment_fraction=1.0 / (8 * SUBSTEPS_PER_SEGMENT),
                increment_growth_factor=1.5,
                increment_cutback_factor=0.5,
                target_newton_iterations_min=4,
                target_newton_iterations_max=7,
                maximum_cutbacks_per_step=8,
            ),
        ),
        increment_observer=observe,
    )
    elapsed = time.perf_counter() - started
    accepted = tuple(
        attempt for attempt in result.diagnostics.load_step_attempts if attempt.accepted
    )
    if len(displacements) != len(accepted) + 1:
        raise RuntimeError("accepted solver steps and archived twin states disagree")
    time_increments = np.asarray(
        [attempt.load_fraction_end - attempt.load_fraction_start for attempt in accepted],
        dtype=np.float64,
    )
    scored_states = tuple(
        index
        for index, attempt in enumerate(accepted, start=1)
        if np.isclose(8.0 * attempt.load_fraction_end, round(8.0 * attempt.load_fraction_end))
    )
    if len(scored_states) != 8 or scored_states[-1] != len(accepted):
        raise RuntimeError("adaptive twin did not preserve all eight macro endpoints")
    return (
        np.asarray(displacements),
        time_increments,
        scored_states,
        asdict(result.diagnostics),
        elapsed,
    )


def _problem(
    *,
    pixels: int,
    displacement_history: np.ndarray,
    time_increments: np.ndarray,
    scored_states: tuple[int, ...],
    library: str,
    threads: int,
    debug: bool,
) -> SrixEquilibriumGapProblem:
    grid = StructuredGrid2D(
        pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels
    )
    orientations = _orientation_map(pixels)
    replayed = tuple(range(1, displacement_history.shape[0]))
    return SrixEquilibriumGapProblem(
        operator=_operator(grid, orientations),
        displacement_history=displacement_history,
        state_indices=replayed,
        scored_states=set(scored_states),
        material_factory=_material_factory(
            pixels=pixels,
            orientations=orientations,
            library=library,
            threads=threads,
        ),
        time_increments=time_increments,
        debug=debug,
    )


def _evaluation_record(evaluation: Any) -> dict[str, Any]:
    return {
        "theta": evaluation.theta.as_runtime_overrides(),
        "cost": evaluation.cost,
        "residual_rms": evaluation.residual_rms,
        "material_evaluations": evaluation.material_evaluations,
        "timing": asdict(evaluation.timing),
        "backend_timing": dict(evaluation.backend_timing),
        "states": [
            {
                "state_index": state.state_index,
                "scored": state.scored,
                "raw_equilibrium_norm": state.raw_equilibrium_norm,
                "pseudo_displacement_norm": state.pseudo_displacement_norm,
                "relative_pseudo_displacement_norm": state.relative_pseudo_displacement_norm,
                "whitened_residual_rms": state.whitened_residual_rms,
            }
            for state in evaluation.states
        ],
    }


def _fd_study(problem: SrixEquilibriumGapProblem, eta: np.ndarray):
    jacobians = {}
    timings = {}
    for step in FD_STEPS:
        started = time.perf_counter()
        jacobians[step] = problem.jacobian_fd(eta, relative_step=step)
        timings[step] = time.perf_counter() - started
    finest = jacobians[FD_STEPS[-1]]
    rows = []
    for step in FD_STEPS:
        current = jacobians[step]
        rows.append(
            {
                "step": step,
                "seconds": timings[step],
                "column_norms": np.linalg.norm(current, axis=0).tolist(),
                "relative_to_finest": (
                    np.linalg.norm(current - finest)
                    / max(np.linalg.norm(finest), np.finfo(float).tiny)
                ),
            }
        )
    return jacobians, rows


def _save_figures(
    output: Path,
    true_theta: SrixTheta4,
    initial_theta: SrixTheta4,
    identified_theta: SrixTheta4,
    singular: np.ndarray,
    cost_history: list[float],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].semilogy(np.arange(1, len(cost_history) + 1), cost_history, marker="o")
    axes[0].set_xlabel("residual evaluation")
    axes[0].set_ylabel("REGM cost")
    axes[0].set_title("Deterministic least squares")
    axes[1].semilogy(np.arange(1, len(singular) + 1), singular / singular[0], marker="o")
    axes[1].set_xlabel("singular direction")
    axes[1].set_ylabel("normalized singular value")
    axes[1].set_title("Sensitivity SVD at truth")
    names = ("tau0", "R", "Q", "b")
    x = np.arange(4)
    axes[2].bar(x - 0.25, true_theta.as_array(), width=0.25, label="truth")
    axes[2].bar(x, initial_theta.as_array(), width=0.25, label="initial")
    axes[2].bar(x + 0.25, identified_theta.as_array(), width=0.25, label="identified")
    axes[2].set_xticks(x, names)
    axes[2].set_yscale("log")
    axes[2].set_title("Physical parameters")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(output / "srix_regm_twin_summary.png", dpi=180)
    figure.savefig(output / "srix_regm_twin_summary.pdf")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--maximum-evaluations", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation/reference_data/srix_regm_twin_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output if arguments.output.is_absolute() else ROOT / arguments.output
    if output.exists() and any(output.iterdir()) and not arguments.overwrite:
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )
    if not Path(library).is_file():
        raise SystemExit(f"missing MFront library: {library}")

    true_theta = _theta_from_preset()
    initial_theta = SrixTheta4(
        tau0_mpa=1.25 * true_theta.tau0_mpa,
        r_mpa=0.80 * true_theta.r_mpa,
        q_mpa=1.30 * true_theta.q_mpa,
        b=0.75 * true_theta.b,
    )
    (
        displacement_history,
        time_increments,
        scored_states,
        forward_diagnostics,
        forward_seconds,
    ) = _generate_twin(
        pixels=arguments.pixels,
        library=library,
        threads=arguments.threads,
        theta=true_theta,
    )
    _progress(
        f"forward twin: {displacement_history.shape[0] - 1} accepted steps in "
        f"{forward_seconds:.3f} s"
    )
    problem = _problem(
        pixels=arguments.pixels,
        displacement_history=displacement_history,
        time_increments=time_increments,
        scored_states=scored_states,
        library=library,
        threads=arguments.threads,
        debug=True,
    )
    truth = problem.evaluate(true_theta)
    _progress(f"REGM truth: rms={truth.residual_rms:.6e}")
    initial = problem.evaluate(initial_theta)
    _progress(f"REGM initial: rms={initial.residual_rms:.6e}")
    optimizer_residual_scale_mm = max(initial.residual_rms, np.finfo(float).tiny)
    jacobians, fd_rows = _fd_study(problem, true_theta.log_coordinates())
    _progress("finite-difference study complete")
    jacobian = jacobians[3.0e-3]
    svd = problem.sensitivity_svd(jacobian, relative_threshold=1.0e-6)
    direction_probes = []
    for direction_index in range(svd.numerical_rank):
        direction = svd.right_singular_vectors[:, direction_index]
        probe = {"direction": direction_index + 1, "log_step": SINGULAR_DIRECTION_LOG_STEP}
        for sign, label in ((-1.0, "minus"), (1.0, "plus")):
            evaluation = problem.evaluate(
                SrixTheta4.from_log_coordinates(
                    true_theta.log_coordinates()
                    + sign * SINGULAR_DIRECTION_LOG_STEP * direction
                )
            )
            probe[f"{label}_cost"] = evaluation.cost
            probe[f"{label}_residual_rms"] = evaluation.residual_rms
        direction_probes.append(probe)
    _progress("singular-direction probes complete")

    cost_history: list[float] = []
    cache: dict[bytes, np.ndarray] = {}

    def residual(eta: np.ndarray) -> np.ndarray:
        key = np.asarray(eta, dtype=np.float64).tobytes()
        if key not in cache:
            values = (
                problem.residual_vector(SrixTheta4.from_log_coordinates(eta))
                / optimizer_residual_scale_mm
            )
            cache[key] = values
            cost_history.append(0.5 * float(values @ values))
        return cache[key]

    def jac(eta: np.ndarray) -> np.ndarray:
        return (
            problem.jacobian_fd(eta, relative_step=3.0e-3)
            / optimizer_residual_scale_mm
        )

    true_eta = true_theta.log_coordinates()
    optimization = least_squares(
        residual,
        initial_theta.log_coordinates(),
        jac=jac,
        bounds=(true_eta - np.log(4.0), true_eta + np.log(4.0)),
        max_nfev=arguments.maximum_evaluations,
        x_scale="jac",
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=None,
        verbose=0,
    )
    _progress(
        f"least squares: success={optimization.success} nfev={optimization.nfev} "
        f"cost={optimization.cost:.6e}"
    )
    identified_theta = SrixTheta4.from_log_coordinates(optimization.x)
    identified = problem.evaluate(identified_theta)
    identifiable = svd.right_singular_vectors[:, : svd.numerical_rank]
    log_error = optimization.x - true_eta
    projected_error = (
        0.0
        if svd.numerical_rank == 0
        else float(np.linalg.norm(identifiable.T @ log_error) / np.sqrt(svd.numerical_rank))
    )

    archive_fields = {
        "displacement_history": displacement_history,
        "orientations_deg": _orientation_map(arguments.pixels),
        "true_residual": truth.residual_vector,
        "initial_residual": initial.residual_vector,
        "identified_residual": identified.residual_vector,
        "jacobian": jacobian,
        **{
            f"truth_delta_u_state_{state.state_index}": state.pseudo_displacement
            for state in truth.states
            if state.pseudo_displacement is not None
            and state.state_index
            in {scored_states[0], scored_states[len(scored_states) // 2], scored_states[-1]}
        },
        **{
            f"initial_delta_u_state_{state.state_index}": state.pseudo_displacement
            for state in initial.states
            if state.pseudo_displacement is not None
            and state.state_index
            in {scored_states[0], scored_states[len(scored_states) // 2], scored_states[-1]}
        },
    }
    np.savez_compressed(output / "fields.npz", **archive_fields)
    report = {
        "schema_version": 1,
        "method": "SRIX-REGM exact digital twin",
        "git_sha": _git_head(),
        "dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "machine": platform.node(),
        "python": platform.python_version(),
        "mesh": [arguments.pixels, arguments.pixels],
        "pixel_size_mm": PIXEL_SIZE_MM,
        "states_replayed": list(range(1, displacement_history.shape[0])),
        "states_scored": list(scored_states),
        "time_increments": time_increments.tolist(),
        "history_origin": 0,
        "observation_profile": "identity_exact_twin",
        "whitening_profile": "identity_exact_twin",
        "k0_type": "orientation_dependent_cubic_plane_stress_sparse_factorized",
        "parameter_preset": DEFAULT_PARAMETER_SET,
        "parameters_true": true_theta.as_runtime_overrides(),
        "parameters_initial": initial_theta.as_runtime_overrides(),
        "parameters_identified": identified_theta.as_runtime_overrides(),
        "log_parameters_true": true_eta.tolist(),
        "log_parameters_initial": initial_theta.log_coordinates().tolist(),
        "log_parameters_identified": optimization.x.tolist(),
        "bounds_log": {
            "lower": (true_eta - np.log(4.0)).tolist(),
            "upper": (true_eta + np.log(4.0)).tolist(),
        },
        "optimizer": {
            "name": "scipy.optimize.least_squares",
            "maximum_evaluations": arguments.maximum_evaluations,
            "residual_scale_mm": optimizer_residual_scale_mm,
            "success": bool(optimization.success),
            "status": int(optimization.status),
            "message": str(optimization.message),
            "nfev": int(optimization.nfev),
            "njev": int(optimization.njev or 0),
            "cost_history": cost_history,
        },
        "fd_step_study": fd_rows,
        "sensitivity": {
            "singular_values": svd.singular_values.tolist(),
            "normalized_singular_values": svd.normalized_singular_values.tolist(),
            "right_singular_vectors": svd.right_singular_vectors.tolist(),
            "numerical_rank": svd.numerical_rank,
            "relative_threshold": svd.relative_threshold,
            "condition_number": svd.condition_number,
            "direction_probes": direction_probes,
        },
        "identifiable_log_error_rms": projected_error,
        "forward": {
            "seconds": forward_seconds,
            "diagnostics": forward_diagnostics,
        },
        "evaluations": {
            "truth": _evaluation_record(truth),
            "initial": _evaluation_record(initial),
            "identified": _evaluation_record(identified),
        },
        "claims": {
            "contains_global_newton_in_regm": False,
            "contains_global_krylov_in_regm": False,
            "p43_authorized": False,
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _save_figures(
        output,
        true_theta,
        initial_theta,
        identified_theta,
        svd.singular_values,
        cost_history,
    )
    print(
        json.dumps(
            {
                "truth_rms": truth.residual_rms,
                "initial_rms": initial.residual_rms,
                "identified_rms": identified.residual_rms,
                "rank": svd.numerical_rank,
                "singular_values": svd.singular_values.tolist(),
                "identified": identified_theta.as_runtime_overrides(),
                "projected_log_error_rms": projected_error,
                "forward_seconds": forward_seconds,
                "regm_truth_seconds": truth.timing.total_seconds,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
