#!/usr/bin/env python3
"""Qualify direct FEMU sensitivities on the exact M8 SRIX twin."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import subspace_angles
from scipy.sparse.linalg import LinearOperator, gmres

from fem_inhouse.core.plane_stress_material import evaluate_in_plane_response
from fem_inhouse.identification.dic_whitening import DICSpectralTransfer
from fem_inhouse.identification.srix_equilibrium_gap import SrixTheta4
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.spectral2d.newton_ebi import (
    EBISpectralSolverConfig,
    pack_interior,
    unpack_interior,
)
from fem_inhouse.spectral2d.newton_two_state import (
    TwoStateIncrementFields,
    solve_two_state_dirichlet_plane_stress,
)
from fem_inhouse.spectral2d.step_control import AdaptiveStepConfig
from scripts.qualify_srix_regm_information_geometry import _geometry, _plot
from scripts.qualify_srix_regm_transfer_noise import _WrapFreeTransfer
from scripts.qualify_srix_regm_twin import (
    PIXEL_SIZE_MM,
    SUBSTEPS_PER_SEGMENT,
    _boundary_history,
    _material_factory,
    _orientation_map,
    _theta_from_preset,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "validation/reference_data/srix_regm_information_geometry_v1"
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
DEFAULT_OUTPUT = ROOT / "validation/reference_data/srix_femu_direct_sensitivity_v1"
FD_STEP = 3.0e-3


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _reference_config() -> EBISpectralSolverConfig:
    return EBISpectralSolverConfig(
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
    )


def _seed_config() -> EBISpectralSolverConfig:
    """Fast, non-scientific policy used only to propose path nodes."""

    return EBISpectralSolverConfig(
        relative_equilibrium_tolerance=1.0e-5,
        maximum_newton_iterations=12,
        verify_final_state=False,
        adaptive_stepping_enabled=True,
        adaptive_step=AdaptiveStepConfig(
            initial_increment_fraction=1.0 / (8 * SUBSTEPS_PER_SEGMENT),
            minimum_increment_fraction=1.0 / 1024.0,
            maximum_increment_fraction=1.0 / (8 * SUBSTEPS_PER_SEGMENT),
            increment_growth_factor=2.0,
            increment_cutback_factor=0.5,
            target_newton_iterations_min=4,
            target_newton_iterations_max=10,
            line_search_difficult_threshold=0.25,
            maximum_cutbacks_per_step=3,
        ),
    )


def _path_search_config() -> EBISpectralSolverConfig:
    """Strict-but-fail-fast policy used only to qualify a common path."""

    return EBISpectralSolverConfig(
        relative_equilibrium_tolerance=1.0e-6,
        maximum_newton_iterations=12,
        gmres_maximum_iterations=40,
        gmres_restart=20,
        verify_final_state=False,
        adaptive_stepping_enabled=False,
        maximum_line_search_reductions=6,
    )


def _oracle_config() -> EBISpectralSolverConfig:
    """Scientific fixed-path policy used for the final common-path replay."""

    return EBISpectralSolverConfig(
        relative_equilibrium_tolerance=1.0e-6,
        maximum_newton_iterations=80,
        verify_final_state=True,
        adaptive_stepping_enabled=False,
        maximum_line_search_reductions=20,
    )


def _reference_trajectory(
    *,
    pixels: int,
    library: str,
    threads: int,
    theta: SrixTheta4,
    config: EBISpectralSolverConfig | None = None,
) -> tuple[list[TwoStateIncrementFields], dict[str, Any], float]:
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    orientations = _orientation_map(pixels)
    material = _material_factory(
        pixels=pixels, orientations=orientations, library=library, threads=threads
    )(theta.as_runtime_overrides())
    fields: list[TwoStateIncrementFields] = []

    def observe(value: TwoStateIncrementFields) -> None:
        fields.append(
            TwoStateIncrementFields(
                increment=value.increment,
                start_fraction=value.start_fraction,
                end_fraction=value.end_fraction,
                time_increment=value.time_increment,
                boundary=np.asarray(value.boundary).copy(),
                displacement=np.asarray(value.displacement).copy(),
                sample_strain=np.asarray(value.sample_strain).copy(),
                stress_in_plane_mpa=np.asarray(value.stress_in_plane_mpa).copy(),
                algorithmic_tangent_in_plane_mpa=np.asarray(
                    value.algorithmic_tangent_in_plane_mpa
                ).copy(),
                plastic_strain_tensor=(
                    None
                    if value.plastic_strain_tensor is None
                    else np.asarray(value.plastic_strain_tensor).copy()
                ),
            )
        )

    started = time.perf_counter()
    result = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=material,
        boundary_displacement_history=_boundary_history(grid),
        config=_reference_config() if config is None else config,
        increment_observer=observe,
    )
    elapsed = time.perf_counter() - started
    diagnostics = {
        "accepted_increments": len(fields),
        "solver": {
            "adaptive_step_history": [
                dict(item) for item in result.diagnostics.adaptive_step_history
            ],
            "load_step_attempts": [
                {
                    "accepted": attempt.accepted,
                    "load_fraction_start": attempt.load_fraction_start,
                    "load_fraction_end": attempt.load_fraction_end,
                    "failure_reason": attempt.failure_reason,
                }
                for attempt in result.diagnostics.load_step_attempts
            ],
        },
        "elapsed_seconds": elapsed,
    }
    return fields, diagnostics, elapsed


def _solve_exact_tangent(
    *,
    grid: StructuredGrid2D,
    kinematics: TwoSubcellDiagnostic2D,
    tangent: np.ndarray,
    rhs: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Solve the exact FEMU matrix-free tangent action for one RHS."""

    size = rhs.size

    def action(vector: np.ndarray) -> np.ndarray:
        nodal = unpack_interior(vector, grid)
        sample_increment = kinematics.strain_samples(nodal)
        stress_increment = np.einsum(
            "xyqij,xyqj->xyqi", tangent, sample_increment
        )
        return pack_interior(kinematics.divergence_from_sample_stress(stress_increment))

    operator = LinearOperator((size, size), matvec=action, dtype=np.float64)
    solution, info = gmres(
        operator,
        np.asarray(rhs, dtype=np.float64),
        rtol=1.0e-10,
        atol=0.0,
        restart=50,
        maxiter=400,
        callback_type="pr_norm",
    )
    if info != 0:
        raise RuntimeError(f"exact FEMU tangent GMRES failed with info={info}")
    return np.asarray(solution, dtype=np.float64), int(info)


def _direct_jacobian(
    *,
    fields: list[TwoStateIncrementFields],
    scored: tuple[int, ...],
    orientations: np.ndarray,
    theta: SrixTheta4,
    library: str,
    threads: int,
    transfer: Any,
    h: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    pixels = fields[0].displacement.shape[0] - 1
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    kinematics = TwoSubcellDiagnostic2D(grid)
    factory = _material_factory(
        pixels=pixels, orientations=orientations, library=library, threads=threads
    )
    eta = theta.log_coordinates()
    parameter_names = ("tau0", "R", "Q", "b")
    shadows: list[tuple[Any, Any]] = []
    for index in range(4):
        plus = eta.copy()
        minus = eta.copy()
        plus[index] += h
        minus[index] -= h
        shadows.append(
            (
                factory(SrixTheta4.from_log_coordinates(plus).as_runtime_overrides()),
                factory(SrixTheta4.from_log_coordinates(minus).as_runtime_overrides()),
            )
        )

    scored_vectors: list[list[np.ndarray]] = [[] for _ in range(4)]
    gmres_calls = 0
    started = time.perf_counter()
    for state_index, accepted in enumerate(fields, start=1):
        base_strain = np.asarray(accepted.sample_strain, dtype=np.float64)
        base_tangent = np.asarray(
            accepted.algorithmic_tangent_in_plane_mpa, dtype=np.float64
        )
        forcings: list[np.ndarray] = []
        for parameter_index, (plus, minus) in enumerate(shadows):
            parameter_name = parameter_names[parameter_index]
            try:
                plus_trial = evaluate_in_plane_response(
                    plus,
                    base_strain.reshape(-1, 3),
                    time_increment=accepted.time_increment,
                    response_level="tangent",
                    consistent_tangent=True,
                )
                minus_trial = evaluate_in_plane_response(
                    minus,
                    base_strain.reshape(-1, 3),
                    time_increment=accepted.time_increment,
                    response_level="tangent",
                    consistent_tangent=True,
                )
            except Exception as error:
                raise RuntimeError(
                    "direct shadow integration failed at accepted increment "
                    f"{state_index}, parameter {parameter_name}, sign pair "
                    f"plus/minus, phase fixed_current_strain: {error}"
                ) from error
            stress_difference = (
                np.asarray(plus_trial.stress_in_plane_mpa)
                - np.asarray(minus_trial.stress_in_plane_mpa)
            ).reshape(*grid.pixel_shape, 2, 3) / (2.0 * h)
            forcings.append(
                -pack_interior(kinematics.divergence_from_sample_stress(stress_difference))
            )
            plus.revert()
            minus.revert()

        sensitivities: list[np.ndarray] = []
        for forcing in forcings:
            solution, _ = _solve_exact_tangent(
                grid=grid,
                kinematics=kinematics,
                tangent=base_tangent,
                rhs=forcing,
            )
            gmres_calls += 1
            sensitivities.append(unpack_interior(solution, grid))

        if state_index in scored:
            for index, sensitivity in enumerate(sensitivities):
                scored_vectors[index].append(
                    np.asarray(transfer.apply(sensitivity), dtype=np.float64).reshape(-1)
                )

        for index, (plus, minus) in enumerate(shadows):
            parameter_name = parameter_names[index]
            sensitivity_strain = kinematics.strain_samples(sensitivities[index])
            plus_strain = base_strain + h * sensitivity_strain
            minus_strain = base_strain - h * sensitivity_strain
            try:
                evaluate_in_plane_response(
                    plus,
                    plus_strain.reshape(-1, 3),
                    time_increment=accepted.time_increment,
                    response_level="residual",
                    consistent_tangent=False,
                )
            except Exception as error:
                raise RuntimeError(
                    "direct shadow history advance failed at accepted increment "
                    f"{state_index}, parameter {parameter_name}, sign plus, "
                    f"phase history_advance: {error}"
                ) from error
            try:
                evaluate_in_plane_response(
                    minus,
                    minus_strain.reshape(-1, 3),
                    time_increment=accepted.time_increment,
                    response_level="residual",
                    consistent_tangent=False,
                )
            except Exception as error:
                raise RuntimeError(
                    "direct shadow history advance failed at accepted increment "
                    f"{state_index}, parameter {parameter_name}, sign minus, "
                    f"phase history_advance: {error}"
                ) from error
            try:
                plus.commit()
                minus.commit()
            except Exception as error:
                raise RuntimeError(
                    "direct shadow history commit failed at accepted increment "
                    f"{state_index}, parameter {parameter_name}: {error}"
                ) from error

    matrix = np.column_stack([np.concatenate(values) for values in scored_vectors])
    return matrix, {"elapsed_seconds": time.perf_counter() - started, "gmres_solves": gmres_calls}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--library", default="build/mfront/src/libBehaviour.so")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)

    fields, forward_diagnostics, forward_seconds = _reference_trajectory(
        pixels=args.pixels,
        library=args.library,
        threads=args.threads,
        theta=_theta_from_preset(),
    )
    scored = tuple(
        int(value)
        for value in json.loads((SOURCE / "report.json").read_text())["states_scored"]
    )
    if scored[-1] > len(fields):
        raise RuntimeError("registered scored endpoint exceeds accepted trajectory")
    orientations = _orientation_map(args.pixels)
    transfer = _WrapFreeTransfer(DICSpectralTransfer.from_sinusoidal_csv(TRANSFER))
    matrix, sensitivity_timing = _direct_jacobian(
        fields=fields,
        scored=scored,
        orientations=orientations,
        theta=_theta_from_preset(),
        library=args.library,
        threads=args.threads,
        transfer=transfer,
        h=FD_STEP,
    )
    reference = np.load(SOURCE / "jacobians.npz")
    femu = np.asarray(reference["FEMU_observed"], dtype=np.float64)
    column_errors = []
    column_cosines = []
    for index in range(4):
        direct = matrix[:, index]
        target = femu[:, index]
        column_errors.append(float(np.linalg.norm(direct - target) / np.linalg.norm(target)))
        column_cosines.append(
            float(np.dot(direct, target) / (np.linalg.norm(direct) * np.linalg.norm(target)))
        )
    geometry = _geometry(matrix)
    geometry["cumulative"] = []
    femu_geometry = json.loads((SOURCE / "report.json").read_text())["geometries"]["FEMU_observed"]
    femu_geometry = {**femu_geometry, "cumulative": []}
    angles = {
        str(count): np.degrees(
            subspace_angles(
                np.asarray(geometry["right_singular_vectors"])[:, :count],
                np.asarray(femu_geometry["right_singular_vectors"])[:, :count],
            )
        ).tolist()
        for count in (1, 2, 3)
    }
    report = {
        "schema_version": 1,
        "method": "direct FEMU matrix-free tangent with persistent constitutive shadow histories",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "states_scored": list(scored),
        "accepted_increments": len(fields),
        "fd_step_log": FD_STEP,
        "forward_seconds": forward_seconds,
        "sensitivity_timing": sensitivity_timing,
        "forward_diagnostics": forward_diagnostics,
        "jacobian_relative_column_l2_errors": column_errors,
        "jacobian_column_cosines": column_cosines,
        "geometry": geometry,
        "femu_geometry": femu_geometry,
        "principal_angles_to_femu_degrees": angles,
        "claims": {"direct_femu_qualified": False, "p43_authorized": False},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(output / "jacobian.npz", FEMU_direct=matrix, FEMU_reference=femu)
    _plot({"FEMU_direct": geometry, "FEMU_observed": femu_geometry}, output)
    print(column_errors, column_cosines, geometry["normalized_singular_values"], flush=True)


if __name__ == "__main__":
    main()
