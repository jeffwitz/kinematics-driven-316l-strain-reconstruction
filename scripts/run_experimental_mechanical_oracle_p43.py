#!/usr/bin/env python3
"""Run the sequential driven-J2 experimental oracle on a P43 crop."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from fem_inhouse.core.driven_j2 import DrivenJ2PlaneStressBatch
from fem_inhouse.core.plane_stress_material import PythonJ2PlaneStressBatch
from fem_inhouse.identification.dic_whitening import DICSpectralWhitener
from fem_inhouse.measurement import image_flow_to_canonical
from fem_inhouse.postprocessing.kinematics import (
    plane_stress_equivalent_strain,
    strain_from_displacement,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.workflows.experimental_mechanical_oracle import (
    ExperimentalOracleIncrementResult,
    ExperimentalOracleObjectiveWeights,
    ExperimentalOracleOptimizationConfig,
    ExperimentalOracleWarmStartRequest,
    ludwik_increment_history_from_measured_displacement,
    solve_experimental_mechanical_oracle_history,
    solve_fixed_plastic_increment_equilibrium,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/case_study"
NOISE = (
    ROOT
    / "validation/reference_data/dic_uncertainty_propagation_p0043_v1"
    / "centred_repeat_flow_pixels.npy"
)
DEFAULT_OUTPUT = ROOT / "validation/_generated/performance/experimental_oracle_p43_m20"
MEASURED_HISTORY = (
    ROOT
    / "validation/reference_data/dic_multistep_history_p0043_repaired_v1"
    / "repaired_history_mm.npy"
)
MEASURED_HISTORY_REPORT = MEASURED_HISTORY.with_name("report.json")
DEFAULT_CROP = (1610, 1630, 1075, 1095)
PIXEL_SIZE_MM = 0.00184
DIC_UNCERTAINTY_MM = 9.40e-5


def _equivalent_total_strain(displacement: np.ndarray) -> np.ndarray:
    strain = strain_from_displacement(
        displacement[..., 0],
        displacement[..., 1],
        spacing_x=PIXEL_SIZE_MM,
        spacing_y=PIXEL_SIZE_MM,
    )
    return plane_stress_equivalent_strain(
        strain.epsilon_xx,
        strain.epsilon_yy,
        strain.gamma_xy,
        poisson_ratio=0.30,
        shear_convention="engineering",
    )


def _relative_l2(values: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return float(np.linalg.norm(values - reference) / denominator)


def _spearman(values: np.ndarray, reference: np.ndarray) -> float:
    return float(spearmanr(values.ravel(), reference.ravel()).statistic)


def _plot_oracle_fields(
    path: Path,
    *,
    measured_displacement: np.ndarray,
    oracle_displacement: np.ndarray,
    ludwik_peeq: np.ndarray,
    oracle_peeq: np.ndarray,
    state_index: int,
) -> None:
    dic_evm = _equivalent_total_strain(measured_displacement)
    oracle_evm = _equivalent_total_strain(oracle_displacement)
    strain_difference = oracle_evm - dic_evm
    plastic_difference = oracle_peeq - ludwik_peeq
    strain_min = float(min(np.min(dic_evm), np.min(oracle_evm)))
    strain_max = float(max(np.max(dic_evm), np.max(oracle_evm)))
    plastic_min = float(min(np.min(ludwik_peeq), np.min(oracle_peeq)))
    plastic_max = float(max(np.max(ludwik_peeq), np.max(oracle_peeq)))
    strain_difference_max = max(float(np.max(np.abs(strain_difference))), 1.0e-15)
    plastic_difference_max = max(float(np.max(np.abs(plastic_difference))), 1.0e-15)
    panels = (
        (dic_evm, "DIC total equivalent strain", "viridis", strain_min, strain_max),
        (
            oracle_evm,
            "Equilibrated oracle total equivalent strain",
            "viridis",
            strain_min,
            strain_max,
        ),
        (
            strain_difference,
            "Oracle - DIC total equivalent strain",
            "coolwarm",
            -strain_difference_max,
            strain_difference_max,
        ),
        (ludwik_peeq, "Ludwik cumulative PEEQ prior", "magma", plastic_min, plastic_max),
        (oracle_peeq, "Driven-J2 cumulative PEEQ oracle", "magma", plastic_min, plastic_max),
        (
            plastic_difference,
            "Oracle - Ludwik cumulative PEEQ",
            "coolwarm",
            -plastic_difference_max,
            plastic_difference_max,
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for axis, (values, title, cmap, lower, upper) in zip(
        axes.flat, panels, strict=True
    ):
        image = axis.imshow(
            100.0 * values.T,
            origin="lower",
            cmap=cmap,
            vmin=100.0 * lower,
            vmax=100.0 * upper,
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xlabel("x index")
        axis.set_ylabel("y index")
        figure.colorbar(image, ax=axis, label="%")
    figure.suptitle(f"P43 M20 experimental mechanical oracle - DIC state {state_index}")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _load_measured_history(
    crop_nodes: tuple[int, int, int, int],
    increment_count: int,
    final_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    report = json.loads(MEASURED_HISTORY_REPORT.read_text(encoding="utf-8"))
    solve_x0, solve_x1, solve_y0, solve_y1 = map(int, report["solve_bounds"])
    x0, x1, y0, y1 = crop_nodes
    if not (
        solve_x0 <= x0 < x1 <= solve_x1
        and solve_y0 <= y0 < y1 <= solve_y1
    ):
        raise ValueError("crop_nodes must lie inside the measured-history solve bounds")
    source = np.load(MEASURED_HISTORY, mmap_mode="r", allow_pickle=False)
    maximum_state = source.shape[0] - 1
    if not 1 <= final_state <= maximum_state:
        raise ValueError(f"final measured state must lie between one and {maximum_state}")
    state_indices = np.rint(
        np.linspace(0.0, float(final_state), increment_count + 1)
    ).astype(np.int64)
    if np.any(np.diff(state_indices) <= 0):
        raise ValueError("increment count exceeds the measured DIC state resolution")
    local_x0, local_x1 = x0 - solve_x0, x1 - solve_x0
    local_y0, local_y1 = y0 - solve_y0, y1 - solve_y0
    history = np.asarray(
        source[
            state_indices,
            local_x0 : local_x1 + 1,
            local_y0 : local_y1 + 1,
            :,
        ],
        dtype=np.float64,
    )
    return history, state_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=DEFAULT_CROP)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--final-measured-state", type=int, default=40)
    parser.add_argument(
        "--history-mode",
        choices=("measured", "proportional-final"),
        default="measured",
    )
    parser.add_argument("--dic-weight", type=float, default=7.0e-5)
    parser.add_argument("--prior-weight", type=float, default=0.03)
    parser.add_argument("--spatial-weight", type=float, default=3.0e-4)
    parser.add_argument("--temporal-weight", type=float, default=0.0)
    parser.add_argument("--equilibrium-rms-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--projected-gradient-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--no-mechanical-warm-start", action="store_true")
    parser.add_argument(
        "--solution-method", choices=("augmented", "reduced"), default="reduced"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.increments <= 40:
        raise ValueError("increments must lie between one and forty")
    x0, x1, y0, y1 = args.crop_nodes
    nx, ny = x1 - x0, y1 - y0
    if nx < 3 or ny < 3:
        raise ValueError("crop must contain at least three pixels per direction")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    ux = np.load(DATA / "displacement_x_mm.npy", mmap_mode="r", allow_pickle=False)
    uy = np.load(DATA / "displacement_y_mm.npy", mmap_mode="r", allow_pickle=False)
    yield_map = np.load(DATA / "yield_stress_mpa.npy", mmap_mode="r", allow_pickle=False)
    hardening_map = np.load(
        DATA / "hardening_coefficient_mpa.npy", mmap_mode="r", allow_pickle=False
    )
    final_displacement = np.stack(
        (
            ux[x0 : x1 + 1, y0 : y1 + 1],
            uy[x0 : x1 + 1, y0 : y1 + 1],
        ),
        axis=-1,
    )
    if args.history_mode == "measured":
        measured_history, state_indices = _load_measured_history(
            (x0, x1, y0, y1), args.increments, args.final_measured_state
        )
        pseudo_times = state_indices.astype(np.float64) / 40.0
        history_description = "repaired measured P43 DIC history"
    else:
        pseudo_times = np.arange(args.increments + 1, dtype=np.float64) / args.increments
        measured_history = (
            pseudo_times[:, None, None, None] * final_displacement[None, ...]
        )
        state_indices = np.rint(40.0 * pseudo_times).astype(np.int64)
        history_description = "linear interpolation of the registered final P43 field"
    grid = StructuredGrid2D(nx, ny, nx * PIXEL_SIZE_MM, ny * PIXEL_SIZE_MM)
    kinematics = TwoSubcellDiagnostic2D(grid)
    point_yield = np.repeat(
        np.asarray(yield_map[x0:x1, y0:y1], dtype=np.float64).reshape(-1),
        kinematics.points_per_pixel,
    )
    point_hardening = np.repeat(
        np.asarray(hardening_map[x0:x1, y0:y1], dtype=np.float64).reshape(-1),
        kinematics.points_per_pixel,
    )
    prior_material = PythonJ2PlaneStressBatch(
        point_yield,
        point_hardening,
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    time_increments = np.diff(pseudo_times)
    ludwik_history = ludwik_increment_history_from_measured_displacement(
        material=prior_material,
        kinematics=kinematics,
        measured_displacement_history=measured_history,
        time_increments=time_increments,
    )

    recorded_noise = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    region_size = max(512, nx + 1, ny + 1)
    canonical_noise = image_flow_to_canonical(
        np.asarray(recorded_noise[:region_size, :region_size]),
        pixel_size_mm=PIXEL_SIZE_MM,
    )
    support_mask = np.ones((*grid.node_shape, 2), dtype=np.float64)
    support_mask[[0, -1], :, :] = 0.0
    support_mask[:, [0, -1], :] = 0.0
    whitener = DICSpectralWhitener.from_stationary_noise_field(
        canonical_noise,
        target_shape=grid.node_shape,
        sample_count=256,
        seed=42,
        remove_spatial_mean=False,
        support_mask=support_mask,
    )
    material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    warm_start_rows: list[dict[str, object]] = []

    def mechanical_warm_start(request: ExperimentalOracleWarmStartRequest) -> np.ndarray:
        warm_start = solve_fixed_plastic_increment_equilibrium(
            material=request.material,
            kinematics=request.kinematics,
            boundary_displacement=request.measured_displacement,
            equivalent_plastic_increment=request.ludwik_increment,
            initial_displacement=request.initial_displacement,
            time_increment=request.time_increment,
            equilibrium_rms_tolerance=1.0e-6,
        )
        warm_start_rows.append(
            {
                "increment": request.increment_index,
                "newton_iterations": warm_start.newton_iterations,
                "krylov_iterations": list(warm_start.krylov_iterations),
                "line_search_steps": list(warm_start.line_search_steps),
                "equilibrium_rms": warm_start.equilibrium_rms,
            }
        )
        return warm_start.displacement

    def report_progress(
        index: int, increment: ExperimentalOracleIncrementResult
    ) -> None:
        print(
            json.dumps(
                {
                    "event": "oracle_increment",
                    "increment": index,
                    "converged": increment.converged,
                    "equilibrium_rms": increment.equilibrium_rms,
                    "projected_gradient_inf": (
                        increment.augmented_iterations[-1].projected_gradient_inf
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    started = time.perf_counter()
    result = solve_experimental_mechanical_oracle_history(
        material=material,
        kinematics=kinematics,
        measured_displacement_history=measured_history,
        whitener=whitener,
        ludwik_increment_history=ludwik_history,
        initial_displacement_history=measured_history,
        displacement_warm_start=(
            mechanical_warm_start
            if args.solution_method == "augmented"
            and not args.no_mechanical_warm_start
            else None
        ),
        progress_callback=report_progress,
        solution_method=args.solution_method,
        weights=ExperimentalOracleObjectiveWeights(
            dic=args.dic_weight,
            ludwik_prior=args.prior_weight,
            spatial_plastic_increment=args.spatial_weight,
            temporal_plastic_increment=args.temporal_weight,
        ),
        config=ExperimentalOracleOptimizationConfig(
            maximum_augmented_iterations=20,
            maximum_inner_iterations=400,
            inner_function_tolerance=1.0e-15,
            equilibrium_rms_tolerance=args.equilibrium_rms_tolerance,
            projected_gradient_tolerance=args.projected_gradient_tolerance,
            initial_penalty=1.0,
            penalty_growth=10.0,
            sufficient_constraint_reduction=0.5,
            displacement_variable_scale=1.0e-6,
            plastic_increment_variable_scale=1.0e-5,
        ),
        time_increments=time_increments,
    )
    elapsed = time.perf_counter() - started
    increment_rows = []
    for index, increment in enumerate(result.increments, start=1):
        raw_dic_misfit = increment.dic_misfit / args.dic_weight
        increment_rows.append(
            {
                "increment": index,
                "converged": increment.converged,
                "message": increment.message,
                "equilibrium_rms": increment.equilibrium_rms,
                "raw_dic_misfit_per_dof": raw_dic_misfit,
                "discrepancy_ratio": raw_dic_misfit / 0.5,
                "augmented_iterations": len(increment.augmented_iterations),
                "projected_gradient_inf": (
                    increment.augmented_iterations[-1].projected_gradient_inf
                ),
                "constitutive_rejections": increment.constitutive_rejections,
                "maximum_delta_p": float(
                    np.max(increment.equivalent_plastic_increment)
                ),
                "augmented_history": [
                    {
                        "index": item.index,
                        "penalty": item.penalty,
                        "objective": item.objective,
                        "equilibrium_rms": item.equilibrium_rms,
                        "projected_gradient_inf": item.projected_gradient_inf,
                        "inner_iterations": item.inner_iterations,
                        "inner_success": item.inner_success,
                    }
                    for item in increment.augmented_iterations
                ],
            }
        )
    report = {
        "schema_version": 1,
        "status": "completed" if result.completed else "failed",
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "elapsed_seconds": elapsed,
        "crop_nodes": list(args.crop_nodes),
        "mesh": [nx, ny],
        "accepted_increments": len(result.equivalent_plastic_increment_history),
        "requested_increments": args.increments,
        "failed_increment": result.failed_increment,
        "dic_history": history_description,
        "dic_history_source": (
            str(MEASURED_HISTORY.relative_to(ROOT))
            if args.history_mode == "measured"
            else None
        ),
        "dic_state_indices": state_indices.tolist(),
        "ordered_pseudo_time_only": True,
        "physical_force_history_available": False,
        "absolute_constitutive_identification": False,
        "weights": {
            "dic": args.dic_weight,
            "ludwik_prior": args.prior_weight,
            "spatial_plastic_increment": args.spatial_weight,
            "temporal_plastic_increment": args.temporal_weight,
        },
        "solution_method": args.solution_method,
        "mechanical_warm_start": {
            "enabled": (
                args.solution_method == "augmented"
                and not args.no_mechanical_warm_start
            ),
            "increments": warm_start_rows,
        },
        "increments": increment_rows,
    }
    accepted_count = len(result.increments) if result.completed else len(result.increments) - 1
    if accepted_count:
        measured_final = measured_history[accepted_count]
        oracle_final = result.displacement_history[-1]
        displacement_correction = oracle_final - measured_final
        measured_evm = _equivalent_total_strain(measured_final)
        oracle_evm = _equivalent_total_strain(oracle_final)
        ludwik_peeq = np.sum(ludwik_history[:accepted_count], axis=0).mean(axis=-1)
        oracle_peeq = result.equivalent_plastic_strain_history[-1].mean(axis=-1)
        report["field_comparison_final_state"] = {
            "dic_state": int(state_indices[accepted_count]),
            "dic_uncertainty_mm": DIC_UNCERTAINTY_MM,
            "displacement_correction_inf_mm": float(
                np.max(np.abs(displacement_correction))
            ),
            "displacement_correction_rms_mm": float(
                np.sqrt(np.mean(displacement_correction**2))
            ),
            "displacement_correction_inf_over_uncertainty": float(
                np.max(np.abs(displacement_correction)) / DIC_UNCERTAINTY_MM
            ),
            "displacement_relative_l2": _relative_l2(oracle_final, measured_final),
            "equivalent_strain_relative_l2": _relative_l2(
                oracle_evm, measured_evm
            ),
            "equivalent_strain_spearman": _spearman(oracle_evm, measured_evm),
            "dic_equivalent_strain_maximum": float(np.max(measured_evm)),
            "oracle_equivalent_strain_maximum": float(np.max(oracle_evm)),
            "peeq_prior_oracle_relative_l2": _relative_l2(
                oracle_peeq, ludwik_peeq
            ),
            "peeq_prior_oracle_spearman": _spearman(oracle_peeq, ludwik_peeq),
            "peeq_prior_maximum": float(np.max(ludwik_peeq)),
            "peeq_oracle_maximum": float(np.max(oracle_peeq)),
        }
    accepted_stress = (
        np.stack(
            [
                item.linearisation.trial.stress_in_plane_mpa.reshape(nx, ny, 2, 3)
                for item in result.increments[:accepted_count]
            ]
        )
        if accepted_count
        else np.empty((0, nx, ny, 2, 3))
    )
    np.savez_compressed(
        output / "fields.npz",
        measured_displacement_history=measured_history,
        oracle_displacement_history=result.displacement_history,
        ludwik_increment_history=ludwik_history,
        oracle_increment_history=result.equivalent_plastic_increment_history,
        oracle_equivalent_plastic_strain_history=(
            result.equivalent_plastic_strain_history
        ),
        oracle_stress_history_mpa=accepted_stress,
    )
    if accepted_count:
        figure_path = output / "fields.png"
        _plot_oracle_fields(
            figure_path,
            measured_displacement=measured_history[accepted_count],
            oracle_displacement=result.displacement_history[-1],
            ludwik_peeq=np.sum(ludwik_history[:accepted_count], axis=0).mean(axis=-1),
            oracle_peeq=result.equivalent_plastic_strain_history[-1].mean(axis=-1),
            state_index=int(state_indices[accepted_count]),
        )
        report["figure"] = str(figure_path.relative_to(ROOT))
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
