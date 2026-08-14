#!/usr/bin/env python3
"""Qualify the P0 oracle on a localised synthetic with measured P43 noise."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from fem_inhouse.core.driven_j2 import DrivenJ2PlaneStressBatch
from fem_inhouse.identification.dic_whitening import DICSpectralWhitener
from fem_inhouse.measurement import image_flow_to_canonical
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.workflows.experimental_mechanical_oracle import (
    ExperimentalOracleObjectiveWeights,
    ExperimentalOracleOptimizationConfig,
    solve_experimental_mechanical_oracle_increment,
)
from fem_inhouse.workflows.experimental_oracle_synthetic import (
    diagonal_localised_plastic_increment,
    solve_fixed_plastic_increment_equilibrium,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOISE = (
    ROOT
    / "validation/reference_data/dic_uncertainty_propagation_p0043_v1"
    / "centred_repeat_flow_pixels.npy"
)
DEFAULT_OUTPUT = (
    ROOT / "validation/_generated/performance/experimental_oracle_synthetic_p43_noise"
)
PIXEL_SIZE_MM = 0.00184


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(np.linalg.norm(reference), np.finfo(float).tiny)
    )


def _tensile_boundary(grid: StructuredGrid2D) -> np.ndarray:
    x = np.linspace(0.0, grid.length_x, grid.node_shape[0])[:, None]
    y = np.linspace(0.0, grid.length_y, grid.node_shape[1])[None, :]
    displacement = np.empty((*grid.node_shape, 2), dtype=np.float64)
    displacement[..., 0] = -0.001 * x
    displacement[..., 1] = 0.008 * y
    return displacement


def _plot(
    path: Path,
    *,
    truth_increment: np.ndarray,
    prior_increment: np.ndarray,
    recovered_increment: np.ndarray,
    noisy_error: np.ndarray,
    recovered_error: np.ndarray,
) -> None:
    truth = truth_increment.mean(axis=-1)
    prior = prior_increment.mean(axis=-1)
    recovered = recovered_increment.mean(axis=-1)
    vmin = float(min(truth.min(), prior.min(), recovered.min()))
    vmax = float(max(truth.max(), prior.max(), recovered.max()))
    difference_limit = float(
        max(np.abs(recovered - truth).max(), np.finfo(float).tiny)
    )
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), constrained_layout=True)
    for axis, field, title in zip(
        axes[0],
        (truth, prior, recovered),
        ("Delta p truth", "Ludwik prior", "Recovered Delta p"),
        strict=True,
    ):
        image = axis.imshow(field.T, origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.8)
    difference = recovered - truth
    image = axes[1, 0].imshow(
        difference.T,
        origin="lower",
        vmin=-difference_limit,
        vmax=difference_limit,
        cmap="coolwarm",
    )
    axes[1, 0].set_title("Recovered - truth")
    figure.colorbar(image, ax=axes[1, 0], shrink=0.8)
    for axis, field, title in zip(
        axes[1, 1:],
        (noisy_error, recovered_error),
        ("Measured displacement error", "Recovered displacement error"),
        strict=True,
    ):
        magnitude = np.linalg.norm(field, axis=-1)
        image = axis.imshow(magnitude.T, origin="lower", cmap="viridis")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=Path, default=DEFAULT_NOISE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--dic-weight", type=float, default=7.0e-5)
    parser.add_argument("--prior-weight", type=float, default=0.03)
    parser.add_argument("--spatial-weight", type=float, default=3.0e-4)
    parser.add_argument("--noise-origin-x", type=int, default=100)
    parser.add_argument("--noise-origin-y", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.grid_size < 3:
        raise ValueError("grid-size must be at least three")
    if args.dic_weight <= 0.0:
        raise ValueError("dic-weight must be positive for discrepancy qualification")
    output = args.output
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    grid = StructuredGrid2D(args.grid_size, args.grid_size, 0.5, 0.5)
    kinematics = TwoSubcellDiagnostic2D(grid)
    boundary = _tensile_boundary(grid)
    truth_width_pixels = max(1.0, 0.15 * args.grid_size)
    prior_width_pixels = 0.65 * truth_width_pixels
    prior_offset_pixels = 0.15 * args.grid_size
    truth_increment = diagonal_localised_plastic_increment(
        grid,
        points_per_pixel=kinematics.points_per_pixel,
        background=1.0e-4,
        amplitude=8.0e-4,
        width_pixels=truth_width_pixels,
    )
    prior_increment = diagonal_localised_plastic_increment(
        grid,
        points_per_pixel=kinematics.points_per_pixel,
        background=1.0e-4,
        amplitude=6.0e-4,
        width_pixels=prior_width_pixels,
        offset=prior_offset_pixels,
    )
    truth_material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    truth = solve_fixed_plastic_increment_equilibrium(
        material=truth_material,
        kinematics=kinematics,
        boundary_displacement=boundary,
        equivalent_plastic_increment=truth_increment,
        equilibrium_rms_tolerance=1.0e-8,
    )

    recorded = np.load(args.noise, mmap_mode="r", allow_pickle=False)
    region_size = max(256, args.grid_size + 1)
    canonical_noise = image_flow_to_canonical(
        np.asarray(recorded[:region_size, :region_size]),
        pixel_size_mm=PIXEL_SIZE_MM,
    )
    support_mask = np.ones((*grid.node_shape, 2), dtype=np.float64)
    support_mask[[0, -1], :, :] = 0.0
    support_mask[:, [0, -1], :] = 0.0
    whitener = DICSpectralWhitener.from_stationary_noise_field(
        canonical_noise,
        target_shape=grid.node_shape,
        sample_count=128,
        seed=42,
        support_mask=support_mask,
    )
    start_x, start_y = args.noise_origin_x, args.noise_origin_y
    stop_x = start_x + grid.node_shape[0]
    stop_y = start_y + grid.node_shape[1]
    if start_x < 0 or start_y < 0 or stop_x > region_size or stop_y > region_size:
        raise ValueError("noise origin must select a full window inside the noise region")
    noise = canonical_noise[start_x:stop_x, start_y:stop_y].copy()
    support_sum = np.sum(support_mask, axis=(0, 1), keepdims=True)
    noise -= np.sum(
        noise * support_mask,
        axis=(0, 1),
        keepdims=True,
    ) / support_sum
    noise *= support_mask
    measured = truth.displacement + noise

    oracle_material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    recovered = solve_experimental_mechanical_oracle_increment(
        material=oracle_material,
        kinematics=kinematics,
        measured_displacement=measured,
        whitener=whitener,
        ludwik_increment=prior_increment,
        initial_displacement=measured,
        initial_equivalent_plastic_increment=prior_increment,
        weights=ExperimentalOracleObjectiveWeights(
            dic=args.dic_weight,
            ludwik_prior=args.prior_weight,
            spatial_plastic_increment=args.spatial_weight,
        ),
        config=ExperimentalOracleOptimizationConfig(
            maximum_augmented_iterations=20,
            maximum_inner_iterations=400,
            equilibrium_rms_tolerance=1.0e-4,
            initial_penalty=1.0,
            penalty_growth=10.0,
            sufficient_constraint_reduction=0.5,
            displacement_variable_scale=1.0e-6,
            plastic_increment_variable_scale=1.0e-3,
        ),
    )
    recovered_increment = recovered.equivalent_plastic_increment
    measured_error = float(np.linalg.norm(noise))
    recovered_displacement_error = float(
        np.linalg.norm(recovered.displacement - truth.displacement)
    )
    prior_error = _relative_l2(prior_increment, truth_increment)
    recovered_error = _relative_l2(recovered_increment, truth_increment)
    raw_dic_misfit = recovered.dic_misfit / args.dic_weight
    discrepancy_target = 0.5
    discrepancy_ratio = raw_dic_misfit / discrepancy_target
    projected_gradient = recovered.augmented_iterations[-1].projected_gradient_inf
    prior_spearman = float(
        spearmanr(truth_increment.ravel(), prior_increment.ravel()).statistic
    )
    recovered_spearman = float(
        spearmanr(truth_increment.ravel(), recovered_increment.ravel()).statistic
    )
    recovery_qualified = bool(
        recovered.converged
        and recovered_error < prior_error
        and recovered_spearman > prior_spearman
        and recovered_displacement_error < measured_error
        and 0.5 <= discrepancy_ratio <= 2.0
        and projected_gradient <= 1.0e-3
    )
    truth_dic_misfit = whitener.quadratic_misfit(
        truth.displacement - measured
    ) / measured.size
    report = {
        "schema_version": 1,
        "status": (
            "qualified_synthetic_recovery"
            if recovery_qualified
            else "completed_but_recovery_not_qualified"
        ),
        "recovery_qualified": recovery_qualified,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "noise_source": str(args.noise.resolve()),
        "noise_pixel_size_mm": PIXEL_SIZE_MM,
        "noise_origin": [start_x, start_y],
        "grid_shape": list(grid.pixel_shape),
        "synthetic_band": {
            "truth_width_pixels": truth_width_pixels,
            "prior_width_pixels": prior_width_pixels,
            "prior_offset_pixels": prior_offset_pixels,
        },
        "weights": {
            "dic": args.dic_weight,
            "ludwik_prior": args.prior_weight,
            "spatial_plastic_increment": args.spatial_weight,
        },
        "truth_equilibrium_rms": truth.equilibrium_rms,
        "oracle_equilibrium_rms": recovered.equilibrium_rms,
        "constitutive_rejections": recovered.constitutive_rejections,
        "prior_relative_l2": prior_error,
        "recovered_relative_l2": recovered_error,
        "prior_spearman": prior_spearman,
        "recovered_spearman": recovered_spearman,
        "measured_displacement_error_l2_mm": measured_error,
        "recovered_displacement_error_l2_mm": recovered_displacement_error,
        "displacement_error_ratio_recovered_over_measured": (
            recovered_displacement_error / measured_error
        ),
        "objective_components": {
            "dic_misfit": recovered.dic_misfit,
            "raw_dic_misfit_per_dof": raw_dic_misfit,
            "truth_dic_misfit": truth_dic_misfit,
            "ludwik_prior": recovered.ludwik_prior,
            "spatial_regularisation": recovered.spatial_regularisation,
            "temporal_regularisation": recovered.temporal_regularisation,
            "augmented_equilibrium": recovered.augmented_equilibrium,
        },
        "discrepancy": {
            "target_per_dof": discrepancy_target,
            "ratio_recovered_over_target": discrepancy_ratio,
        },
        "augmented_iterations": [
            {
                "index": item.index,
                "penalty": item.penalty,
                "objective": item.objective,
                "equilibrium_rms": item.equilibrium_rms,
                "inner_iterations": item.inner_iterations,
                "inner_success": item.inner_success,
                "projected_gradient_inf": item.projected_gradient_inf,
            }
            for item in recovered.augmented_iterations
        ],
        "interpretation": (
            "Formulation/gradient P0 qualification only. This synthetic does not "
            "identify an absolute constitutive law and uses no synchronized force."
        ),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output / "fields.npz",
        truth_displacement=truth.displacement,
        measured_displacement=measured,
        recovered_displacement=recovered.displacement,
        truth_increment=truth_increment,
        ludwik_prior_increment=prior_increment,
        recovered_increment=recovered_increment,
        measured_noise=noise,
        truth_stress_mpa=truth.linearisation.trial.stress_in_plane_mpa,
        recovered_stress_mpa=recovered.linearisation.trial.stress_in_plane_mpa,
    )
    _plot(
        output / "fields.png",
        truth_increment=truth_increment,
        prior_increment=prior_increment,
        recovered_increment=recovered_increment,
        noisy_error=noise,
        recovered_error=recovered.displacement - truth.displacement,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
