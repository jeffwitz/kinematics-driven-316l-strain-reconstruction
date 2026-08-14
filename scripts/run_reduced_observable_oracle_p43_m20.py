#!/usr/bin/env python3
"""Run the constrained reduced plastic oracle on archived P43 M20 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fem_inhouse.core.driven_j2 import DrivenJ2PlaneStressBatch
from fem_inhouse.identification.dic_whitening import DICSpectralWhitener
from fem_inhouse.measurement import image_flow_to_canonical
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.workflows.experimental_mechanical_oracle import (
    ExperimentalOracleObjectiveWeights,
    ExperimentalOracleOptimizationConfig,
    solve_experimental_mechanical_oracle_history,
)

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ROOT / "validation/_generated/performance/experimental_oracle_p43_m20/fields.npz"
NOISE = (
    ROOT
    / (
        "validation/reference_data/dic_uncertainty_propagation_p0043_v1/"
        "centred_repeat_flow_pixels.npy"
    )
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--prior-weight", type=float, default=0.03)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fields = np.load(FIELDS, allow_pickle=False)
    measured = np.asarray(fields["measured_displacement_history"])
    ludwik = np.asarray(fields["ludwik_increment_history"])
    basis = np.asarray(
        np.load(
            ROOT
            / (
                "validation/_generated/performance/experimental_oracle_p43_m20/"
                "observability_transfer_40states/basis.npz"
            ),
            allow_pickle=False,
        )["basis"][:, : args.rank]
    )
    nx, ny = measured.shape[1] - 1, measured.shape[2] - 1
    grid = StructuredGrid2D(nx, ny, 0.00184 * nx, 0.00184 * ny)
    kinematics = TwoSubcellDiagnostic2D(grid)
    noise = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    canonical_noise = image_flow_to_canonical(
        np.asarray(noise[:512, :512]), pixel_size_mm=0.00184
    )
    support = np.ones((*grid.node_shape, 2), dtype=np.float64)
    support[[0, -1], :, :] = 0.0
    support[:, [0, -1], :] = 0.0
    whitener = DICSpectralWhitener.from_stationary_noise_field(
        canonical_noise,
        target_shape=grid.node_shape,
        sample_count=256,
        seed=42,
        remove_spatial_mean=False,
        support_mask=support,
    )
    material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )

    config = ExperimentalOracleOptimizationConfig(
        equilibrium_rms_tolerance=1.0e-4,
        projected_gradient_tolerance=1.0e-2,
        maximum_inner_iterations=80,
    )
    weights = ExperimentalOracleObjectiveWeights(
        dic=7.0e-5,
        ludwik_prior=args.prior_weight,
        spatial_plastic_increment=3.0e-4,
        temporal_plastic_increment=0.0,
    )
    result = solve_experimental_mechanical_oracle_history(
        material=material,
        kinematics=kinematics,
        measured_displacement_history=measured,
        whitener=whitener,
        ludwik_increment_history=ludwik,
        solution_method="reduced",
        weights=weights,
        config=config,
        plastic_basis=basis,
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "fields.npz",
        displacement_history=np.asarray(result.displacement_history),
        increment_history=np.asarray(result.equivalent_plastic_increment_history),
    )
    report = {
        "schema_version": 1,
        "rank": args.rank,
        "prior_weight": args.prior_weight,
        "converged": result.completed,
        "accepted_increments": len(result.increments),
        "failed_increment": result.failed_increment,
        "equilibrium_rms": [item.equilibrium_rms for item in result.increments],
        "constitutive_rejections": [
            item.constitutive_rejections for item in result.increments
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
