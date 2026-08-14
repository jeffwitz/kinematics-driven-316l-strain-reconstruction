#!/usr/bin/env python3
"""Diagnose Ludwik baseline and local Delta-p observability on a P43 oracle run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from fem_inhouse.core.driven_j2 import DrivenJ2PlaneStressBatch
from fem_inhouse.identification.dic_whitening import DICSpectralWhitener
from fem_inhouse.measurement import image_flow_to_canonical
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior
from fem_inhouse.workflows.experimental_mechanical_oracle import (
    evaluate_experimental_mechanical_oracle,
)

ROOT = Path(__file__).resolve().parents[1]
NOISE = (
    ROOT
    / "validation/reference_data/dic_uncertainty_propagation_p0043_v1"
    / "centred_repeat_flow_pixels.npy"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fields",
        type=Path,
        default=ROOT
        / "validation/_generated/performance/experimental_oracle_p43_m20/fields.npz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "validation/_generated/performance/experimental_oracle_p43_m20/report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "validation/_generated/performance/experimental_oracle_p43_m20/identifiability.json",
    )
    parser.add_argument("--directions", type=int, default=12)
    args = parser.parse_args()

    fields = np.load(args.fields, allow_pickle=False)
    measured = fields["measured_displacement_history"]
    oracle = fields["oracle_displacement_history"]
    ludwik = fields["ludwik_increment_history"]
    oracle_increment = fields["oracle_increment_history"]
    nx, ny = (value - 1 for value in measured.shape[1:3])
    pixel_size = 0.00184
    grid = StructuredGrid2D(nx, ny, nx * pixel_size, ny * pixel_size)
    kinematics = TwoSubcellDiagnostic2D(grid)
    baseline_material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    for index in range(1, measured.shape[0]):
        trial = baseline_material.evaluate(
            kinematics.strain_samples(measured[index]).reshape(-1, 3),
            ludwik[index - 1].reshape(-1),
            time_increment=1.0,
            consistent_tangent=True,
        )
        baseline_material.commit()
    baseline_stress = trial.stress_in_plane_mpa.reshape(nx, ny, 2, 3)
    baseline_residual = kinematics.divergence_from_sample_stress(baseline_stress)
    material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    for index in range(1, oracle.shape[0]):
        trial = material.evaluate(
            kinematics.strain_samples(oracle[index]).reshape(-1, 3),
            oracle_increment[index - 1].reshape(-1),
            time_increment=1.0,
            consistent_tangent=True,
        )
        material.commit()
    oracle_linearisation = evaluate_experimental_mechanical_oracle(
        material,
        kinematics,
        oracle[-1],
        oracle_increment[-1],
        time_increment=1.0,
    )
    zero_plastic = np.zeros_like(oracle_increment[-1])
    unknown_count = (nx - 1) * (ny - 1) * 2

    def stiffness_action(vector: np.ndarray) -> np.ndarray:
        displacement = unpack_interior(vector, grid)
        return pack_interior(
            oracle_linearisation.jacobian_action(displacement, zero_plastic)
        )

    stiffness = LinearOperator(
        (unknown_count, unknown_count), matvec=stiffness_action, dtype=np.float64
    )
    noise = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    canonical_noise = image_flow_to_canonical(
        np.asarray(noise[:512, :512]), pixel_size_mm=pixel_size
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
    rng = np.random.default_rng(20260814)
    directional_rows: list[dict[str, float | int]] = []
    for index in range(args.directions):
        direction = rng.normal(size=zero_plastic.shape)
        direction /= np.linalg.norm(direction)
        rhs = pack_interior(
            oracle_linearisation.jacobian_action(
                np.zeros_like(oracle[-1]), direction
            )
        )
        displacement_vector, info = gmres(
            stiffness,
            -rhs,
            rtol=1.0e-7,
            atol=0.0,
            maxiter=2000,
        )
        displacement = unpack_interior(displacement_vector, grid)
        directional_rows.append(
            {
                "direction": index,
                "gmres_info": int(info),
                "whitened_displacement_norm_per_unit_delta_p": float(
                    np.linalg.norm(whitener.apply(displacement))
                ),
                "displacement_rms_mm_per_unit_delta_p": float(
                    np.sqrt(np.mean(displacement**2))
                ),
                "displacement_inf_mm_per_unit_delta_p": float(
                    np.max(np.abs(displacement))
                ),
            }
        )
    report = json.loads(args.report.read_text(encoding="utf-8"))
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "oracle_report": str(args.report),
        "baseline_ludwik": {
            "equilibrium_rms_mpa_mm": float(
                np.sqrt(np.mean(baseline_residual[1:-1, 1:-1] ** 2))
            ),
            "equilibrium_inf_mpa_mm": float(
                np.max(np.abs(baseline_residual[1:-1, 1:-1]))
            ),
            "dic_is_the_baseline_displacement": True,
        },
        "oracle_final": report.get("field_comparison_final_state", {}),
        "typical_oracle_delta_p_l2": float(
            np.linalg.norm(oracle_increment[-1])
        ),
        "directional_sensitivity": directional_rows,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
