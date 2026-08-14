#!/usr/bin/env python3
"""Qualify mechanically/DIC-observable perturbations of the J2 flow direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fem_inhouse.core.driven_j2 import DrivenJ2PlaneStressBatch
from fem_inhouse.identification.dic_whitening import DICSpectralTransfer, DICSpectralWhitener
from fem_inhouse.identification.plastic_observability import (
    DirectionMetric,
    DirectionObservabilityOperator,
    PlasticObservabilityState,
)
from fem_inhouse.measurement import image_flow_to_canonical
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.workflows.experimental_mechanical_oracle import (
    evaluate_experimental_mechanical_oracle,
)

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ROOT / "validation/_generated/performance/experimental_oracle_p43_m20/fields.npz"
NOISE = (
    ROOT / "validation/reference_data/dic_uncertainty_propagation_p0043_v1"
    / "centred_repeat_flow_pixels.npy"
)
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--state-count", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fields = np.load(FIELDS, allow_pickle=False)
    displacement = np.asarray(fields["oracle_displacement_history"])
    increments = np.asarray(fields["oracle_increment_history"])
    node_shape = displacement.shape[1:3]
    grid = StructuredGrid2D(
        node_shape[0] - 1,
        node_shape[1] - 1,
        0.00184 * (node_shape[0] - 1),
        0.00184 * (node_shape[1] - 1),
    )
    kinematics = TwoSubcellDiagnostic2D(grid)
    state_indices = np.rint(
        np.linspace(0, increments.shape[0] - 1, args.state_count)
    ).astype(int)
    material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    selected = {}
    for index in range(increments.shape[0]):
        linearisation = evaluate_experimental_mechanical_oracle(
            material, kinematics, displacement[index + 1], increments[index], time_increment=1.0
        )
        if index in state_indices:
            selected[index] = linearisation
        material.commit()

    noise = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    canonical_noise = image_flow_to_canonical(np.asarray(noise[:512, :512]), pixel_size_mm=0.00184)
    support = np.ones((*grid.node_shape, 2), dtype=np.float64)
    support[[0, -1], :, :] = 0.0
    support[:, [0, -1], :] = 0.0
    whitener = DICSpectralWhitener.from_stationary_noise_field(
        canonical_noise, target_shape=grid.node_shape, sample_count=256, seed=42,
        remove_spatial_mean=False, support_mask=support
    )
    operator = DirectionObservabilityOperator(
        tuple(PlasticObservabilityState(selected[index]) for index in state_indices),
        grid, whitener, transfer=DICSpectralTransfer.from_sinusoidal_csv(TRANSFER),
        gmres_rtol=1.0e-9, gmres_maxiter=2000,
    )
    eigenvalues, modes = operator.generalized_modes(
        min(args.rank, operator.direction_size - 1),
        metric=DirectionMetric(operator.states),
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "basis.npz",
        basis=modes,
        eigenvalues=eigenvalues,
        state_indices=state_indices,
    )
    figure, axes = plt.subplots(
        len(eigenvalues), 5, squeeze=False, figsize=(20, 3 * len(eigenvalues))
    )
    for j in range(len(eigenvalues)):
        mode = modes[:, j].reshape(operator.direction_shape)
        components = mode.mean(axis=-2)
        magnitude = np.linalg.norm(components, axis=-1)
        induced = operator.sensitivity(operator.states[0], mode)
        measured = operator.transfer.apply(induced)
        whitened = operator.whitener.apply(measured)
        axes[j, 0].imshow(magnitude.T, origin="lower", cmap="coolwarm")
        axes[j, 0].set_title(f"|direction mode {j + 1}|")
        axes[j, 1].imshow(components[..., 0].T, origin="lower", cmap="coolwarm")
        axes[j, 1].set_title("delta n11")
        axes[j, 2].imshow(components[..., 1].T, origin="lower", cmap="coolwarm")
        axes[j, 2].set_title("delta n22")
        axes[j, 3].imshow(components[..., 2].T, origin="lower", cmap="coolwarm")
        axes[j, 3].set_title("delta n12")
        axes[j, 4].imshow(
            np.linalg.norm(whitened, axis=-1).T, origin="lower", cmap="magma"
        )
        axes[j, 4].set_title("|W_D M_D S_n mode|")
        for axis in axes[j]:
            axis.set_aspect("equal")
    figure.tight_layout()
    figure.savefig(output / "mode_fields.png", dpi=180)
    plt.close(figure)
    report = {
        "schema_version": 1,
        "state_count": len(state_indices),
        "state_indices": state_indices.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "adjoint_checks": operator.adjoint_errors(),
        "direction_shape": list(operator.direction_shape),
        "interpretation": "instantaneous mechanically observable J2 flow-direction perturbations",
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
