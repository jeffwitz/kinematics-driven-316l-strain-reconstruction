#!/usr/bin/env python3
"""Build and qualify observability modes on the archived P43 M20 oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fem_inhouse.core.driven_j2 import DrivenJ2PlaneStressBatch
from fem_inhouse.identification.dic_whitening import (
    DICSpectralTransfer,
    DICSpectralWhitener,
)
from fem_inhouse.identification.plastic_observability import (
    PlasticMetric,
    PlasticObservabilityOperator,
    PlasticObservabilityState,
)
from fem_inhouse.measurement import image_flow_to_canonical
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.workflows.experimental_mechanical_oracle import (
    ExperimentalMechanicalOracleLinearisation,
    evaluate_experimental_mechanical_oracle,
)

ROOT = Path(__file__).resolve().parents[1]
NOISE = (
    ROOT
    / "validation/reference_data/dic_uncertainty_propagation_p0043_v1"
    / "centred_repeat_flow_pixels.npy"
)
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
DEFAULT_FIELDS = (
    ROOT / "validation/_generated/performance/experimental_oracle_p43_m20/fields.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "validation/_generated/performance/experimental_oracle_p43_m20/observability"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=Path, default=DEFAULT_FIELDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--state-count", type=int, default=5)
    parser.add_argument("--state-indices", nargs="+", type=int)
    parser.add_argument("--spatial-weight", type=float, default=0.0)
    parser.add_argument("--reference-scale", type=float)
    args = parser.parse_args()

    fields = np.load(args.fields, allow_pickle=False)
    displacement_history = np.asarray(fields["oracle_displacement_history"])
    increment_history = np.asarray(fields["oracle_increment_history"])
    ludwik_history = np.asarray(fields["ludwik_increment_history"])
    node_shape = displacement_history.shape[1:3]
    nx, ny = node_shape[0] - 1, node_shape[1] - 1
    grid = StructuredGrid2D(nx, ny, 0.00184 * nx, 0.00184 * ny)
    kinematics = TwoSubcellDiagnostic2D(grid)
    if increment_history.shape[0] != displacement_history.shape[0] - 1:
        raise ValueError("oracle history and increments have incompatible lengths")
    if args.state_indices is None:
        state_count = min(args.state_count, increment_history.shape[0])
        state_indices = np.rint(
            np.linspace(0, increment_history.shape[0] - 1, state_count)
        ).astype(int)
    else:
        state_indices = np.asarray(args.state_indices, dtype=int)
    if np.any(state_indices < 0) or np.any(state_indices >= increment_history.shape[0]):
        raise ValueError("state indices must refer to archived increments")
    reference_scale = args.reference_scale
    if reference_scale is None:
        active_ludwik = ludwik_history[ludwik_history > 0.0]
        reference_scale = float(np.sqrt(np.mean(active_ludwik**2)))

    material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    selected: dict[int, ExperimentalMechanicalOracleLinearisation] = {}
    for index in range(increment_history.shape[0]):
        linearisation = evaluate_experimental_mechanical_oracle(
            material,
            kinematics,
            displacement_history[index + 1],
            increment_history[index],
            time_increment=1.0,
        )
        if index in state_indices:
            selected[index] = linearisation
        material.commit()
    if len(selected) != len(state_indices):
        raise RuntimeError("failed to construct all selected states")

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
    operator = PlasticObservabilityOperator(
        tuple(PlasticObservabilityState(selected[index]) for index in state_indices),
        grid,
        whitener,
        transfer=DICSpectralTransfer.from_sinusoidal_csv(TRANSFER),
        gmres_rtol=1.0e-9,
        gmres_maxiter=2000,
    )
    metric = PlasticMetric(
        spatial_weight=args.spatial_weight,
        reference_scale=reference_scale,
    )
    eigenvalues, modes = operator.generalized_modes(args.rank, metric=metric)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "basis.npz",
        basis=modes,
        eigenvalues=eigenvalues,
        state_indices=state_indices,
    )
    mode_count = min(args.rank, 4)
    figure, axes = plt.subplots(mode_count, 3, squeeze=False, figsize=(12, 3 * mode_count))
    spectral_rows: list[dict[str, float | int]] = []
    state = PlasticObservabilityState(selected[int(state_indices[0])])
    for mode_index in range(mode_count):
        mode = modes[:, mode_index].reshape(operator.plastic_shape)
        plastic_map = np.mean(mode, axis=-1)
        induced = operator.sensitivity(state, mode)
        whitened = operator.whitener.apply(induced)
        axes[mode_index, 0].imshow(plastic_map.T, origin="lower", cmap="coolwarm")
        axes[mode_index, 0].set_title(f"phi {mode_index + 1}")
        axes[mode_index, 1].imshow(
            np.linalg.norm(induced, axis=-1).T, origin="lower", cmap="viridis"
        )
        axes[mode_index, 1].set_title("|S_p phi|")
        axes[mode_index, 2].imshow(
            np.linalg.norm(whitened, axis=-1).T, origin="lower", cmap="magma"
        )
        axes[mode_index, 2].set_title("|W_D M_D S_p phi|")
        for axis in axes[mode_index]:
            axis.set_aspect("equal")
        spectrum = np.abs(np.fft.fftn(plastic_map, norm="ortho")) ** 2
        nyquist_x = spectrum.shape[0] // 2
        nyquist_y = spectrum.shape[1] // 2
        spectral_rows.append(
            {
                "mode": mode_index + 1,
                "dc_power_fraction": float(spectrum[0, 0] / np.sum(spectrum)),
                "nyquist_axis_power_fraction": float(
                    (np.sum(spectrum[nyquist_x, :]) + np.sum(spectrum[:, nyquist_y]))
                    / (2.0 * np.sum(spectrum))
                ),
            }
        )
    figure.tight_layout()
    figure.savefig(output / "mode_fields.png", dpi=180)
    plt.close(figure)
    (output / "mode_spectra.json").write_text(
        json.dumps(spectral_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "fields": str(args.fields),
        "state_indices": state_indices.tolist(),
        "rank": int(args.rank),
        "plastic_shape": list(operator.plastic_shape),
        "metric": {
            "type": "amplitude_plus_neighbour_differences",
            "amplitude_weight": 1.0,
            "spatial_weight": args.spatial_weight,
            "reference_scale": reference_scale,
        },
        "measurement_transfer": {
            "path": str(TRANSFER),
            "model": "isotropic average of horizontal and vertical sinusoidal gains",
        },
        "eigenvalues": eigenvalues.tolist(),
        "sqrt_eigenvalues": np.sqrt(np.maximum(eigenvalues, 0.0)).tolist(),
        "mode_spectra": spectral_rows,
        "adjoint_checks": operator.adjoint_errors(),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
