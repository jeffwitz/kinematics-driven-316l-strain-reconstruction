#!/usr/bin/env python3
"""What do the observable plastic modes look like, and where does the signal live?

A mode can be mathematically observable and still be an artefact of the
boundary conditions. Before anything is reconstructed in this basis, each mode
is described: how much of it sits away from the edges, how it splits between
the three tensor components, and at what spatial scale.

The reconstruction is then confronted with the experiment. Combining the modes
with the coefficients left after the elastic-heterogeneity subspace is removed
gives a plastic field, and its equivalent measure
``p_eq = sqrt(z^T M^-1 z)`` is compared with the equivalent strain the DIC
shows directly. If the detected signal is plasticity, it should live where the
measured field localises. If it lives on the edges instead, it is the boundary
condition talking.

That comparison is not circular: the modes come from the mechanics and the
measurement chain, never from the data, and the DIC equivalent strain is a
direct kinematic reading that enters no part of their construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import svds

from fem_inhouse.core.constitutive import PLANE_STRESS_VON_MISES_METRIC, von_mises
from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.identification.dic_whitening import DICSpectralTransfer, DICSpectralWhitener
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.measurement import image_flow_to_canonical
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior

ROOT = Path(__file__).resolve().parents[1]
NOISE = (
    ROOT
    / "validation/reference_data/dic_uncertainty_propagation_p0043_v1"
    / "centred_repeat_flow_pixels.npy"
)
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
HISTORY = (
    ROOT
    / "validation/reference_data/dic_multistep_history_p0043_repaired_v1"
    / "repaired_history_mm.npy"
)
PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", nargs=2, type=int, default=(1610, 1075))
    parser.add_argument("--pixels", type=int, default=100)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--elastic-states", nargs=2, type=int, default=(3, 20))
    parser.add_argument("--subspace-rank", type=int, default=3)
    parser.add_argument("--border-fraction", type=float, default=0.15)
    parser.add_argument(
        "--observation-border",
        type=int,
        default=1,
        help=(
            "nodes masked out of the observation on each side. The default of 1 "
            "removes only the Dirichlet ring. A wider band asks what the BULK can "
            "show: the crop boundary is an artefact of choosing a window inside a "
            "larger measured field, and a near-edge eigenstrain has the strongest "
            "lever on the interior displacement, so leaving the band observed lets "
            "the operator rank boundary directions first."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    pixels = arguments.pixels
    x0, y0 = arguments.origin
    report = json.loads((HISTORY.with_name("report.json")).read_text(encoding="utf-8"))
    bounds = list(map(int, report["solve_bounds"]))
    source = np.load(HISTORY, mmap_mode="r", allow_pickle=False)
    history = np.asarray(
        source[
            :,
            x0 - bounds[0] : x0 + pixels - bounds[0] + 1,
            y0 - bounds[2] : y0 + pixels - bounds[2] + 1,
            :,
        ],
        dtype=np.float64,
    )
    history = history - history[0]

    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    mask_width = max(1, arguments.observation_border)
    support = np.zeros((*grid.node_shape, 2), dtype=np.float64)
    support[mask_width:-mask_width, mask_width:-mask_width, :] = 1.0
    noise = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    canonical = image_flow_to_canonical(np.asarray(noise[:512, :512]), pixel_size_mm=PIXEL_SIZE_MM)
    whitener = DICSpectralWhitener.from_stationary_noise_field(
        canonical,
        target_shape=grid.node_shape,
        sample_count=256,
        seed=42,
        remove_spatial_mean=False,
        support_mask=support,
    )
    transfer = DICSpectralTransfer.from_sinusoidal_csv(TRANSFER)
    operator = TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=transfer,
        whitener=whitener,
    )
    left, singular, right_transposed = svds(
        operator.as_linear_operator(), k=arguments.modes, tol=0
    )
    order = np.argsort(singular)[::-1]
    left, singular = left[:, order], singular[order]
    right = right_transposed[order].T

    elasticity = plane_stress_elasticity(YOUNG_MPA, POISSON)
    weight = float(operator.kinematics.sample_quadrature_weight)
    residuals = []
    for state in range(history.shape[0]):
        measured = history[state]
        strain = np.asarray(operator.kinematics.strain(measured)).reshape(-1, 3)
        forcing = (
            -pack_interior(
                operator.kinematics.divergence_from_sample_stress(
                    (strain @ elasticity).reshape((pixels, pixels, 2, 3))
                )
            )
            / weight
        )
        elastic = measured.copy()
        elastic[1:-1, 1:-1, :] -= operator.solve_stiffness(forcing).reshape(
            grid.node_shape[0] - 2, grid.node_shape[1] - 2, 2
        )
        residuals.append(
            np.asarray(whitener.apply(measured - transfer.apply(elastic))).reshape(-1)
        )
    residual = np.asarray(residuals)
    first, last = arguments.elastic_states
    basis, _, _ = np.linalg.svd(residual[first : last + 1].T, full_matrices=False)
    heterogeneity = basis[:, : arguments.subspace_rank]
    corrected = residual - (residual @ heterogeneity) @ heterogeneity.T
    coefficients = (left.T @ corrected.T) / singular[:, None]

    gauge = np.linalg.inv(PLANE_STRESS_VON_MISES_METRIC)
    border = max(1, round(arguments.border_fraction * pixels))
    interior_mask = np.zeros((pixels, pixels), dtype=bool)
    interior_mask[border:-border, border:-border] = True

    def anatomy(field: np.ndarray) -> dict[str, float]:
        blocks = field.reshape(pixels, pixels, 2, 3)
        equivalent = np.sqrt(
            np.maximum(np.einsum("xyci,ij,xycj->xyc", blocks, gauge, blocks), 0.0)
        ).mean(axis=2)
        energy = equivalent**2
        total = float(energy.sum())
        component = (blocks**2).sum(axis=(0, 1, 2))
        peak = float(equivalent.max())
        return {
            "interior_fraction": float(energy[interior_mask].sum() / total) if total > 0 else 0.0,
            "component_shares": (component / component.sum()).tolist(),
            "participation_ratio": float(total**2 / float((energy**2).sum() * energy.size))
            if total > 0
            else 0.0,
            "peak_over_rms": peak / float(np.sqrt(energy.mean())) if total > 0 else 0.0,
        }

    interior_area_fraction = float(interior_mask.sum() / interior_mask.size)
    modes = [
        {
            "mode": index + 1,
            "singular_value": float(singular[index]),
            "coefficient_final_state": float(left[:, index] @ corrected[-1]),
            **anatomy(operator.plastic_mode(right[:, index])),
        }
        for index in range(arguments.modes)
    ]

    reconstruction = sum(
        coefficients[index, -1] * operator.plastic_mode(right[:, index])
        for index in range(arguments.modes)
    )
    blocks = np.asarray(reconstruction).reshape(pixels, pixels, 2, 3)
    reconstructed_equivalent = np.sqrt(
        np.maximum(np.einsum("xyci,ij,xycj->xyc", blocks, gauge, blocks), 0.0)
    ).mean(axis=2)

    measured_strain = np.asarray(operator.kinematics.strain(history[-1])).reshape(
        pixels, pixels, 2, 3
    )
    measured_equivalent = von_mises(measured_strain).mean(axis=2)

    flat_model = reconstructed_equivalent.reshape(-1)
    flat_data = measured_equivalent.reshape(-1)
    correlation = float(np.corrcoef(flat_model, flat_data)[0, 1])
    top = flat_data >= np.quantile(flat_data, 0.9)
    localisation_capture = float(flat_model[top].sum() / flat_model.sum())

    output = {
        "schema_version": 1,
        "pixels": pixels,
        "origin_nodes": [x0, y0],
        "border_pixels": border,
        "observation_border_nodes": mask_width,
        "interior_area_fraction": interior_area_fraction,
        "modes": modes,
        "reconstruction": {
            "peak_equivalent_eigenstrain": float(reconstructed_equivalent.max()),
            "rms_equivalent_eigenstrain": float(np.sqrt((reconstructed_equivalent**2).mean())),
            "correlation_with_dic_equivalent_strain": correlation,
            "share_inside_the_dic_top_decile": localisation_capture,
            "share_expected_if_unstructured": 0.1,
            "measured_peak_equivalent_strain": float(measured_equivalent.max()),
        },
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "report.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        arguments.output / "modes.npz",
        modes=np.column_stack(
            [operator.plastic_mode(right[:, index]) for index in range(arguments.modes)]
        ),
        singular_values=singular,
        coefficients=coefficients,
        reconstructed_equivalent=reconstructed_equivalent,
        measured_equivalent=measured_equivalent,
    )

    print(f"pixels={pixels}  border={border} px  interior area = {interior_area_fraction:.2f}")
    print("\n  j |    sigma_j | interior | e11 / e22 / g12 shares | particip | peak/rms")
    for entry in modes[:12]:
        shares = " / ".join(f"{value:.2f}" for value in entry["component_shares"])
        print(
            f"{entry['mode']:3d} | {entry['singular_value']:10.3e} | "
            f"{entry['interior_fraction']:8.3f} | {shares:>22} | "
            f"{entry['participation_ratio']:8.3f} | {entry['peak_over_rms']:8.2f}"
        )
    reconstructed = output["reconstruction"]
    print(
        f"\nreconstruction at the final state: peak p_eq = "
        f"{reconstructed['peak_equivalent_eigenstrain']:.3e}, "
        f"RMS = {reconstructed['rms_equivalent_eigenstrain']:.3e}"
    )
    print(
        f"correlation with the DIC equivalent strain : "
        f"{reconstructed['correlation_with_dic_equivalent_strain']:+.3f}"
    )
    print(
        f"share of the reconstruction inside the DIC top decile : "
        f"{reconstructed['share_inside_the_dic_top_decile']:.3f}  (0.10 if unstructured)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
