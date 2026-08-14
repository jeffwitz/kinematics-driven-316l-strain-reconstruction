#!/usr/bin/env python3
"""Separate the load-proportional residual from the one that appears with yield.

The residual to a homogeneous isotropic elastic model contains at least two
things. Elastic heterogeneity -- this crop spans many anisotropic grains -- is
proportional to the load and keeps a fixed spatial pattern. Plasticity appears
only after yield and has its own pattern. A free tensor eigenstrain reproduces
both, so no regularisation in the plastic space can tell them apart.

They separate on the data instead, without EBSD and without a scaling law. The
normalised residual fields of the early states are mutually parallel to within
a few per cent, and progressively rotate away later, so the early states span
the heterogeneity subspace directly. Removing that subspace from the late
residuals leaves what elasticity, however heterogeneous, cannot produce at a
fixed pattern.

The assumption is weak and stated plainly: the heterogeneity contribution keeps
a constant *shape* over the path, whatever its amplitude. It does not assume
proportionality, isotropy, or any grain information. It does assume the early
states are free of plasticity, which the null test at state 1 and the pattern
stability up to state 20 both support.

What survives is a lower bound on the plastic content: any plastic component
that happens to lie inside the early subspace is removed with it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import svds

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
    parser.add_argument("--elastic-states", nargs=2, type=int, default=(3, 20))
    parser.add_argument("--subspace-rank", type=int, default=3)
    parser.add_argument("--modes", type=int, default=20)
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
    support = np.ones((*grid.node_shape, 2), dtype=np.float64)
    support[[0, -1], :, :] = 0.0
    support[:, [0, -1], :] = 0.0
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
    early = residual[first : last + 1]
    basis, spectrum, _ = np.linalg.svd(early.T, full_matrices=False)
    rank = arguments.subspace_rank
    heterogeneity = basis[:, :rank]
    captured = float((spectrum[:rank] ** 2).sum() / (spectrum**2).sum())

    corrected = residual - (residual @ heterogeneity) @ heterogeneity.T

    left, singular, _ = svds(operator.as_linear_operator(), k=arguments.modes, tol=0)
    order = np.argsort(singular)[::-1]
    left, singular = left[:, order], singular[order]
    raw_coefficients = left.T @ residual.T
    corrected_coefficients = left.T @ corrected.T

    interior = 2 * (grid.node_shape[0] - 2) * (grid.node_shape[1] - 2)
    noise_norm = float(np.sqrt(interior))

    output = {
        "schema_version": 1,
        "pixels": pixels,
        "origin_nodes": [x0, y0],
        "elastic_states": [first, last],
        "subspace_rank": rank,
        "early_variance_captured": captured,
        "pure_noise_norm": noise_norm,
        "singular_values": singular.tolist(),
        "raw_norm_over_noise": (np.linalg.norm(residual, axis=1) / noise_norm).tolist(),
        "corrected_norm_over_noise": (np.linalg.norm(corrected, axis=1) / noise_norm).tolist(),
        "raw_coefficients_in_noise_sigma": raw_coefficients.tolist(),
        "corrected_coefficients_in_noise_sigma": corrected_coefficients.tolist(),
        "inferred_equivalent_eigenstrain_final_state": (
            corrected_coefficients[:, -1] / singular
        ).tolist(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", "utf-8")

    print(
        f"pixels={pixels}  elastic subspace from states {first}-{last}, rank {rank}, "
        f"capturing {captured * 100:.2f} % of their variance"
    )
    print("\n state | raw norm/noise | corrected | max|c| raw | max|c| corrected")
    for state in (1, 5, 10, 20, 25, 30, 35, 40):
        print(
            f" {state:5d} | {np.linalg.norm(residual[state]) / noise_norm:14.3f} | "
            f"{np.linalg.norm(corrected[state]) / noise_norm:9.3f} | "
            f"{np.abs(raw_coefficients[:, state]).max():10.2f} | "
            f"{np.abs(corrected_coefficients[:, state]).max():16.2f}"
        )
    print("\n  j | corrected c(final) | equivalent eigenstrain")
    for index in range(min(8, arguments.modes)):
        print(
            f"{index + 1:3d} | {corrected_coefficients[index, -1]:18.2f} | "
            f"{corrected_coefficients[index, -1] / singular[index]:+.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
