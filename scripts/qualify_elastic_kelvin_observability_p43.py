#!/usr/bin/env python3
"""Can an effective elastic heterogeneity produce the early mechanical defect?

The EBSD surface map does not explain it, but the specimen is millimetres thick
and the map is one section. What surface DIC sees is the effective operator of
everything underneath, so this looks for that effective heterogeneity directly.

The unknown is a dimensionless relative perturbation of the stiffness,
`C = C0^(1/2) (I + A) C0^(1/2)`, with `A` symmetric: six channels per pixel in
an orthonormal Kelvin basis. Crucially there is **one** field for all states,
so a single unknown must explain eighteen images at once -- much better
conditioned than one plastic increment per state.

Three results, and nothing else:

1. the observable spectrum over states 3-20;
2. the principal angles between what those modes can produce and the empirical
   rank-3 early subspace;
3. the same restricted to the two isotropic channels, for comparison.

If a handful of modes reproduce the early subspace, then the three patterns
seen from the first states are exactly what a weak effective stiffness
heterogeneity produces -- established without knowing any orientation at depth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import svds

from fem_inhouse.identification.dic_whitening import DICSpectralTransfer, DICSpectralWhitener
from fem_inhouse.identification.elastic_kelvin_observability import (
    CHANNEL_FAMILIES,
    ISOTROPIC_CHANNELS,
    ElasticKelvinObservabilityOperator,
)
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
    parser.add_argument("--origin", nargs=2, type=int, default=(1580, 1030))
    parser.add_argument("--pixels", type=int, default=100)
    parser.add_argument("--elastic-states", nargs=2, type=int, default=(3, 20))
    parser.add_argument("--subspace-rank", type=int, default=3)
    parser.add_argument("--modes", type=int, default=12)
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
    mechanics = TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=transfer,
        whitener=whitener,
    )
    weight = float(mechanics.kinematics.sample_quadrature_weight)

    first, last = arguments.elastic_states
    states = list(range(first, last + 1))
    references, residuals = [], []
    for state in range(history.shape[0]):
        measured = history[state]
        strain = np.asarray(mechanics.kinematics.strain(measured)).reshape(-1, 3)
        stress = np.einsum("pi,pij->pj", strain, mechanics.elasticity)
        forcing = (
            -pack_interior(
                mechanics.kinematics.divergence_from_sample_stress(
                    stress.reshape((pixels, pixels, 2, 3))
                )
            )
            / weight
        )
        elastic = measured.copy()
        elastic[1:-1, 1:-1, :] -= mechanics.solve_stiffness(forcing).reshape(
            grid.node_shape[0] - 2, grid.node_shape[1] - 2, 2
        )
        references.append(elastic)
        residuals.append(
            np.asarray(whitener.apply(measured - transfer.apply(elastic))).reshape(-1)
        )
    residual = np.asarray(residuals)
    early_basis, _, _ = np.linalg.svd(residual[first : last + 1].T, full_matrices=False)
    early = early_basis[:, : arguments.subspace_rank]

    observation_size = mechanics.observation_size
    results: dict[str, object] = {}
    variants = (
        ("all_six_channels", None),
        ("two_isotropic_channels", ISOTROPIC_CHANNELS),
    )
    for name, indices in variants:
        operator = ElasticKelvinObservabilityOperator.build(
            mechanics,
            [references[state] for state in states],
            young_modulus_mpa=YOUNG_MPA,
            poisson_ratio=POISSON,
            channel_indices=indices,
        )
        left, singular, right_transposed = svds(
            operator.as_linear_operator(), k=arguments.modes, tol=0
        )
        order = np.argsort(singular)[::-1]
        left, singular = left[:, order], singular[order]
        right = right_transposed[order]

        # What those modes can produce in a single state's observation space.
        produced = left.reshape(len(states), observation_size, arguments.modes)
        columns = np.concatenate([produced[index] for index in range(len(states))], axis=1)
        span, spectrum, _ = np.linalg.svd(columns, full_matrices=False)
        reachable = span[:, : arguments.subspace_rank]
        cosines = np.linalg.svd(reachable.T @ early, compute_uv=False)
        angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
        captured = float((np.linalg.norm(reachable.T @ early, axis=0) ** 2).mean())

        composition = None
        if indices is None:
            per_channel = right.reshape(arguments.modes, pixels * pixels, 6)
            energy = (per_channel**2).sum(axis=1)
            composition = (energy / energy.sum(axis=1, keepdims=True)).tolist()

        results[name] = {
            "channels": list(operator.channel_indices),
            "singular_values": singular.tolist(),
            "principal_angles_deg": angles.tolist(),
            "mean_squared_cosine_with_the_early_subspace": captured,
            "reachable_spectrum": spectrum[:6].tolist(),
            "channel_composition": composition,
        }
        print(
            f"{name:24s}: sigma1 {singular[0]:.4e}  sigma1/sigma{arguments.modes} "
            f"{singular[0] / singular[-1]:6.2f}  angles(deg) "
            f"{np.round(angles, 2).tolist()}  captured {captured:.4f}"
        )

    output = {
        "schema_version": 1,
        "pixels": pixels,
        "origin_nodes": [x0, y0],
        "elastic_states": [first, last],
        "subspace_rank": arguments.subspace_rank,
        "modes": arguments.modes,
        "channel_families": list(CHANNEL_FAMILIES),
        "results": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", "utf-8")

    composition = results["all_six_channels"]["channel_composition"]  # type: ignore[index]
    if composition is not None:
        print("\n  j | " + " | ".join(f"{name[:11]:>11}" for name in CHANNEL_FAMILIES))
        for index, row in enumerate(composition[:6]):
            print(f"{index + 1:3d} | " + " | ".join(f"{value:11.3f}" for value in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
