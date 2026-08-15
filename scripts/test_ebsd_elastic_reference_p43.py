#!/usr/bin/env python3
"""How much of the mechanical defect is just crystallographic elasticity?

The defect of the measured field is currently taken against a homogeneous
isotropic elastic reference. A polycrystal is not that, and a free eigenstrain
reproduces an elastic inclusion exactly, so any elastic heterogeneity lands in
the defect and is indistinguishable from plasticity. This replaces the
reference elasticity with the real crystallographic one and changes nothing
else: same gauge `H = M^-1`, same measurement chain, same boundary conditions,
same crop, same diagnostics.

No plasticity model, no crystal plasticity, no tangent, no parameter fitting.
The only question is what fraction of the defect disappears when the measured
microstructure enters the elastic reference.

Three variants, because adding anisotropy is not the same as adding the *right*
anisotropy:

* `isotropic`  -- the current reference;
* `ebsd`       -- the recorded orientation map, in place;
* `shuffled`   -- the same orientations permuted across the map.

`ebsd` against `shuffled` is the discriminating pair. If only the map in its
recorded arrangement removes the defect, then the spatial structure of the real
microstructure is what explained it, not the mere presence of anisotropy. The
control needs no new data.

Under pure Dirichlet conditions a uniform scaling of the stiffness leaves the
displacement unchanged, so no absolute modulus can be identified without a
force measurement. That is not what is asked here: the kinematics responds to
the anisotropy ratio, its orientation and its spatial arrangement, all of which
are fixed by the EBSD map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import svds

from fem_inhouse.identification.crystal_plane_stress_elasticity import (
    cubic_stiffness_from_engineering_constants,
    rotated_plane_stress_stiffness,
)
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
EBSD = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")
PIXEL_SIZE_MM = 0.00184
#: Crystal-frame constants of the FCC law this repository already declares, in
#: `mfront/Fcc316LForestRubinSrix.mfront`, so the elastic reference cannot drift
#: away from the constitutive one.
CRYSTAL_YOUNG_MPA = 99950.31055900622
CRYSTAL_POISSON = 0.3881987577639752
CRYSTAL_SHEAR_MPA = 122000.0
ISOTROPIC_YOUNG_MPA = 205_000.0
ISOTROPIC_POISSON = 0.30


def _load_orientations(origin: tuple[int, int], pixels: int) -> tuple[np.ndarray, dict[str, str]]:
    import h5py

    x0, y0 = origin
    with h5py.File(EBSD, "r") as handle:
        angles = np.stack(
            [
                np.asarray(
                    handle[f"orientation/{name}"][x0 : x0 + pixels, y0 : y0 + pixels], dtype=float
                )
                for name in ("phi1", "Phi", "phi2")
            ],
            axis=-1,
        )
    digest = hashlib.sha256(np.ascontiguousarray(angles).tobytes()).hexdigest()
    return angles, {"source": str(EBSD), "sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", nargs=2, type=int, default=(1610, 1075))
    parser.add_argument("--pixels", type=int, default=100)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--elastic-states", nargs=2, type=int, default=(3, 20))
    parser.add_argument("--subspace-rank", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260815)
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

    angles, provenance = _load_orientations((x0, y0), pixels)
    crystal = cubic_stiffness_from_engineering_constants(
        CRYSTAL_YOUNG_MPA, CRYSTAL_POISSON, CRYSTAL_SHEAR_MPA
    )
    flat_angles = angles.reshape(-1, 3)
    generator = np.random.default_rng(arguments.seed)
    permutation = generator.permutation(flat_angles.shape[0])

    def per_point(pixel_angles: np.ndarray) -> np.ndarray:
        """Both sub-cells of a pixel share its orientation."""

        stiffness = rotated_plane_stress_stiffness(crystal, pixel_angles)
        return np.repeat(stiffness, 2, axis=0)

    variants = {
        "isotropic": None,
        "ebsd": per_point(flat_angles),
        "shuffled": per_point(flat_angles[permutation]),
    }

    results: dict[str, dict[str, object]] = {}
    residuals: dict[str, np.ndarray] = {}
    for name, point_elasticity in variants.items():
        operator = TensorPlasticObservabilityOperator.build(
            grid,
            young_modulus_mpa=ISOTROPIC_YOUNG_MPA,
            poisson_ratio=ISOTROPIC_POISSON,
            transfer=transfer,
            whitener=whitener,
            point_elasticity=point_elasticity,
        )
        weight = float(operator.kinematics.sample_quadrature_weight)
        collected = []
        for state in range(history.shape[0]):
            measured = history[state]
            strain = np.asarray(operator.kinematics.strain(measured)).reshape(-1, 3)
            stress = np.einsum("pi,pij->pj", strain, operator.elasticity)
            forcing = (
                -pack_interior(
                    operator.kinematics.divergence_from_sample_stress(
                        stress.reshape((pixels, pixels, 2, 3))
                    )
                )
                / weight
            )
            elastic = measured.copy()
            elastic[1:-1, 1:-1, :] -= operator.solve_stiffness(forcing).reshape(
                grid.node_shape[0] - 2, grid.node_shape[1] - 2, 2
            )
            collected.append(
                np.asarray(whitener.apply(measured - transfer.apply(elastic))).reshape(-1)
            )
        residual = np.asarray(collected)
        residuals[name] = residual

        left, singular, _ = svds(operator.as_linear_operator(), k=arguments.modes, tol=0)
        order = np.argsort(singular)[::-1]
        left, singular = left[:, order], singular[order]
        coefficients = left.T @ residual.T
        first, last = arguments.elastic_states
        _, spectrum, _ = np.linalg.svd(residual[first : last + 1].T, full_matrices=False)
        interior = 2 * (grid.node_shape[0] - 2) * (grid.node_shape[1] - 2)
        noise_norm = float(np.sqrt(interior))
        results[name] = {
            "norm_over_noise": (np.linalg.norm(residual, axis=1) / noise_norm).tolist(),
            "maximum_coefficient_by_state": np.abs(coefficients).max(axis=0).tolist(),
            "early_variance_in_rank_three": float(
                (spectrum[: arguments.subspace_rank] ** 2).sum() / (spectrum**2).sum()
            ),
            "leading_singular_values": singular[:6].tolist(),
        }
        print(
            f"{name:10s}: state1 {results[name]['norm_over_noise'][1]:7.3f}  "
            f"state20 {results[name]['norm_over_noise'][20]:7.3f}  "
            f"state40 {results[name]['norm_over_noise'][40]:8.3f}  "
            f"early rank-3 variance {results[name]['early_variance_in_rank_three']:.4f}"
        )

    # Does the EBSD correction live in the subspace that was fitted empirically
    # on the early isotropic states? If it does, the "early heterogeneity"
    # subspace has found its physical origin.
    first, last = arguments.elastic_states
    basis, _, _ = np.linalg.svd(residuals["isotropic"][first : last + 1].T, full_matrices=False)
    early = basis[:, : arguments.subspace_rank]
    captured = {}
    for name in ("ebsd", "shuffled"):
        correction = residuals["isotropic"] - residuals[name]
        projected = (correction @ early) @ early.T
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(
                np.linalg.norm(correction, axis=1) > 0.0,
                (np.linalg.norm(projected, axis=1) / np.linalg.norm(correction, axis=1)) ** 2,
                0.0,
            )
        captured[name] = {
            "share_in_the_early_subspace_by_state": share.tolist(),
            "share_at_state_twenty": float(share[20]),
            "share_at_final_state": float(share[-1]),
        }
        print(
            f"correction {name:8s} inside the early rank-3 subspace: "
            f"state20 {share[20]:.4f}   state40 {share[-1]:.4f}"
        )

    output = {
        "schema_version": 1,
        "pixels": pixels,
        "origin_nodes": [x0, y0],
        "orientation_provenance": provenance,
        "crystal_constants_mpa": {
            "young": CRYSTAL_YOUNG_MPA,
            "poisson": CRYSTAL_POISSON,
            "shear": CRYSTAL_SHEAR_MPA,
            "zener_anisotropy": float(
                2 * crystal[3, 3] / (crystal[0, 0] - crystal[0, 1])
            ),
        },
        "elastic_states": [first, last],
        "subspace_rank": arguments.subspace_rank,
        "variants": results,
        "ebsd_correction_in_the_early_subspace": captured,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", "utf-8")

    print("\n state | isotropic | ebsd  | shuffled   (residual norm over pure noise)")
    for state in (1, 5, 10, 20, 25, 30, 35, 40):
        print(
            f" {state:5d} | {results['isotropic']['norm_over_noise'][state]:9.3f} | "
            f"{results['ebsd']['norm_over_noise'][state]:5.3f} | "
            f"{results['shuffled']['norm_over_noise'][state]:8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
