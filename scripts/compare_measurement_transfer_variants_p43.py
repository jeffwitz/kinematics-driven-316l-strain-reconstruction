#!/usr/bin/env python3
"""Is the DIC spatial transfer needed for the mechanical inversion at all?

The transfer and the whitener play different roles. `W_D` says a difference at
a frequency the DIC measures badly counts for less, and that is wanted. `M_D`
corrects the model for the finite spatial resolution of the instrument -- and it
is `M_D`, filtered through a periodic FFT, that was found to fabricate nine
sigma of edge artefact out of a plain affine ramp.

So `W_D` is kept throughout and only `M_D` varies:

* `identity`  -- the zero model, no extra assumption;
* `wrap_free` -- the repaired transfer, exact on affine fields;
* `periodic`  -- the historical one, for reference only.

`identity` against `wrap_free` is the question. If they tell the same story,
the transfer characterises the measurement chain but is not needed to invert
mechanics at these scales, and a good deal of machinery can go.

The criterion is per mode rather than global: for each observable mechanical
mode, how parallel are the two predicted observations, and how large is their
difference relative to one of them. A transfer that matters only at wavelengths
no mode uses is a transfer that can be dropped.

`identity` is not claimed to be *more* correct. The DIC field genuinely has a
finite resolution, so at very short wavelengths a fluctuation could be
attributed to the material that the instrument could never have transmitted.
It is the model that adds nothing beyond the data, which is a good place to
stand once a correction has been found to have been doing harm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import svds

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


class IdentityTransfer:
    """`M_D = I`: the model is compared with the data as measured."""

    def apply(self, values: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(values, dtype=np.float64)


class WrapFreeTransfer:
    """The repaired transfer, with its true adjoint.

    The forward action is `P + T (I - P)`, with `P` the orthogonal projector on
    affine fields and `T` the periodic filter. Its adjoint is `P + (I - P) T`,
    which is *not* the same operator: the projector and the filter do not
    commute. Reusing the forward action as its own adjoint would leave the
    partial SVD converging to a well-formed wrong answer, so the pair is built
    explicitly and checked.
    """

    def __init__(self, transfer: DICSpectralTransfer, node_shape: tuple[int, int]) -> None:
        self._transfer = transfer
        rows, columns = node_shape
        x, y = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
        self._basis = (
            np.stack([np.ones_like(x), x, y], axis=-1).reshape(-1, 3).astype(np.float64)
        )
        self._pseudo_inverse = np.linalg.pinv(self._basis)

    def _affine(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        flat = field.reshape(-1, field.shape[2])
        return (self._basis @ (self._pseudo_inverse @ flat)).reshape(field.shape)

    def apply(self, values: ArrayLike) -> NDArray[np.float64]:
        field = np.asarray(values, dtype=np.float64)
        affine = self._affine(field)
        return np.asarray(affine + self._transfer.apply(field - affine), dtype=np.float64)

    def adjoint(self, values: ArrayLike) -> NDArray[np.float64]:
        field = np.asarray(values, dtype=np.float64)
        filtered = np.asarray(self._transfer.apply(field), dtype=np.float64)
        return self._affine(field) + filtered - self._affine(filtered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", nargs=2, type=int, default=(1580, 1030))
    parser.add_argument("--pixels", type=int, default=100)
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
    periodic = DICSpectralTransfer.from_sinusoidal_csv(TRANSFER)
    variants = {
        "identity": IdentityTransfer(),
        "wrap_free": WrapFreeTransfer(periodic, grid.node_shape),
        "periodic": periodic,
    }

    interior = 2 * (grid.node_shape[0] - 2) * (grid.node_shape[1] - 2)
    noise_norm = float(np.sqrt(interior))
    results: dict[str, dict[str, object]] = {}
    left_vectors: dict[str, np.ndarray] = {}
    right_vectors: dict[str, np.ndarray] = {}

    for name, transfer in variants.items():
        operator = TensorPlasticObservabilityOperator.build(
            grid,
            young_modulus_mpa=YOUNG_MPA,
            poisson_ratio=POISSON,
            transfer=transfer,
            whitener=whitener,
        )
        generator = np.random.default_rng(9)
        probe_x = generator.normal(size=operator.plastic_size)
        probe_y = generator.normal(size=operator.observation_size)
        forward = float(operator.matvec(probe_x) @ probe_y)
        adjoint_error = abs(forward - float(probe_x @ operator.rmatvec(probe_y))) / abs(forward)

        weight = float(operator.kinematics.sample_quadrature_weight)
        residual = []
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
            residual.append(
                np.asarray(whitener.apply(measured - transfer.apply(elastic))).reshape(-1)
            )
        residual = np.asarray(residual)

        left, singular, right_transposed = svds(
            operator.as_linear_operator(), k=arguments.modes, tol=0
        )
        order = np.argsort(singular)[::-1]
        left, singular = left[:, order], singular[order]
        left_vectors[name] = left
        right_vectors[name] = right_transposed[order].T
        coefficients = left.T @ residual.T

        results[name] = {
            "adjoint_relative_error": adjoint_error,
            "singular_values": singular.tolist(),
            "residual_norm_over_noise": (np.linalg.norm(residual, axis=1) / noise_norm).tolist(),
            "maximum_coefficient_by_state": np.abs(coefficients).max(axis=0).tolist(),
        }
        print(
            f"{name:10s}: adjoint {adjoint_error:.2e}  sigma1 {singular[0]:.4e}  "
            f"state1 {results[name]['residual_norm_over_noise'][1]:6.3f}  "
            f"state20 {results[name]['residual_norm_over_noise'][20]:6.3f}  "
            f"state40 {results[name]['residual_norm_over_noise'][40]:7.3f}"
        )

    # Per-mode agreement between the identity and the repaired transfer.
    identity_modes = right_vectors["identity"]
    identity_operator = TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=variants["identity"],
        whitener=whitener,
    )
    wrap_operator = TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=variants["wrap_free"],
        whitener=whitener,
    )
    per_mode = []
    for index in range(arguments.modes):
        mode = identity_modes[:, index]
        plain = identity_operator.matvec(mode)
        filtered = wrap_operator.matvec(mode)
        correlation = float(
            plain @ filtered / (np.linalg.norm(plain) * np.linalg.norm(filtered))
        )
        error = float(np.linalg.norm(plain - filtered) / np.linalg.norm(plain))
        per_mode.append({"mode": index + 1, "correlation": correlation, "relative_error": error})

    angles = {}
    for pair in (("identity", "wrap_free"), ("identity", "periodic")):
        cosines = np.linalg.svd(
            right_vectors[pair[0]][:, :4].T @ right_vectors[pair[1]][:, :4], compute_uv=False
        )
        angles["_vs_".join(pair)] = np.degrees(
            np.arccos(np.clip(cosines, -1.0, 1.0))
        ).tolist()

    output = {
        "schema_version": 1,
        "pixels": pixels,
        "origin_nodes": [x0, y0],
        "variants": results,
        "per_mode_identity_against_wrap_free": per_mode,
        "principal_angles_of_the_leading_four_modes_deg": angles,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", "utf-8")

    print("\n state | identity | wrap_free | periodic   (residual norm over pure noise)")
    for state in (1, 5, 10, 20, 25, 30, 35, 40):
        print(
            f" {state:5d} | {results['identity']['residual_norm_over_noise'][state]:8.3f} | "
            f"{results['wrap_free']['residual_norm_over_noise'][state]:9.3f} | "
            f"{results['periodic']['residual_norm_over_noise'][state]:8.3f}"
        )
    print("\n  j | correlation | relative error   (identity against wrap-free)")
    for entry in per_mode[:8]:
        print(
            f"{entry['mode']:3d} | {entry['correlation']:11.5f} | {entry['relative_error']:15.4f}"
        )
    print("\nprincipal angles of the leading four modes (deg):")
    for key, value in angles.items():
        print(f"  {key:24s} {np.round(value, 3).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
