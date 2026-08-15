#!/usr/bin/env python3
"""Observability and projections in the metric the Dirichlet crop does not suppress.

Everything before this measured the residual in displacement and whitened it
with the covariance of the raw DIC noise. Both were wrong for this observable.

A full Dirichlet residual is pinned to zero on the boundary, so its displacement
norm is structurally small while its strain recovers the full amplitude of a
narrow feature. And the residual is `(I - E P_b) u`, so under pure noise it is
`(I - E P_b) n`: the elastic extension absorbs most of the noise, and the raw
DIC covariance overstates the reference by a factor of eighteen.

Both are fixed here at once. The observable is the **strain** of the residual,
and its whitener is estimated from real noise realisations pushed through the
*same* crop-and-extend operator, so a projection onto a unit left singular
vector has unit variance by construction and the coefficients are genuine
z-scores rather than norm ratios.

The operator is the strain-space twin of the plastic one,

```text
A = W_eps B K^-1 B^T C H^-1/2,
```

with the same gauge `H = M^-1` and the same mechanics. Only what is looked at,
and what it is compared against, has changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator, svds

from fem_inhouse.identification.dic_whitening import DICSpectralWhitener
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.measurement import image_flow_to_canonical
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior

ROOT = Path(__file__).resolve().parents[1]
NOISE = (
    ROOT
    / "validation/reference_data/dic_uncertainty_propagation_p0043_v1"
    / "centred_repeat_flow_pixels.npy"
)
HISTORY = (
    ROOT
    / "validation/reference_data/dic_multistep_history_p0043_repaired_v1"
    / "repaired_history_mm.npy"
)
PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30
#: RMS accumulated equivalent plastic strain of the archived oracle, the
#: amplitude at which a mode's detectability is judged.
REFERENCE_AMPLITUDE = 5.669788370458351e-03


class _Identity:
    def apply(self, values: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(values, dtype=np.float64)


def _elastic_extension(
    operator: TensorPlasticObservabilityOperator, field: np.ndarray
) -> np.ndarray:
    pixels = operator.grid.nx
    strain = np.asarray(operator.kinematics.strain(field)).reshape(-1, 3)
    stress = np.einsum("pi,pij->pj", strain, operator.elasticity)
    forcing = (
        -pack_interior(
            operator.kinematics.divergence_from_sample_stress(
                stress.reshape((pixels, pixels, 2, 3))
            )
        )
        / operator.quadrature_weight
    )
    extension = field.copy()
    extension[1:-1, 1:-1, :] -= operator.solve_stiffness(forcing).reshape(
        pixels - 1, pixels - 1, 2
    )
    return extension


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", nargs=2, type=int, default=(1580, 1030))
    parser.add_argument("--pixels", type=int, default=100)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--noise-realisations", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    pixels = arguments.pixels
    x0, y0 = arguments.origin
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    operator = TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=_Identity(),
        whitener=_Identity(),
    )
    strain_shape = (pixels, pixels, 6)

    # The noise of the observable, not of the measurement: real realisations
    # pushed through the same crop-and-extend operator, then differentiated.
    noise_source = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    canonical = np.asarray(
        image_flow_to_canonical(
            np.asarray(noise_source[:1024, :1024]), pixel_size_mm=PIXEL_SIZE_MM
        )
    )
    generator = np.random.default_rng(20260816)
    samples = np.empty((arguments.noise_realisations, *strain_shape), dtype=np.float64)
    for index in range(arguments.noise_realisations):
        row = generator.integers(0, canonical.shape[0] - pixels - 1)
        column = generator.integers(0, canonical.shape[1] - pixels - 1)
        patch = np.ascontiguousarray(
            canonical[row : row + pixels + 1, column : column + pixels + 1, :]
        )
        residual = patch - _elastic_extension(operator, patch)
        samples[index] = np.asarray(operator.kinematics.strain(residual)).reshape(strain_shape)

    whitener = DICSpectralWhitener.from_noise_realisations(samples)

    def whiten(strain: np.ndarray) -> np.ndarray:
        return np.asarray(whitener.apply(strain.reshape(strain_shape)), dtype=np.float64)

    observation_size = int(np.prod(strain_shape))

    def matvec(values: ArrayLike) -> NDArray[np.float64]:
        vector = np.asarray(values, dtype=np.float64).reshape(-1, 3)
        stress = np.einsum(
            "pi,pij->pj", vector @ operator.inverse_gauge_root, operator.elasticity
        )
        displacement = unpack_interior(
            operator.solve_stiffness(operator._strain_transpose(stress.reshape(-1))), grid
        )
        strain = np.asarray(operator.kinematics.strain(displacement))
        return whiten(strain).reshape(-1)

    def rmatvec(values: ArrayLike) -> NDArray[np.float64]:
        field = np.asarray(values, dtype=np.float64).reshape(strain_shape)
        dual = np.asarray(whitener.adjoint(field), dtype=np.float64).reshape(-1, 3)
        forcing = operator._strain_transpose(dual.reshape(-1))
        displacement = unpack_interior(operator.solve_stiffness(forcing), grid)
        strain = np.asarray(operator.kinematics.strain(displacement)).reshape(-1, 3)
        stress = np.einsum("pi,pij->pj", strain, operator.elasticity)
        return (stress @ operator.inverse_gauge_root).reshape(-1)

    action = LinearOperator(
        (observation_size, operator.plastic_size),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float64,
    )
    probe_x = generator.normal(size=operator.plastic_size)
    probe_y = generator.normal(size=observation_size)
    forward = float(matvec(probe_x) @ probe_y)
    adjoint_error = abs(forward - float(probe_x @ rmatvec(probe_y))) / abs(forward)

    left, singular, _ = svds(action, k=arguments.modes, tol=0)
    order = np.argsort(singular)[::-1]
    left, singular = left[:, order], singular[order]

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

    coefficients = np.empty((arguments.modes, history.shape[0]), dtype=np.float64)
    norms = np.empty(history.shape[0], dtype=np.float64)
    for state in range(history.shape[0]):
        residual = history[state] - _elastic_extension(operator, history[state])
        whitened = whiten(np.asarray(operator.kinematics.strain(residual))).reshape(-1)
        coefficients[:, state] = left.T @ whitened
        norms[state] = float(np.linalg.norm(whitened))

    # Calibration of the whitener itself: whitened noise must have unit variance,
    # so its norm is the square root of the component count.
    checks = np.array(
        [float(np.linalg.norm(whiten(sample))) for sample in samples[: min(40, len(samples))]]
    )
    expected = float(np.sqrt(observation_size))

    signal_to_noise = REFERENCE_AMPLITUDE * singular / np.sqrt(observation_size)
    output = {
        "schema_version": 1,
        "pixels": pixels,
        "origin_nodes": [x0, y0],
        "observation": "whitened strain of the elastic-closure residual",
        "noise_realisations": arguments.noise_realisations,
        "adjoint_relative_error": adjoint_error,
        "whitened_noise_norm_over_expected": float(checks.mean() / expected),
        "reference_amplitude": REFERENCE_AMPLITUDE,
        "singular_values": singular.tolist(),
        "signal_to_noise_at_reference_amplitude": signal_to_noise.tolist(),
        "modes_above_one_sigma": int(np.count_nonzero(signal_to_noise > 1.0)),
        "modes_above_three_sigma": int(np.count_nonzero(signal_to_noise > 3.0)),
        "residual_norm_over_expected_noise": (norms / expected).tolist(),
        "coefficients_in_noise_sigma": coefficients.tolist(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", "utf-8")

    print(f"adjoint relative error          : {adjoint_error:.3e}")
    print(f"whitened noise norm / expected  : {checks.mean() / expected:.4f}")
    print(f"spectrum sigma1/sigma{arguments.modes:<2d}         : {singular[0] / singular[-1]:.3f}")
    print(
        f"modes above 1 sigma at p_ref    : {output['modes_above_one_sigma']} "
        f"(above 3 sigma: {output['modes_above_three_sigma']})"
    )
    print("\n state | residual/noise | max |c| sigma | c1     | c2     | c3")
    for state in (1, 5, 10, 20, 25, 30, 35, 40):
        print(
            f" {state:5d} | {norms[state] / expected:14.3f} | "
            f"{np.abs(coefficients[:, state]).max():13.2f} | "
            + " | ".join(f"{coefficients[index, state]:6.2f}" for index in range(3))
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
