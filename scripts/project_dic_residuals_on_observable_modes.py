#!/usr/bin/env python3
"""Are the observable plastic modes actually excited by the experiment?

The observability spectrum says what the DIC chain *could* see if a mode were
present at a reference amplitude. It says nothing about whether the experiment
contains it. This projects the real residuals of the 40 measured states onto
the left singular vectors of the observability operator, which turns

```text
"this mode is observable"   into   "this mode is present, at N sigma".
```

The residual is the part of the measured field that elasticity cannot explain
under the same boundary displacement. Writing the measurement as
``u_DIC = M_D u_true + noise`` and ``u_true = u_elastic + K^-1 G eps_p``,

```text
r_n = W_D ( u_DIC,n - M_D u_elastic,n ) = A (H^1/2 eps_p,n) + whitened noise.
```

Two details decide whether the numbers mean anything.

The transfer applies to the MODEL only, never to the measurement: the measured
field has already been through the instrument. Applying `M_D` to both sides
would blur the data a second time.

And the elastic reference needs no separate boundary-coupling block. The
measured field satisfies ``K u_int + K_ib u_b = f_int`` with ``f_int`` its own
interior out-of-balance force, while the elastic field satisfies the same
equation with a zero right-hand side, so
``u_elastic,int = u_DIC,int - K^-1 f_int`` exactly.

Because `W_D` whitens, a projection onto a unit left singular vector has unit
variance under pure noise: the coefficients come out already expressed in noise
sigma, with no separate uncertainty propagation. The earliest states are
therefore a built-in null test -- before yield the residual should be noise,
and any mode reading far above one sigma there is measuring model error rather
than plasticity.
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
HISTORY_REPORT = HISTORY.with_name("report.json")
PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30


def _load_history(crop: tuple[int, int, int, int]) -> np.ndarray:
    report = json.loads(HISTORY_REPORT.read_text(encoding="utf-8"))
    x0_solve, x1_solve, y0_solve, y1_solve = map(int, report["solve_bounds"])
    x0, x1, y0, y1 = crop
    if not (x0_solve <= x0 < x1 <= x1_solve and y0_solve <= y0 < y1 <= y1_solve):
        raise ValueError(f"crop {crop} lies outside the solve bounds of the measured history")
    source = np.load(HISTORY, mmap_mode="r", allow_pickle=False)
    window = np.asarray(
        source[:, x0 - x0_solve : x1 - x0_solve + 1, y0 - y0_solve : y1 - y0_solve + 1, :],
        dtype=np.float64,
    )
    return window - window[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=(1610, 1710, 1075, 1175))
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    crop = tuple(int(value) for value in arguments.crop_nodes)
    history = _load_history(crop)  # type: ignore[arg-type]
    states, nodes_x, nodes_y, _ = history.shape
    pixels_x, pixels_y = nodes_x - 1, nodes_y - 1
    if pixels_x != pixels_y:
        raise ValueError("the crop must be square")
    pixels = pixels_x

    grid = StructuredGrid2D(
        pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels
    )
    noise = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    canonical = image_flow_to_canonical(np.asarray(noise[:512, :512]), pixel_size_mm=PIXEL_SIZE_MM)
    support = np.ones((*grid.node_shape, 2), dtype=np.float64)
    support[[0, -1], :, :] = 0.0
    support[:, [0, -1], :] = 0.0
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
    left, singular, _ = svds(operator.as_linear_operator(), k=arguments.modes, tol=0)
    order = np.argsort(singular)[::-1]
    left, singular = left[:, order], singular[order]

    elasticity = plane_stress_elasticity(YOUNG_MPA, POISSON)
    weight = float(operator.kinematics.sample_quadrature_weight)
    coefficients = np.empty((arguments.modes, states), dtype=np.float64)
    residual_norms = np.empty(states, dtype=np.float64)

    for state in range(states):
        measured = history[state]
        strain = np.asarray(operator.kinematics.strain(measured)).reshape(-1, 3)
        stress = strain @ elasticity
        out_of_balance = (
            -pack_interior(
                operator.kinematics.divergence_from_sample_stress(
                    stress.reshape((pixels, pixels, 2, 3))
                )
            )
            / weight
        )
        elastic = measured.copy()
        correction = operator.solve_stiffness(out_of_balance)
        elastic[1:-1, 1:-1, :] -= correction.reshape(nodes_x - 2, nodes_y - 2, 2)
        residual = whitener.apply(measured - transfer.apply(elastic))
        flattened = np.asarray(residual, dtype=np.float64).reshape(-1)
        coefficients[:, state] = left.T @ flattened
        residual_norms[state] = float(np.linalg.norm(flattened))

    observed = operator.observation_size
    # A projection onto a unit vector of whitened noise has unit variance, so the
    # coefficients are already z-scores. The residual norm is compared against
    # the chi distribution mean of a field of independent unit-variance
    # components, restricted to the interior the whitener supports.
    interior_components = 2 * (nodes_x - 2) * (nodes_y - 2)
    expected_noise_norm = float(np.sqrt(interior_components))

    early = slice(1, 6)
    late = slice(states - 5, states)
    report = {
        "schema_version": 1,
        "crop_nodes": list(crop),
        "pixels": pixels,
        "states": states,
        "modes": arguments.modes,
        "observed_components": observed,
        "singular_values": singular.tolist(),
        "coefficients_in_noise_sigma": coefficients.tolist(),
        "residual_norm": residual_norms.tolist(),
        "expected_pure_noise_residual_norm": expected_noise_norm,
        "null_test_early_states": {
            "states": list(range(early.start, early.stop)),
            "maximum_absolute_coefficient": float(np.abs(coefficients[:, early]).max()),
        },
        "per_mode": [
            {
                "mode": index + 1,
                "singular_value": float(singular[index]),
                "maximum_absolute_coefficient": float(np.abs(coefficients[index]).max()),
                "final_state_coefficient": float(coefficients[index, -1]),
                "mean_absolute_early": float(np.abs(coefficients[index, early]).mean()),
                "mean_absolute_late": float(np.abs(coefficients[index, late]).mean()),
                "monotone_fraction": float(
                    np.mean(np.diff(coefficients[index]) * np.sign(coefficients[index, -1]) > 0)
                ),
            }
            for index in range(arguments.modes)
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")

    print(f"crop {crop}  pixels={pixels}  states={states}  modes={arguments.modes}")
    print(
        f"residual norm: state 1 = {residual_norms[1]:.1f}, "
        f"state {states - 1} = {residual_norms[-1]:.1f}, "
        f"pure noise would give {expected_noise_norm:.1f}"
    )
    print(
        "null test, states 1-5: maximum |coefficient| = "
        f"{report['null_test_early_states']['maximum_absolute_coefficient']:.2f} sigma"
    )
    print("\n  j |    sigma_j |  |c| early |  |c| late | c(final) | monotone")
    for entry in report["per_mode"][: min(12, arguments.modes)]:
        print(
            f"{entry['mode']:3d} | {entry['singular_value']:10.3e} | "
            f"{entry['mean_absolute_early']:10.2f} | {entry['mean_absolute_late']:9.2f} | "
            f"{entry['final_state_coefficient']:8.2f} | {entry['monotone_fraction']:8.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
