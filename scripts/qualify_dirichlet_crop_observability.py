#!/usr/bin/env python3
"""Does a Dirichlet crop destroy the plastic signal it is meant to reveal?

The procedure takes the measured displacement on the four sides of a crop,
extends it elastically, and calls the interior difference the mechanical
defect. Two objections have to be settled before any conclusion drawn from that
number means anything.

**The boundary already carries the signal.** A localisation band crossing the
crop shows up in the displacement of all four sides, so imposing them injects
part of its own signature into the reference. The residual sees only what an
elastic extension of that boundary cannot reproduce.

**And the displacement metric hides it.** In a full Dirichlet problem the
residual is pinned to zero on the boundary, so its displacement norm is
structurally small: a band of width `w` at plastic strain `e` offsets the
displacement by only `e w`. Differentiating recovers the full amplitude, so the
same residual that looks negligible in millimetres can be of order `e` in
strain. Judging in displacement systematically underweights exactly the short,
localised features being looked for.

This answers both with a controlled field: a known eigenstrain band is imposed
on a large domain, the crop is taken from its middle, and the procedure is run
on it. Nothing is measured, nothing is fitted, and the answer is known in
advance -- so if the method reports "below the noise" here, the method is what
is losing the signal.

The noise reference is propagated through the *same* operator. Under pure noise
the residual is `(I - E P_b) n`, not `n`, so real noise realisations are cropped
and extended identically and the experimental residual is compared with their
distribution, in each metric.

Finally the guard is varied: the Dirichlet boundary is pushed away from the
measurement window while the window stays fixed. If the signal returns as the
boundary recedes, the boundary conditioning is what was removing it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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
PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30
DIC_UNCERTAINTY_MM = 9.40e-5


class _Identity:
    def apply(self, values):
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values):
        return np.asarray(values, dtype=np.float64)


def _operator(pixels: int) -> TensorPlasticObservabilityOperator:
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    return TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=_Identity(),
        whitener=_Identity(),
    )


def _elastic_extension(
    operator: TensorPlasticObservabilityOperator, field: np.ndarray
) -> np.ndarray:
    """The elastic field carrying the same boundary displacement as `field`."""

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


def _band_eigenstrain(pixels: int, amplitude: float, width_pixels: float) -> np.ndarray:
    """A shear-dominated localisation band crossing the domain diagonally."""

    x, y = np.meshgrid(np.arange(pixels), np.arange(pixels), indexing="ij")
    distance = (x + y - pixels) / np.sqrt(2.0)
    profile = np.exp(-0.5 * (distance / width_pixels) ** 2)
    field = np.zeros((pixels, pixels, 2, 3), dtype=np.float64)
    # Plastically incompressible in plane: e11 = -e22, plus shear.
    field[..., 0] = amplitude * profile[:, :, None]
    field[..., 1] = -amplitude * profile[:, :, None]
    field[..., 2] = 1.5 * amplitude * profile[:, :, None]
    return field


def _strain_norm(operator: TensorPlasticObservabilityOperator, field: np.ndarray) -> float:
    strain = np.asarray(operator.kinematics.strain(field))
    return float(np.sqrt((strain**2).mean()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", type=int, default=200)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--guards", nargs="+", type=int, default=[0, 10, 25, 50])
    parser.add_argument("--amplitude", type=float, default=1.0e-2)
    parser.add_argument("--band-width-pixels", type=float, default=6.0)
    parser.add_argument("--noise-realisations", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    outer, window = arguments.outer, arguments.window
    if window + 2 * max(arguments.guards) > outer:
        raise ValueError("the largest guard does not fit inside the outer domain")

    # A controlled truth: a plastic band on the large domain, under a far field.
    outer_operator = _operator(outer)
    eigenstrain = _band_eigenstrain(outer, arguments.amplitude, arguments.band_width_pixels)
    x, y = np.meshgrid(
        np.arange(outer + 1) * PIXEL_SIZE_MM, np.arange(outer + 1) * PIXEL_SIZE_MM, indexing="ij"
    )
    far_field = np.zeros((outer + 1, outer + 1, 2), dtype=np.float64)
    far_field[:, :, 0] = 8.0e-3 * x
    far_field[:, :, 1] = -2.4e-3 * y
    stress = np.einsum(
        "pi,pij->pj", eigenstrain.reshape(-1, 3), outer_operator.elasticity
    )
    forcing = (
        pack_interior(
            outer_operator.kinematics.divergence_from_sample_stress(
                stress.reshape((outer, outer, 2, 3))
            )
        )
        / outer_operator.quadrature_weight
    )
    truth = _elastic_extension(outer_operator, far_field)
    truth[1:-1, 1:-1, :] += outer_operator.solve_stiffness(-forcing).reshape(
        outer - 1, outer - 1, 2
    )

    centre = outer // 2
    half = window // 2

    noise_source = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    canonical = np.asarray(
        image_flow_to_canonical(np.asarray(noise_source[:1024, :1024]), pixel_size_mm=PIXEL_SIZE_MM)
    )
    generator = np.random.default_rng(20260815)

    records = []
    for guard in arguments.guards:
        pixels = window + 2 * guard
        operator = _operator(pixels)
        low = centre - half - guard
        high = centre + half + guard + 1
        cropped = truth[low:high, low:high, :]

        residual = cropped - _elastic_extension(operator, cropped)
        measured = residual[guard : guard + window + 1, guard : guard + window + 1, :]
        window_operator = _operator(window)

        noise_displacement, noise_strain = [], []
        for _ in range(arguments.noise_realisations):
            row = generator.integers(0, canonical.shape[0] - pixels - 1)
            column = generator.integers(0, canonical.shape[1] - pixels - 1)
            patch = np.ascontiguousarray(
                canonical[row : row + pixels + 1, column : column + pixels + 1, :]
            )
            noise_residual = patch - _elastic_extension(operator, patch)
            inner_noise = noise_residual[
                guard : guard + window + 1, guard : guard + window + 1, :
            ]
            noise_displacement.append(float(np.sqrt((inner_noise**2).mean())))
            noise_strain.append(_strain_norm(window_operator, inner_noise))

        signal_displacement = float(np.sqrt((measured**2).mean()))
        signal_strain = _strain_norm(window_operator, measured)
        injected = float(
            np.sqrt(
                (
                    eigenstrain[
                        centre - half : centre + half, centre - half : centre + half, :, :
                    ]
                    ** 2
                ).mean()
            )
        )
        records.append(
            {
                "guard_pixels": guard,
                "domain_pixels": pixels,
                "signal_displacement_rms_mm": signal_displacement,
                "signal_strain_rms": signal_strain,
                "noise_displacement_rms_mm": float(np.mean(noise_displacement)),
                "noise_strain_rms": float(np.mean(noise_strain)),
                "displacement_ratio": signal_displacement / float(np.mean(noise_displacement)),
                "strain_ratio": signal_strain / float(np.mean(noise_strain)),
                "injected_plastic_strain_rms": injected,
                "strain_recovered_fraction": signal_strain / injected,
            }
        )
        entry = records[-1]
        print(
            f"guard {guard:3d} px (domain {pixels:3d}) | "
            f"displacement {signal_displacement:.3e} mm vs noise "
            f"{entry['noise_displacement_rms_mm']:.3e}  ratio {entry['displacement_ratio']:7.2f} | "
            f"strain {signal_strain:.3e} vs noise {entry['noise_strain_rms']:.3e}  "
            f"ratio {entry['strain_ratio']:8.2f} | "
            f"recovered {entry['strain_recovered_fraction']:.4f}"
        )

    output = {
        "schema_version": 1,
        "outer_pixels": outer,
        "window_pixels": window,
        "injected_amplitude": arguments.amplitude,
        "band_width_pixels": arguments.band_width_pixels,
        "noise_realisations": arguments.noise_realisations,
        "dic_uncertainty_mm": DIC_UNCERTAINTY_MM,
        "note": (
            "noise is propagated through the same crop-and-extend operator as the signal, "
            "so both ratios compare like with like; the strain metric is the one a "
            "Dirichlet crop does not structurally suppress"
        ),
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
