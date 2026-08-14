#!/usr/bin/env python3
"""Is the unexplained part of the measured field plasticity, or model error?

Every reconstruction of a plastic field from DIC rests on one premise: that
what a homogeneous isotropic elastic model cannot explain is plasticity. The
premise is testable without any plastic model at all.

For each measured state, take the elastic field carrying the same boundary
displacement and measure what is left in the interior. Two signatures separate
the candidates:

* **noise** does not grow with the load, so the residual is flat in state and
  its ratio to the displacement amplitude falls as the load rises;
* **model error** -- elastic heterogeneity, out-of-plane motion, a systematic
  instrument effect -- is proportional to the load, so that ratio is constant;
* **plasticity** appears only after yield, so the ratio is zero early and rises.

The early states are the decisive ones: before yield the material cannot be
plastic, so anything above the noise there is the model failing.

This matters beyond a sanity check. A free tensor eigenstrain field can
reproduce the effect of an elastic inclusion exactly -- that is Eshelby's
equivalent inclusion -- so elastic heterogeneity and plasticity produce the
same class of forcing ``f = G z``. If the residual is heterogeneity, a plastic
reconstruction will absorb it and report it as plastic strain, and no amount of
regularisation in the plastic space can separate the two.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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
DIC_UNCERTAINTY_MM = 9.40e-5
#: Lower-left corner of the crops, shared so the sizes are nested.
ORIGIN = (1610, 1075)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 40, 60, 100])
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report_data = json.loads(HISTORY_REPORT.read_text(encoding="utf-8"))
    bounds = list(map(int, report_data["solve_bounds"]))
    source = np.load(HISTORY, mmap_mode="r", allow_pickle=False)
    noise = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    canonical = image_flow_to_canonical(np.asarray(noise[:512, :512]), pixel_size_mm=PIXEL_SIZE_MM)
    transfer = DICSpectralTransfer.from_sinusoidal_csv(TRANSFER)
    elasticity = plane_stress_elasticity(YOUNG_MPA, POISSON)

    records = []
    for pixels in arguments.sizes:
        x0, y0 = ORIGIN
        x1, y1 = x0 + pixels, y0 + pixels
        if not (bounds[0] <= x0 < x1 <= bounds[1] and bounds[2] <= y0 < y1 <= bounds[3]):
            raise ValueError(f"the {pixels}-pixel crop leaves the measured window")
        history = np.asarray(
            source[:, x0 - bounds[0] : x1 - bounds[0] + 1, y0 - bounds[2] : y1 - bounds[2] + 1, :],
            dtype=np.float64,
        )
        history = history - history[0]

        grid = StructuredGrid2D(
            pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels
        )
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
        operator = TensorPlasticObservabilityOperator.build(
            grid,
            young_modulus_mpa=YOUNG_MPA,
            poisson_ratio=POISSON,
            transfer=transfer,
            whitener=whitener,
        )
        weight = float(operator.kinematics.sample_quadrature_weight)
        interior_components = 2 * (grid.node_shape[0] - 2) * (grid.node_shape[1] - 2)
        noise_norm = float(np.sqrt(interior_components))

        states = []
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
            deviation = (measured - transfer.apply(elastic))[1:-1, 1:-1, :]
            amplitude = float(np.sqrt((measured**2).mean()))
            deviation_rms = float(np.sqrt((deviation**2).mean()))
            whitened = float(np.linalg.norm(whitener.apply(measured - transfer.apply(elastic))))
            states.append(
                {
                    "state": state,
                    "displacement_rms_mm": amplitude,
                    "deviation_rms_mm": deviation_rms,
                    "deviation_over_displacement": deviation_rms / amplitude
                    if amplitude > 0.0
                    else 0.0,
                    "deviation_over_dic_sigma": deviation_rms / DIC_UNCERTAINTY_MM,
                    "whitened_norm_over_noise": whitened / noise_norm,
                }
            )

        ratios = np.array([entry["deviation_over_displacement"] for entry in states[1:]])
        records.append(
            {
                "pixels": pixels,
                "window_mm": PIXEL_SIZE_MM * pixels,
                "crop_nodes": [x0, x1, y0, y1],
                "pure_noise_whitened_norm": noise_norm,
                "states": states,
                "deviation_over_displacement_median": float(np.median(ratios)),
                "deviation_over_displacement_first_five": ratios[:5].tolist(),
            }
        )
        early = states[1]
        final = states[-1]
        print(
            f"pixels={pixels:4d}  window={PIXEL_SIZE_MM * pixels * 1e3:6.1f} um  "
            f"median deviation/displacement = {np.median(ratios):.3e}  "
            f"state1 = {early['whitened_norm_over_noise']:7.3f} noise  "
            f"state40 = {final['whitened_norm_over_noise']:8.3f} noise"
        )

    output = {
        "schema_version": 1,
        "pixel_size_mm": PIXEL_SIZE_MM,
        "dic_uncertainty_mm": DIC_UNCERTAINTY_MM,
        "origin_nodes": list(ORIGIN),
        "note": (
            "deviation is the interior part of the measured field that the homogeneous "
            "isotropic elastic extension of its own boundary cannot reproduce"
        ),
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", "utf-8")

    print("\ndeviation / displacement, by state -- flat means proportional to the load")
    header = "  state |" + "".join(f"  {record['pixels']:>5d} px" for record in records)
    print(header)
    for state in (1, 5, 10, 20, 30, 40):
        row = f"  {state:5d} |"
        for record in records:
            row += f"  {record['states'][state]['deviation_over_displacement']:8.2e}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
