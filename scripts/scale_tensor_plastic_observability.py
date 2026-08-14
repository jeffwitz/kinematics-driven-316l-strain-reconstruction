#!/usr/bin/env python3
"""How the tensor plastic observability scales with the size of the crop.

The M20 spectrum says no plastic mode reaches the DIC noise, but M20 is a
`36.8 um` window and an eigenstrain produces a displacement of order
`strain x length` against a per-node noise that does not change. The M20 answer
is therefore about the window, not yet about the experiment.

This sweeps the window with the matrix-free operator, keeping the plastic
amplitude fixed at the two scales the archived oracle actually reached: the
root-mean-square of a single increment, and of the accumulated equivalent
plastic strain at the final state. Only the leading modes are computed --
the M20 spectrum drops by a factor of 330 after the second, so a partial
decomposition is enough to follow the question that matters.

Nothing is fitted and no experimental state enters: the operator is built from
the mechanics and the measurement chain alone.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import svds

from fem_inhouse.identification.dic_whitening import DICSpectralTransfer, DICSpectralWhitener
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.measurement import image_flow_to_canonical
from fem_inhouse.spectral2d.grid import StructuredGrid2D

ROOT = Path(__file__).resolve().parents[1]
NOISE = (
    ROOT
    / "validation/reference_data/dic_uncertainty_propagation_p0043_v1"
    / "centred_repeat_flow_pixels.npy"
)
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "validation/_generated/performance/experimental_oracle_p43_m20"
    / "tensor_observability_scaling.json"
)
PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30
REFERENCE_AMPLITUDES = {
    "single_increment_rms": 2.3503361528920064e-04,
    "accumulated_rms_final_state": 5.669788370458351e-03,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 30, 40, 60, 80, 100])
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    noise = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    canonical = image_flow_to_canonical(np.asarray(noise[:512, :512]), pixel_size_mm=PIXEL_SIZE_MM)
    transfer = DICSpectralTransfer.from_sinusoidal_csv(TRANSFER)

    records = []
    for pixels in arguments.sizes:
        started = time.perf_counter()
        grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
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
        singular = svds(
            operator.as_linear_operator(),
            k=arguments.modes,
            return_singular_vectors=False,
            tol=0,
            maxiter=50_000,
        )
        singular = np.sort(np.asarray(singular, dtype=np.float64))[::-1]
        observed = operator.observation_size
        signal_to_noise = {
            name: (amplitude * singular / np.sqrt(observed)).tolist()
            for name, amplitude in REFERENCE_AMPLITUDES.items()
        }
        records.append(
            {
                "pixels": pixels,
                "window_mm": PIXEL_SIZE_MM * pixels,
                "plastic_components": operator.plastic_size,
                "mechanical_free_dofs": operator.free_size,
                "observed_components": observed,
                "singular_values": singular.tolist(),
                "gap_after_second_mode": float(singular[1] / singular[2])
                if singular.size > 2
                else None,
                "signal_to_noise": signal_to_noise,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        best = signal_to_noise["accumulated_rms_final_state"][0]
        print(
            f"pixels={pixels:4d}  window={PIXEL_SIZE_MM * pixels * 1e3:7.1f} um  "
            f"dofs={operator.free_size:6d}  s1={singular[0]:.4e}  "
            f"gap={records[-1]['gap_after_second_mode']:8.1f}  "
            f"SNR1(total)={best:.4f}  [{records[-1]['elapsed_seconds']:.1f} s]"
        )

    report = {
        "schema_version": 1,
        "pixel_size_mm": PIXEL_SIZE_MM,
        "reference_amplitudes": REFERENCE_AMPLITUDES,
        "modes": arguments.modes,
        "note": (
            "operator built from mechanics and measurement chain only; no experimental "
            "state enters, so the spectrum does not depend on the loading history"
        ),
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")

    print("\nleading SNR against window size")
    print(f"{'pixels':>7} | {'window um':>9} | {'SNR increment':>13} | {'SNR accumulated':>15}")
    for record in records:
        print(
            f"{record['pixels']:7d} | {record['window_mm'] * 1e3:9.1f} | "
            f"{record['signal_to_noise']['single_increment_rms'][0]:13.4f} | "
            f"{record['signal_to_noise']['accumulated_rms_final_state'][0]:15.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
