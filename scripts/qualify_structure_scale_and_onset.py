#!/usr/bin/env python3
"""Two gates before any convolutional model: what scale, and does it appear?

Both are cheap, both come from the measurements alone, and each can change what
gets built.

## Gate one, the scale the data entitles us to

A pooled noise-to-signal ratio of 0.100 does not say at which wavelength the
signal stops being signal. Radial spectra of the equivalent strain and of the
null-test difference do, and their crossing is the smallest structure that can
honestly be learned. Below it a convolutional network would fit the noise --
which is reproducible enough across the field for a network to learn it very
well, and that is the failure mode this gate exists to prevent. The crossing
sets the patch size, the useful depth, and the smallest feature any later claim
may rest on.

## Gate two, whether the structure is plastic at all

The programme assumes the structured heterogeneity appears *with* plasticity.
If it is already there under low load, then what a network would learn is a
fixed signature -- microstructural elastic contrast, or the measurement chain --
and not plastic morphology.

The test removes the macroscopic field and watches the remainder:

```text
h(x, t) = EVM(x, t) - smooth(EVM(x, t))
```

Its bare amplitude is not the discriminant, because everything grows with load.
The discriminant is `RMS(h) / mean(EVM)`. Elastic heterogeneity of a fixed
microstructure scales with the applied load, so that ratio stays flat; plastic
localisation concentrates strain into a shrinking fraction of the material, so
it rises. A flat curve falsifies the premise, and the noise floor is drawn on
the same axes so that the early states are read against metrology rather than
against zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from compare_disflow_profiles_p43 import (  # type: ignore[import-not-found]
    _radial_spectrum,
    equivalent_strain,
)
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

FloatArray = NDArray[np.float64]

DATA = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/p0043_disflow_history_tuned.h5")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DATA)
    parser.add_argument("--smoothing-sigma", type=float, default=64.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    with h5py.File(arguments.history, "r") as handle:
        rows_group = handle["displacement_pixel/along_rows"]
        columns_group = handle["displacement_pixel/along_columns"]
        states = rows_group.shape[0]
        final_rows = np.asarray(rows_group[states - 1], dtype=np.float64)
        final_columns = np.asarray(columns_group[states - 1], dtype=np.float64)
        null_rows = np.asarray(handle["null_test_pixel/along_rows"], dtype=np.float64)
        null_columns = np.asarray(handle["null_test_pixel/along_columns"], dtype=np.float64)

        # --- Gate one -----------------------------------------------------
        signal = equivalent_strain(final_rows, final_columns)
        noise = equivalent_strain(null_rows - final_rows, null_columns - final_columns)
        frequency, signal_power = _radial_spectrum(signal)
        _, noise_power = _radial_spectrum(noise)
        ratio = np.sqrt(noise_power / np.maximum(signal_power, 1.0e-300))
        usable = frequency[(ratio < 1.0) & (frequency > 0.005)]
        crossing = float(usable.max()) if usable.size else float("nan")
        print(f"signal exceeds noise up to {crossing:.4f} cycles/px "
              f"= {1.0 / crossing:.1f} px wavelength", flush=True)
        for target in (0.5, 0.7, 0.9):
            reached = frequency[(ratio < target) & (frequency > 0.005)]
            if reached.size:
                print(
                    f"  noise/signal below {target:.1f} up to "
                    f"{1.0 / reached.max():.1f} px",
                    flush=True,
                )

        # --- Gate two -----------------------------------------------------
        onset = []
        for state in range(states):
            rows = np.asarray(rows_group[state], dtype=np.float64)
            columns = np.asarray(columns_group[state], dtype=np.float64)
            field = equivalent_strain(rows, columns)
            fluctuation = field - gaussian_filter(field, arguments.smoothing_sigma)
            macroscopic = float(field.mean())
            deviation = float(np.sqrt((fluctuation**2).mean()))
            onset.append(
                {
                    "state": state,
                    "mean_evm": macroscopic,
                    "fluctuation_rms": deviation,
                    "relative_fluctuation": deviation / max(macroscopic, 1.0e-30),
                }
            )
            if state % 5 == 0 or state == states - 1:
                print(
                    f"  state {state:2d}: mean EVM {macroscopic:.3e}  "
                    f"fluctuation {deviation:.3e}  ratio "
                    f"{onset[-1]['relative_fluctuation']:.4f}",
                    flush=True,
                )

    noise_field = equivalent_strain(null_rows - final_rows, null_columns - final_columns)
    noise_detail = noise_field - gaussian_filter(noise_field, arguments.smoothing_sigma)
    noise_fluctuation = float(np.sqrt((noise_detail**2).mean()))

    report = {
        "schema_version": 1,
        "status": "completed_structure_scale_and_onset",
        "history": str(arguments.history),
        "smoothing_sigma_pixel": arguments.smoothing_sigma,
        "scale_gate": {
            "crossing_cycles_per_pixel": crossing,
            "crossing_wavelength_pixel": 1.0 / crossing if crossing == crossing else None,
            "frequency": frequency.tolist(),
            "noise_over_signal_amplitude": ratio.tolist(),
        },
        "onset_gate": {
            "definition": "RMS(EVM - smooth(EVM)) / mean(EVM)",
            "noise_fluctuation_rms": noise_fluctuation,
            "per_state": onset,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    early = np.mean([entry["relative_fluctuation"] for entry in onset[1:6]])
    late = np.mean([entry["relative_fluctuation"] for entry in onset[-5:]])
    print(
        f"\nrelative fluctuation: early states {early:.4f}, late states {late:.4f}, "
        f"growth {late / max(early, 1e-30):.2f}x"
    )
    print(f"noise fluctuation floor {noise_fluctuation:.3e}")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
