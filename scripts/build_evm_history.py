#!/usr/bin/env python3
"""Equivalent-strain history and its band energies, once, for the benchmark.

Every later step -- the POD baseline, the convolutional autoencoder, the
holdout evaluations -- reads the same equivalent-strain maps, so they are
computed once here rather than derived independently three times with three
chances of disagreeing.

The stored quantity is the physical `EVM(x, y, t)` together with its spatial
mean `m_t`. The normalised morphology map is deliberately *not* stored: the
specification allows two definitions,

```text
A:  H_t = EVM_t / (m_t + eps)
B:  H_t = EVM_t - G_macro * EVM_t
```

and keeping `EVM` plus `m_t` leaves both reachable at no cost while preserving
the round trip back to the physical field, which a stored `H` alone would not.

The band decomposition of section 5 is produced in the same pass, since it
needs exactly the same maps. Four fixed wavelength bands, 2-8, 8-32, 32-128 and
above 128 pixels, obtained as differences of Gaussian low passes. The band
edges are not tuned and are not meant to be: the question is only whether fine
texture and wide bands move differently over the loading, which a coarse
partition answers as well as a fine one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from compare_disflow_profiles_p43 import equivalent_strain  # type: ignore[import-not-found]
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

FloatArray = NDArray[np.float64]

DATA_ROOT = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical")
SOURCE = DATA_ROOT / "p0043_disflow_history_tuned.h5"
#: Gaussian sigmas whose low passes bracket the four fixed wavelength bands.
#: A Gaussian of sigma s suppresses wavelengths below roughly 2 pi s / 3, so
#: these land near 8, 32 and 128 pixels.
BAND_SIGMAS = (1.3, 5.1, 20.4)
BAND_NAMES = ("2-8 px", "8-32 px", "32-128 px", "above 128 px")


def band_energies(field: FloatArray) -> list[float]:
    """Energy of `field` in four fixed wavelength bands, by low-pass difference."""

    low_passes = [gaussian_filter(field, sigma) for sigma in BAND_SIGMAS]
    parts = [
        field - low_passes[0],
        low_passes[0] - low_passes[1],
        low_passes[1] - low_passes[2],
        low_passes[2],
    ]
    return [float((part**2).mean()) for part in parts]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "p0043_evm_history.h5")
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()

    with h5py.File(arguments.source, "r") as handle:
        rows_group = handle["displacement_pixel/along_rows"]
        columns_group = handle["displacement_pixel/along_columns"]
        states = int(rows_group.shape[0])
        probe = equivalent_strain(
            np.asarray(rows_group[0], dtype=np.float64),
            np.asarray(columns_group[0], dtype=np.float64),
        )
        shape = probe.shape
        print(f"{states} states, EVM shape {shape}", flush=True)

        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(arguments.output, "w") as target:
            dataset = target.create_dataset(
                "evm",
                shape=(states, *shape),
                dtype=np.float32,
                chunks=(1, min(512, shape[0]), min(512, shape[1])),
                compression="gzip",
                compression_opts=4,
            )
            means, energies = [], []
            started = time.perf_counter()
            for state in range(states):
                field = equivalent_strain(
                    np.asarray(rows_group[state], dtype=np.float64),
                    np.asarray(columns_group[state], dtype=np.float64),
                )
                dataset[state] = field
                means.append(float(field.mean()))
                energies.append(band_energies(field))
                if state % 5 == 0 or state == states - 1:
                    print(
                        f"  state {state:2d}: mean {means[-1]:.3e}  "
                        f"({time.perf_counter() - started:.0f} s)",
                        flush=True,
                    )

            null = equivalent_strain(
                np.asarray(handle["null_test_pixel/along_rows"], dtype=np.float64)
                - np.asarray(rows_group[states - 1], dtype=np.float64),
                np.asarray(handle["null_test_pixel/along_columns"], dtype=np.float64)
                - np.asarray(columns_group[states - 1], dtype=np.float64),
            )
            target.create_dataset("noise_evm", data=null.astype(np.float32),
                                  compression="gzip", compression_opts=4)
            target.attrs["source"] = str(arguments.source)
            target.attrs["states"] = states
            target.attrs["mean_evm"] = np.asarray(means)
            target.attrs["definition"] = (
                "von Mises equivalent of the first-difference strain with the "
                "incompressible closure eps_zz = -(eps_xx + eps_yy)"
            )
            target.attrs["noise_note"] = (
                "noise_evm is the equivalent strain of the repeated-state "
                "difference, the metrology floor at these settings"
            )

    noise_energies = band_energies(null)
    report = {
        "schema_version": 1,
        "status": "completed_evm_history",
        "source": str(arguments.source),
        "output": str(arguments.output),
        "states": states,
        "shape": list(shape),
        "mean_evm": means,
        "band_names": list(BAND_NAMES),
        "band_energy_per_state": energies,
        "noise_band_energy": noise_energies,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    formatted = ", ".join(f"{value:.2e}" for value in noise_energies)
    print(f"\nnoise band energies: {formatted}")
    print(f"wrote {arguments.output} and {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
