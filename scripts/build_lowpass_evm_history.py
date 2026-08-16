#!/usr/bin/env python3
"""The equivalent-strain history with its non-mechanical fine band removed.

The band diagnostic established that the 2-8 pixel content of these fields is
flat in absolute terms across the whole loading -- energy growth 1.07 between
states 5 and 40, against 18.6 above 128 pixels and 17.45 for the squared mean.
A field that does not grow with the load is not mechanical, and the null test
accounts for only fourteen percent of it, so it is a fixed pattern rather than
acquisition noise; most plausibly a correlation bias tied to the speckle, which
repeats between two acquisitions and therefore escapes a null test.

That band dominates the gradient error, which is why the convolutional
benchmark was being judged mostly on its failure to reproduce an artefact while
POD at rank 32 scored well by interpolating the very snapshots it was fitted
to. Removing the band makes the metric measure mechanics.

The cut is the one already justified in the band diagnostic, not a new tuned
parameter: a Gaussian low pass at sigma 1.3 pixels, the same value that bounded
the first band there. Both models then read this file through their existing
`--history` option, so nothing else about the comparison changes and the two
remain judged identically.

Filtering is done on whole states rather than on bands, since a per-band filter
would leave seams exactly where the gradient metric looks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

DATA_ROOT = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DATA_ROOT / "p0043_evm_history.h5")
    parser.add_argument("--output", type=Path,
                        default=DATA_ROOT / "p0043_evm_history_lowpass.h5")
    parser.add_argument("--sigma", type=float, default=1.3)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()

    with h5py.File(arguments.source, "r") as source, h5py.File(arguments.output, "w") as target:
        evm = source["evm"]
        states, rows, columns = evm.shape
        dataset = target.create_dataset(
            "evm", shape=evm.shape, dtype=np.float32,
            chunks=(1, min(512, rows), min(512, columns)),
            compression="gzip", compression_opts=4,
        )
        means, retained = [], []
        for state in range(states):
            field = np.asarray(evm[state], dtype=np.float64)
            smooth = gaussian_filter(field, arguments.sigma)
            dataset[state] = smooth
            means.append(float(smooth.mean()))
            total = float((field**2).mean())
            retained.append(float((smooth**2).mean()) / max(total, 1e-30))
            if state % 10 == 0:
                print(f"  state {state}: retains {retained[-1]:.3f} of the energy", flush=True)
        noise = gaussian_filter(np.asarray(source["noise_evm"], dtype=np.float64),
                                arguments.sigma)
        target.create_dataset("noise_evm", data=noise.astype(np.float32),
                              compression="gzip", compression_opts=4)
        target.attrs["mean_evm"] = np.asarray(means)
        target.attrs["source"] = str(arguments.source)
        target.attrs["lowpass_sigma"] = arguments.sigma
        target.attrs["definition"] = (
            "equivalent strain low-passed at sigma 1.3 px, removing the 2-8 px "
            "band shown to be load-independent and therefore not mechanical"
        )

    report = {
        "schema_version": 1,
        "status": "completed_lowpass_evm_history",
        "sigma": arguments.sigma,
        "mean_evm": means,
        "energy_fraction_retained": retained,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
