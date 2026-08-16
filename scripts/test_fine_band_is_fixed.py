#!/usr/bin/env python3
"""Is the 2-8 px content one fixed pattern, or does it evolve?

Constant energy in a band does not make its spatial map constant, and the
distinction decides the preprocessing. If the fine content is essentially the
same picture at every state, it should be extracted as a static term,

```text
H(x, t) = H_fixed(x) + H_evolving(x, t)
```

and only the evolving part learned -- which preserves any genuinely fine
plastic structure appearing late. If instead the maps decorrelate between
states, the band carries changing information and must not be removed at all.
Low-passing everything below eight pixels would destroy both cases
indiscriminately, which is why it is not applied before this runs.

The band is isolated as `field - gaussian(field, 1.3)`, the same cut that
bounded the first band of the energy diagnostic, and the maps are correlated
pairwise across states. A high correlation between distant states, and one that
does not decay as the states separate, is the signature of a fixed pattern.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

DATA = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/p0043_evm_history.h5")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DATA)
    parser.add_argument("--sigma", type=float, default=1.3)
    parser.add_argument("--decimate", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    step = arguments.decimate
    with h5py.File(arguments.history, "r") as handle:
        evm = handle["evm"]
        states = int(evm.shape[0])
        means = np.asarray(handle.attrs["mean_evm"], dtype=np.float64)
        bands = []
        for state in range(1, states):
            field = np.asarray(evm[state], dtype=np.float64) / means[state]
            # Decimation is applied after filtering, so the band is isolated at
            # full resolution and only the correlation is subsampled.
            detail = field - gaussian_filter(field, arguments.sigma)
            bands.append(detail[::step, ::step].astype(np.float32).ravel())
            if state % 10 == 0:
                print(f"  state {state}", flush=True)
        noise = np.asarray(handle["noise_evm"], dtype=np.float64)
        noise_detail = (noise - gaussian_filter(noise, arguments.sigma))[::step, ::step].ravel()

    matrix = np.stack(bands)
    matrix -= matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(matrix, axis=1)
    correlation = (matrix @ matrix.T) / np.outer(norms, norms)
    indices = list(range(1, states))

    # A fixed pattern would correlate as strongly between distant states as
    # between neighbours; a decaying curve means the band evolves.
    separations = {}
    for gap in range(1, len(indices)):
        values = [correlation[i, i + gap] for i in range(len(indices) - gap)]
        separations[gap] = float(np.mean(values))

    common = matrix.mean(axis=0)
    explained = float((common @ common) * len(indices) / (matrix**2).sum())
    noise_correlation = float(
        np.corrcoef(common, noise_detail - noise_detail.mean())[0, 1]
    )
    print(f"\nmean correlation at gap 1: {separations[1]:.4f}, "
          f"at gap {max(separations)}: {separations[max(separations)]:.4f}")
    print(f"static component explains {100 * explained:.1f} % of the band energy")
    print(f"static component against the null-test detail: {noise_correlation:+.4f}")

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.3), constrained_layout=True)
    image = axes[0].imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r",
                           extent=(1, states - 1, states - 1, 1))
    axes[0].set_title("correlation of the 2-8 px maps between states", fontsize=10)
    axes[0].set_xlabel("state")
    axes[0].set_ylabel("state")
    figure.colorbar(image, ax=axes[0], shrink=0.85)

    gaps = sorted(separations)
    axes[1].plot(gaps, [separations[g] for g in gaps], "o-")
    axes[1].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[1].set_xlabel("separation between states")
    axes[1].set_ylabel("mean correlation")
    axes[1].set_title("Flat means a fixed pattern, decaying means it evolves", fontsize=10)
    axes[1].grid(alpha=0.3)

    axes[2].plot(indices, [float(np.corrcoef(common, row)[0, 1]) for row in matrix], "o-")
    axes[2].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[2].set_xlabel("state")
    axes[2].set_ylabel("correlation with the across-state mean map")
    axes[2].set_title(
        f"Static component holds {100 * explained:.1f} % of the band energy", fontsize=10
    )
    axes[2].grid(alpha=0.3)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=160)
    plt.close(figure)
    arguments.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "sigma": arguments.sigma,
                "states": indices,
                "mean_correlation_by_separation": separations,
                "static_energy_fraction": explained,
                "static_versus_null_test": noise_correlation,
            },
            indent=2,
        )
        + "\n",
        "utf-8",
    )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
