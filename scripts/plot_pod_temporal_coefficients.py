#!/usr/bin/env python3
"""The temporal coefficients of the POD, which is how the morphology evolves.

The mode ordering is a property of the ensemble, not of any one state: the POD
sorts by the total energy a mode carries over the whole training set, so the
mode that dominates state 40 need not be the second one. Plotting the
coefficient trajectories settles what the single-state histogram could only
suggest.

The snapshot form gives the training trajectories with no extra work at all:
with `C = X X^T = U diag(lambda) U^T`, the coefficient of mode k at training
state i is exactly `sqrt(lambda_k) U[i, k]`. Only the held-out states need a
projection, and therefore one pass over the spatial modes.

Held-out states are drawn as markers on the same axes. If the trajectories are
smooth and the held-out markers fall on them, the reduced coordinates behave
like a low-dimensional path through the loading rather than a per-state fit --
which is the property any reduced model would need, and which the reconstruction
error alone does not show.
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
from benchmark_pod_morphology import HOLDOUT_REGIONS  # type: ignore[import-not-found]
from morphology_benchmark_split import split_states  # type: ignore[import-not-found]

DATA = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/p0043_evm_history.h5")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DATA)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--band", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    with h5py.File(arguments.history, "r") as handle:
        evm = handle["evm"]
        means = np.asarray(handle.attrs["mean_evm"], dtype=np.float64)
        indices = list(range(1, int(evm.shape[0])))
        train, test = split_states(indices)
        shape = (evm.shape[1], evm.shape[2])
        mask = np.ones(shape, dtype=bool)
        for row, column, size in HOLDOUT_REGIONS:
            mask[row : row + size, column : column + size] = False
        bands = [
            (start, min(start + arguments.band, shape[0]))
            for start in range(0, shape[0], arguments.band)
        ]

        def rows(index: int, start: int, stop: int) -> np.ndarray:
            state = indices[index]
            return np.asarray(evm[state, start:stop], dtype=np.float64) / means[state]

        centres = {}
        for start, stop in bands:
            local = mask[start:stop]
            centres[start] = sum(rows(i, start, stop)[local] for i in train) / len(train)
        correlation = np.zeros((len(train), len(train)))
        for start, stop in bands:
            local = mask[start:stop]
            block = np.stack([rows(i, start, stop)[local] - centres[start] for i in train])
            correlation += block @ block.T
        values, vectors = np.linalg.eigh(correlation)
        order = np.argsort(values)[::-1]
        values, vectors = np.maximum(values[order], 0.0), vectors[:, order]

        modes = arguments.modes
        # Training coefficients come straight out of the snapshot eigenvectors.
        training_coefficients = vectors[:, :modes] * np.sqrt(values[:modes])
        # Held-out states have to be projected, which needs the spatial modes.
        scale = 1.0 / np.sqrt(np.maximum(values[:modes], 1e-30))
        holdout_coefficients = np.zeros((len(test), modes))
        for start, stop in bands:
            local = mask[start:stop]
            block = np.stack([rows(i, start, stop)[local] - centres[start] for i in train])
            spatial = (vectors[:, :modes] * scale).T @ block
            for position, index in enumerate(test):
                holdout_coefficients[position] += spatial @ (
                    rows(index, start, stop)[local] - centres[start]
                )
            print(f"  band {start}", flush=True)

    train_states = [indices[i] for i in train]
    test_states = [indices[i] for i in test]
    share = values[:modes] / values.sum()

    figure, axes = plt.subplots(2, 1, figsize=(11.0, 8.0), constrained_layout=True,
                                height_ratios=[2, 1])
    for k in range(modes):
        line, = axes[0].plot(
            train_states, training_coefficients[:, k], "o-", markersize=4,
            label=f"mode {k + 1} ({100 * share[k]:.1f} % of the ensemble)",
        )
        axes[0].plot(
            test_states, holdout_coefficients[:, k], "D", markersize=9,
            markerfacecolor="none", markeredgewidth=1.6, color=line.get_color(),
        )
    axes[0].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[0].set_xlabel("state")
    axes[0].set_ylabel("coefficient a_k")
    axes[0].set_title(
        "POD temporal coefficients; open diamonds are the held-out states, "
        "projected rather than fitted",
        fontsize=10,
    )
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(np.arange(1, len(values) + 1), values / values.sum(), "o-")
    axes[1].set_xlabel("mode")
    axes[1].set_ylabel("share of the ensemble energy")
    axes[1].set_title("Ensemble spectrum, which is what orders the modes", fontsize=10)
    axes[1].grid(alpha=0.3)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=160)
    plt.close(figure)
    arguments.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "train_states": train_states,
                "holdout_states": test_states,
                "ensemble_energy_share": (values / values.sum()).tolist(),
                "training_coefficients": training_coefficients.tolist(),
                "holdout_coefficients": holdout_coefficients.tolist(),
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
