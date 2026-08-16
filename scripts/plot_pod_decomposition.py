#!/usr/bin/env python3
"""How the POD actually spends its modes on the final state.

An error curve says how much is left; it does not say what each mode is doing.
This shows the decomposition itself: the first modes weighted by the
coefficients state 40 gives them, and the reconstruction as they accumulate.

The basis is the benchmark's own -- same snapshot correlation, same training
states, same excluded regions -- rebuilt band by band rather than reloaded,
since the benchmark keeps errors and not modes. Only every fourth pixel of each
mode is retained, which is a display decision and affects nothing but the
figure: the modes of a global linear basis are dominated by large scales, which
is itself part of the point being shown.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--state", type=int, default=40)
    parser.add_argument("--modes", type=int, default=5)
    parser.add_argument("--display-step", type=int, default=4)
    parser.add_argument("--band", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    view = arguments.display_step
    with h5py.File(arguments.history, "r") as handle:
        evm = handle["evm"]
        means = np.asarray(handle.attrs["mean_evm"], dtype=np.float64)
        indices = list(range(1, int(evm.shape[0])))
        train, _ = split_states(indices)
        shape = (evm.shape[1], evm.shape[2])
        mask = np.ones(shape, dtype=bool)
        for row, column, size in HOLDOUT_REGIONS:
            mask[row : row + size, column : column + size] = False

        def band_rows(index: int, start: int, stop: int) -> np.ndarray:
            state = indices[index]
            return np.asarray(evm[state, start:stop], dtype=np.float64) / means[state]

        bands = [
            (start, min(start + arguments.band, shape[0]))
            for start in range(0, shape[0], arguments.band)
        ]
        print(f"{len(train)} training states, {len(bands)} bands", flush=True)

        centres = {}
        for start, stop in bands:
            local = mask[start:stop]
            centres[start] = sum(band_rows(i, start, stop)[local] for i in train) / len(train)
        correlation = np.zeros((len(train), len(train)))
        for start, stop in bands:
            local = mask[start:stop]
            block = np.stack([band_rows(i, start, stop)[local] - centres[start] for i in train])
            correlation += block @ block.T
            print(f"  correlation band {start}", flush=True)
        values, vectors = np.linalg.eigh(correlation)
        order = np.argsort(values)[::-1]
        values, vectors = np.maximum(values[order], 0.0), vectors[:, order]

        rank = len(train)
        scale = 1.0 / np.sqrt(np.maximum(values[:rank], 1e-30))
        coefficients = np.zeros(rank)
        display_modes = []
        display_field = []
        display_centre = []
        display_mask = []
        for start, stop in bands:
            local = mask[start:stop]
            block = np.stack([band_rows(i, start, stop)[local] - centres[start] for i in train])
            modes = (vectors[:, :rank] * scale).T @ block
            observed = np.asarray(
                evm[arguments.state, start:stop], dtype=np.float64
            ) / means[arguments.state]
            coefficients += modes @ (observed[local] - centres[start])
            full = np.zeros((rank, stop - start, shape[1]))
            full[:, local] = modes
            display_modes.append(full[: arguments.modes, ::view, ::view])
            display_field.append(observed[::view, ::view])
            centre_full = np.zeros((stop - start, shape[1]))
            centre_full[local] = centres[start]
            display_centre.append(centre_full[::view, ::view])
            display_mask.append(local[::view, ::view])
            print(f"  modes band {start}", flush=True)

    modes = np.concatenate(display_modes, axis=1)
    field = np.concatenate(display_field, axis=0)
    # The reconstruction is the mean field plus the mode contributions; showing
    # the sum of modes alone against the measurement compares a fluctuation
    # with a field and makes every partial sum look empty.
    centre_image = np.concatenate(display_centre, axis=0)
    valid = np.concatenate(display_mask, axis=0)
    energy = coefficients**2
    print(f"first five coefficients: {np.round(coefficients[:5], 1)}", flush=True)

    ranks = [1, 2, 4, 8, min(16, rank)]
    figure = plt.figure(figsize=(15.5, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(3, arguments.modes + 1)

    span = float(np.percentile(np.abs(modes[0][valid]), 99.0))
    for k in range(arguments.modes):
        axis = figure.add_subplot(grid[0, k])
        contribution = coefficients[k] * modes[k]
        limit = float(np.percentile(np.abs(contribution[valid]), 99.0))
        axis.imshow(contribution.T, vmin=-limit, vmax=limit, cmap="RdBu_r", origin="lower")
        axis.set_title(f"mode {k + 1} x a{k + 1}\n{100 * energy[k] / energy.sum():.1f} % of energy",
                       fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    del span

    axis = figure.add_subplot(grid[0, arguments.modes])
    axis.semilogy(np.arange(1, rank + 1), np.maximum(energy, 1e-30) / energy.sum(), "o-")
    axis.set_xlabel("mode")
    axis.set_ylabel("share of the coefficient energy")
    axis.set_title(f"state {arguments.state} spectrum", fontsize=9)
    axis.grid(alpha=0.3)

    ceiling = float(np.percentile(field[valid], 99.0))
    floor = float(np.percentile(field[valid], 1.0))
    partial = centre_image.copy()
    previous = 0
    for position, target in enumerate(ranks):
        for k in range(previous, target):
            partial = partial + coefficients[k] * modes[k] if k < arguments.modes else partial
        previous = target
        axis = figure.add_subplot(grid[1, position])
        axis.imshow(partial.T, vmin=floor, vmax=ceiling, cmap="inferno", origin="lower")
        axis.set_title(f"mean + first {target} modes", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    axis = figure.add_subplot(grid[1, arguments.modes])
    image = axis.imshow(field.T, vmin=floor, vmax=ceiling, cmap="inferno", origin="lower")
    axis.set_title(f"measured, state {arguments.state}", fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    figure.colorbar(image, ax=axis, shrink=0.8)

    axis = figure.add_subplot(grid[2, :])
    axis.text(
        0.01,
        0.5,
        "Reconstructions are the mean field plus the mode contributions. Only the "
        f"first {arguments.modes} modes are drawn, so the partial sums stop there; "
        "the coefficient spectrum shows how much is left beyond them. Modes are "
        "displayed at every fourth pixel, a figure decision only. Grey areas are "
        "the two spatial holdout regions, excluded from the fit, where a global "
        "linear basis is undefined by construction.",
        fontsize=10,
        va="center",
        wrap=True,
    )
    axis.axis("off")
    figure.suptitle(
        f"POD decomposition of state {arguments.state}, basis fitted on "
        f"{len(train)} states outside the temporal holdout",
        fontsize=11,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=150)
    plt.close(figure)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
