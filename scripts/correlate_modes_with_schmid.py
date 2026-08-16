#!/usr/bin/env python3
"""Do the POD modes, and the measured strain itself, follow the Schmid factor?

A curiosity run, outside the milestone, and worth its cost: if the morphology
the reduced basis describes lines up with the crystallography, a later tensor
decoder would be better parameterised by slip systems than by learned patterns.

Two correlations, and the first needs no model at all:

```text
corr(EVM_t, S)      does strain concentrate where slip is easiest, and when
corr(Phi_k, S)      does any individual mode carry that alignment
```

Three caveats travel with any number produced here, and none of them is
rhetorical.

The Schmid factor predicts the *initial* yield of an isolated grain under
uniaxial load. At half a percent of mean strain, grain interactions dominate,
so a weak correlation would not refute crystal plasticity -- it would only say
that a single-crystal criterion does not survive into the polycrystal.

`max_schmid_factor` collapses twelve systems into one scalar, discarding which
system and in what direction. The orientations are the richer object.

The inventory records the EBSD and DIC arrays as "declared co-registered after
cropping; method absent". For a spatial correlation the registration *is* the
measurement, so an unverified alignment is the dominant uncertainty here. A
shift scan is included for that reason: if the correlation peaks away from zero
offset, the registration is wrong and every number moves.
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
CRYSTAL = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DATA)
    parser.add_argument("--crystal", type=Path, default=CRYSTAL)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--band", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    with h5py.File(arguments.crystal, "r") as handle:
        schmid_full = np.asarray(handle["schmid/max_schmid_factor"], dtype=np.float64)
    print(f"Schmid map {schmid_full.shape}, "
          f"range {schmid_full.min():.3f} to {schmid_full.max():.3f}", flush=True)

    with h5py.File(arguments.history, "r") as handle:
        evm = handle["evm"]
        means = np.asarray(handle.attrs["mean_evm"], dtype=np.float64)
        indices = list(range(1, int(evm.shape[0])))
        train, _ = split_states(indices)
        shape = (evm.shape[1], evm.shape[2])
        # The EVM grid is one cell smaller than the displacement grid because it
        # is built from first differences; the Schmid map is trimmed to match
        # rather than resampled, so no interpolation enters the correlation.
        schmid = schmid_full[: shape[0], : shape[1]]
        # A Schmid factor cannot exceed 0.5. A handful of pixels carry values up
        # to 1449, and a single one of those among ten million dominates the
        # variance completely -- it drove the first run's correlations to zero.
        # They are excluded rather than clipped, since their true value is
        # unknown.
        valid = (schmid > 0.0) & (schmid <= 0.5)
        print(f"excluding {(~valid).sum()} pixels outside (0, 0.5]", flush=True)
        mask = valid.copy()
        for row, column, size in HOLDOUT_REGIONS:
            mask[row : row + size, column : column + size] = False
        print(f"{mask.sum()} pixels retained", flush=True)
        bands = [
            (start, min(start + arguments.band, shape[0]))
            for start in range(0, shape[0], arguments.band)
        ]

        def rows(index: int, start: int, stop: int) -> np.ndarray:
            state = indices[index]
            return np.asarray(evm[state, start:stop], dtype=np.float64) / means[state]

        # Model-free first: the correlation of the measured field with Schmid,
        # state by state, accumulated as sums so nothing full-field is held.
        schmid_sum = float(schmid[mask].sum())
        schmid_square = float((schmid[mask] ** 2).sum())
        count = int(mask.sum())
        per_state = []
        state_deviations = []
        for position, state in enumerate(indices):
            total = cross = square = 0.0
            for start, stop in bands:
                local = mask[start:stop]
                values = rows(position, start, stop)[local]
                total += float(values.sum())
                square += float((values**2).sum())
                cross += float((values * schmid[start:stop][local]).sum())
            numerator = cross - total * schmid_sum / count
            denominator = np.sqrt(
                max(square - total**2 / count, 1e-30)
                * max(schmid_square - schmid_sum**2 / count, 1e-30)
            )
            per_state.append(float(numerator / denominator))
            state_deviations.append(
                float(np.sqrt(max(square / count - (total / count) ** 2, 1e-30)))
            )
            if state % 10 == 0:
                print(f"  state {state}: corr(EVM, Schmid) {per_state[-1]:+.4f}", flush=True)

        centres = {}
        for start, stop in bands:
            local = mask[start:stop]
            centres[start] = sum(rows(i, start, stop)[local] for i in train) / len(train)
        correlation = np.zeros((len(train), len(train)))
        for start, stop in bands:
            local = mask[start:stop]
            block = np.stack([rows(i, start, stop)[local] - centres[start] for i in train])
            correlation += block @ block.T
        values_, vectors = np.linalg.eigh(correlation)
        order = np.argsort(values_)[::-1]
        values_, vectors = np.maximum(values_[order], 0.0), vectors[:, order]

        modes = arguments.modes
        scale = 1.0 / np.sqrt(np.maximum(values_[:modes], 1e-30))
        mode_sum = np.zeros(modes)
        mode_square = np.zeros(modes)
        mode_cross = np.zeros(modes)
        # A mode is a fixed picture, so corr(Phi_k, S) has no time dependence.
        # What evolves is each mode's contribution, and covariance is linear:
        #   cov(EVM_t, S) = cov(centre, S) + sum_k a_k(t) cov(Phi_k, S).
        # Accumulating the coefficients of every state, held-out ones included,
        # therefore decomposes the rising correlation curve term by term.
        coefficients = np.zeros((len(indices), modes))
        centre_cross = centre_sum = 0.0
        for start, stop in bands:
            local = mask[start:stop]
            block = np.stack([rows(i, start, stop)[local] - centres[start] for i in train])
            spatial = (vectors[:, :modes] * scale).T @ block
            local_schmid = schmid[start:stop][local]
            mode_sum += spatial.sum(axis=1)
            mode_square += (spatial**2).sum(axis=1)
            mode_cross += spatial @ local_schmid
            centre_cross += float(centres[start] @ local_schmid)
            centre_sum += float(centres[start].sum())
            for position in range(len(indices)):
                coefficients[position] += spatial @ (
                    rows(position, start, stop)[local] - centres[start]
                )
        numerator = mode_cross - mode_sum * schmid_sum / count
        denominator = np.sqrt(
            np.maximum(mode_square - mode_sum**2 / count, 1e-30)
            * max(schmid_square - schmid_sum**2 / count, 1e-30)
        )
        mode_correlation = numerator / denominator

    mode_covariance = mode_cross / count - (mode_sum / count) * (schmid_sum / count)
    schmid_deviation = np.sqrt(max(schmid_square / count - (schmid_sum / count) ** 2, 1e-30))
    state_deviation = np.asarray(state_deviations)
    contributions = (
        coefficients * mode_covariance[None, :]
        / (state_deviation[:, None] * schmid_deviation)
    )
    share = values_[:modes] / values_.sum()
    print(f"\nmode correlations with Schmid: {np.round(mode_correlation, 4)}", flush=True)

    figure, axes = plt.subplots(1, 3, figsize=(17.5, 4.6), constrained_layout=True)
    axes[0].plot(indices, per_state, "o-")
    axes[0].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[0].set_xlabel("state")
    axes[0].set_ylabel("corr(EVM, max Schmid factor)")
    axes[0].set_title("Measured field against the Schmid map, no model", fontsize=10)
    axes[0].grid(alpha=0.3)

    positions = np.arange(1, modes + 1)
    bars = axes[1].bar(positions, mode_correlation, color="tab:blue")
    for position, value, weight in zip(positions, mode_correlation, share, strict=True):
        axes[1].text(position, value, f"{100 * weight:.0f} %", ha="center",
                     va="bottom" if value >= 0 else "top", fontsize=8)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel("POD mode (label: share of the ensemble energy)")
    axes[1].set_ylabel("corr(mode, max Schmid factor)")
    axes[1].set_title("Individual modes against the Schmid map", fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")
    del bars

    for k in range(modes):
        axes[2].plot(indices, contributions[:, k], "o-", markersize=3,
                     label=f"mode {k + 1}")
    axes[2].plot(indices, contributions.sum(axis=1), "k--", lw=2,
                 label="sum of the six")
    axes[2].plot(indices, per_state, "k-", lw=1.2, alpha=0.6, label="measured total")
    axes[2].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[2].set_xlabel("state")
    axes[2].set_ylabel("contribution to corr(EVM, Schmid)")
    axes[2].set_title("Where the rising correlation comes from", fontsize=10)
    axes[2].legend(fontsize=7, ncol=2)
    axes[2].grid(alpha=0.3)

    figure.suptitle(
        "Schmid factor against the measured morphology. Registration is declared, "
        "not verified; a single-crystal criterion is not expected to survive intact "
        "into a polycrystal at this strain.",
        fontsize=10,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=160)
    plt.close(figure)
    arguments.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "states": indices,
                "evm_schmid_correlation": per_state,
                "mode_schmid_correlation": mode_correlation.tolist(),
                "mode_energy_share": share.tolist(),
                "mode_contribution_per_state": contributions.tolist(),
                "mode_covariance_with_schmid": mode_covariance.tolist(),
                "caveats": [
                    "EBSD/DIC registration is declared in the inventory but its "
                    "method is absent, and for a spatial correlation the "
                    "registration is the measurement",
                    "max_schmid_factor collapses twelve systems into one scalar",
                    "the Schmid factor predicts initial single-crystal yield, not "
                    "polycrystal behaviour at half a percent of mean strain",
                ],
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
