#!/usr/bin/env python3
"""Where does the dimension actually live? POD band by band.

A single reduced space has been assumed for phenomena that need not share an
origin or a compressibility. Two architectures have now returned a gradient
error near one, a number that collapses every scale into one verdict and says
nothing about which scale is responsible. This measures it instead.

The maps are split by a reconstructible multiresolution decomposition, the
Laplacian-pyramid form `L_s = G_s - G_{s+1}` with the coarsest low pass kept, so
the four bands sum back to the field exactly:

```text
H = H(2-8) + H(8-32) + H(32-128) + H(>128)
```

and a POD is fitted to each band separately, on the same training states and
the same temporal holdout. The result is an intrinsic dimension per scale
rather than one global curve. If the coarse bands need three modes and the
finest needs thirty, no architecture has to discover that -- and forcing one
latent to carry both was the mistake.

Nothing is filtered away. The 2-8 band is measured, not discarded: its
compressibility is the question, and the band-pass correlation test already
showed half of its energy evolves with the loading.

## Why only a Gram matrix is needed

Snapshot POD makes the whole thing cheap. With the centred states as rows of X
and `C = X X^T` over all forty states, the training eigenproblem is a submatrix
of C, and the coefficient of a held-out state on mode k follows from the same
matrix,

```text
a_k(j) = U[:, k] . C[train, j] / sqrt(lambda_k),
```

so the holdout error is exact without ever reconstructing a spatial mode. One
pass per band over the data, and a 40x40 matrix in memory.

Row bands are read with a halo and trimmed after filtering, since a Gaussian
applied per band would otherwise leave seams exactly where the fine bands are
measured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from benchmark_pod_morphology import HOLDOUT_REGIONS  # type: ignore[import-not-found]
from morphology_benchmark_split import split_states  # type: ignore[import-not-found]
from scipy.ndimage import gaussian_filter

DATA = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/p0043_evm_history.h5")
#: Sigmas bounding 8, 32 and 128 pixel wavelengths, as in the energy diagnostic.
BAND_SIGMAS = (1.3, 5.1, 20.4)
BAND_NAMES = ("2-8 px", "8-32 px", "32-128 px", "above 128 px")
HALO = 80


def band_of(field: np.ndarray, index: int) -> np.ndarray:
    """One Laplacian-pyramid band; the four sum back to `field` exactly."""

    if index == 0:
        return field - gaussian_filter(field, BAND_SIGMAS[0])
    if index == len(BAND_SIGMAS):
        return gaussian_filter(field, BAND_SIGMAS[-1])
    return gaussian_filter(field, BAND_SIGMAS[index - 1]) - gaussian_filter(
        field, BAND_SIGMAS[index]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DATA)
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 31])
    parser.add_argument("--band-rows", type=int, default=400)
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
        strips = [
            (start, min(start + arguments.band_rows, shape[0]))
            for start in range(0, shape[0], arguments.band_rows)
        ]

        def band_rows_of(state_index: int, start: int, stop: int, band: int) -> np.ndarray:
            low, high = max(0, start - HALO), min(shape[0], stop + HALO)
            state = indices[state_index]
            block = np.asarray(evm[state, low:high], dtype=np.float64) / means[state]
            filtered = band_of(block, band)
            return filtered[start - low : stop - low]

        results = []
        for band in range(len(BAND_SIGMAS) + 1):
            centre = {}
            for start, stop in strips:
                local = mask[start:stop]
                centre[start] = sum(
                    band_rows_of(i, start, stop, band)[local] for i in train
                ) / len(train)
            gram = np.zeros((len(indices), len(indices)))
            for start, stop in strips:
                local = mask[start:stop]
                block = np.stack(
                    [band_rows_of(i, start, stop, band)[local] - centre[start]
                     for i in range(len(indices))]
                )
                gram += block @ block.T
            print(f"{BAND_NAMES[band]}: Gram assembled", flush=True)

            sub = gram[np.ix_(train, train)]
            values, vectors = np.linalg.eigh(sub)
            order = np.argsort(values)[::-1]
            values, vectors = np.maximum(values[order], 0.0), vectors[:, order]
            total = values.sum()
            entry = {"band": BAND_NAMES[band], "eigenvalues": values.tolist(),
                     "errors": {}}
            for rank in arguments.ranks:
                if rank > len(train):
                    continue
                train_error = float(np.sqrt(max(values[rank:].sum(), 0.0) / max(total, 1e-30)))
                projected = []
                for j in test:
                    cross = gram[train, j]
                    coefficients = (vectors[:, :rank].T @ cross) / np.sqrt(
                        np.maximum(values[:rank], 1e-30)
                    )
                    residual = gram[j, j] - float((coefficients**2).sum())
                    projected.append(np.sqrt(max(residual, 0.0) / max(gram[j, j], 1e-30)))
                entry["errors"][str(rank)] = {
                    "train": train_error,
                    "temporal_holdout": float(np.mean(projected)),
                }
            results.append(entry)
            summary = "  ".join(
                f"r{r}: {entry['errors'][str(r)]['temporal_holdout']:.3f}"
                for r in arguments.ranks if str(r) in entry["errors"]
            )
            print(f"  holdout {summary}", flush=True)

    report = {
        "schema_version": 1,
        "status": "completed_pod_per_scale",
        "bands": list(BAND_NAMES),
        "band_sigmas": list(BAND_SIGMAS),
        "train_states": [indices[i] for i in train],
        "temporal_holdout_states": [indices[i] for i in test],
        "results": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
