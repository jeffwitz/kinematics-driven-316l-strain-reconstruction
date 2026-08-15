#!/usr/bin/env python3
"""POD baseline: how compact is a global linear basis for the EVM morphology?

The mandatory reference the convolutional model has to beat. It answers one
question -- how many linear coefficients per state are needed to reconstruct
the measured morphology -- and it answers it on exactly the data, the
normalisation, and the holdouts the network will face, so the two numbers are
comparable rather than merely coexisting.

The normalisation is fixed here and not revisited: `H_t = EVM_t / m_t`, with
`m_t` the spatial mean. It removes the load level, which is otherwise the first
thing any model spends capacity on, and it keeps the round trip to the physical
field exact through the stored `m_t`.

## What POD can and cannot be asked

A POD mode is a fixed picture of the specimen, so the basis is tied to absolute
positions. It can be asked to generalise **in time**, to a state it never saw,
and that is a fair test. It cannot be asked to generalise **in space**: modes
fitted where the training pixels are say nothing about a region excluded from
the fit. That asymmetry is not a flaw in the experiment, it is the structural
claim the convolutional route rests on, so the spatial holdout is reported as
undefined for POD rather than filled with a meaningless number.

Two error measures are kept, because reconstruction error alone rewards
smoothing: the field error and a gradient error, which a blurred reconstruction
cannot fake.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from morphology_benchmark_split import split_states  # type: ignore[import-not-found]
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

DATA = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/p0043_evm_history.h5")
#: Two square regions never used for fitting, quoted in pixels of the EVM grid.
#: They sit away from the crop border, where the correlation window is
#: one-sided, and away from each other so that a lucky local texture cannot
#: carry both.
HOLDOUT_REGIONS = ((600, 600, 600), (2400, 2100, 600))


def _relative(candidate: FloatArray, reference: FloatArray) -> float:
    return float(np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), 1e-30))


def _gradient_error(candidate: FloatArray, reference: FloatArray) -> float:
    """Error on the first differences, which a smoothed reconstruction cannot fake."""

    errors, norms = 0.0, 0.0
    for axis in (0, 1):
        difference = np.diff(candidate, axis=axis) - np.diff(reference, axis=axis)
        errors += float((difference**2).sum())
        norms += float((np.diff(reference, axis=axis) ** 2).sum())
    return float(np.sqrt(errors / max(norms, 1e-30)))


#: The window the convolutional benchmark evaluates on, in full-resolution
#: pixels, kept here so both scripts quote the same region.
WINDOW_CORNER = (0, 700)
WINDOW_SIDE = 600


def _window_errors(fields, centre_bands, basis_band, mask, subset, corner, side, step,
                   rank, shape):
    """POD error on the window the convolutional benchmark reports on.

    Comparing a domain-wide POD error against a network error measured on one
    window compares different populations of pixels, and a locally easy or hard
    region would then decide the verdict. The gradient error needs the field
    laid out in two dimensions, so this one window is reassembled -- it is
    small.
    """

    row, column = corner[0] // step, corner[1] // step
    extent = side // step
    if not mask[row : row + extent, column : column + extent].all():
        raise SystemExit("the comparison window overlaps a spatial holdout")
    coefficients = {index: np.zeros(rank) for index in subset}
    for start, stop in fields.bands():
        modes = basis_band(rank, start, stop)
        local = mask[start:stop]
        for index in subset:
            coefficients[index] += modes @ (
                fields.rows(index, start, stop)[local] - centre_bands[start]
            )
    errors, gradients = [], []
    for index in subset:
        reconstruction = np.zeros((extent, extent))
        reference = np.zeros((extent, extent))
        for start, stop in fields.bands():
            overlap = slice(max(start, row), min(stop, row + extent))
            if overlap.start >= overlap.stop:
                continue
            modes = basis_band(rank, start, stop)
            local = mask[start:stop]
            band = np.zeros((stop - start, shape[1]))
            band[local] = coefficients[index] @ modes + centre_bands[start]
            observed = np.zeros((stop - start, shape[1]))
            observed[local] = fields.rows(index, start, stop)[local]
            rows = slice(overlap.start - start, overlap.stop - start)
            target = slice(overlap.start - row, overlap.stop - row)
            reconstruction[target] = band[rows, column : column + extent]
            reference[target] = observed[rows, column : column + extent]
        errors.append(_relative(reconstruction, reference))
        gradients.append(_gradient_error(reconstruction, reference))
    return {
        "field_error": float(np.mean(errors)),
        "gradient_error": float(np.mean(gradients)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DATA)
    parser.add_argument("--ranks", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--decimate", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    step = arguments.decimate
    handle = h5py.File(arguments.history, "r")
    evm = handle["evm"]
    states = int(evm.shape[0])
    means = np.asarray(handle.attrs["mean_evm"], dtype=np.float64)
    # State 0 is the undeformed reference and its mean is zero, so it has no
    # normalised morphology; the history proper starts at one.
    indices = list(range(1, states))
    shape = (evm.shape[1] // step + (evm.shape[1] % step > 0),
             evm.shape[2] // step + (evm.shape[2] % step > 0))
    print(f"{len(indices)} states, shape {shape}", flush=True)

    class Fields:
        """Normalised morphology maps, read in row bands rather than stacked.

        With forty states and eleven million pixels the snapshot form of the
        POD needs only a forty-by-forty correlation matrix in memory; stacking
        the fields to get it is what exhausted the RAM, not the method. Bands
        are cached so a single pass over the states costs one read each.
        """

        def __init__(self, band: int = 400) -> None:
            self.band = band
            self._cache: dict[tuple[int, int], np.ndarray] = {}

        def rows(self, index: int, start: int, stop: int) -> np.ndarray:
            key = (index, start)
            if key not in self._cache:
                self._cache.clear()
                state = indices[index]
                block = np.asarray(
                    evm[state, start * step : stop * step : step], dtype=np.float64
                )
                self._cache[key] = block / means[state]
            return self._cache[key]

        def bands(self):
            for start in range(0, shape[0], self.band):
                yield start, min(start + self.band, shape[0])

    fields = Fields()

    mask = np.ones(shape, dtype=bool)
    for row, column, size in HOLDOUT_REGIONS:
        mask[row // step : (row + size) // step, column // step : (column + size) // step] = False
    print(f"spatial holdout covers {100.0 * (1.0 - mask.mean()):.1f} % of the field", flush=True)

    train, test = split_states(indices)
    print(f"train {len(train)} states, temporal holdout "
          f"{[indices[i] for i in test]}", flush=True)

    # Snapshot POD: accumulate the state-by-state correlation matrix band by
    # band, so only a small square matrix is ever resident. The mean field is
    # accumulated the same way.
    count = len(train)
    centre_bands: dict[int, np.ndarray] = {}
    for start, stop in fields.bands():
        local = mask[start:stop]
        centre_bands[start] = sum(
            fields.rows(index, start, stop)[local] for index in train
        ) / count
    correlation = np.zeros((count, count))
    for start, stop in fields.bands():
        local = mask[start:stop]
        block = np.stack(
            [fields.rows(index, start, stop)[local] - centre_bands[start] for index in train]
        )
        correlation += block @ block.T
    values, vectors = np.linalg.eigh(correlation)
    order = np.argsort(values)[::-1]
    values, vectors = np.maximum(values[order], 0.0), vectors[:, order]
    pixels = int(mask.sum())
    print(f"snapshot correlation {correlation.shape}, {pixels} fitted pixels", flush=True)

    def basis_band(rank: int, start: int, stop: int) -> np.ndarray:
        """Spatial modes over one band, rebuilt from the snapshot vectors."""

        local = mask[start:stop]
        block = np.stack(
            [fields.rows(index, start, stop)[local] - centre_bands[start] for index in train]
        )
        scale = 1.0 / np.sqrt(np.maximum(values[:rank], 1.0e-30))
        return (vectors[:, :rank] * scale).T @ block

    results = []
    for rank in arguments.ranks:
        if rank > count:
            continue
        entry: dict[str, object] = {
            "rank": rank,
            "coefficients_per_state": rank,
            "model_parameters": int((rank + 1) * pixels),
        }
        # Projection coefficients are accumulated over bands, then the squared
        # errors are accumulated in a second pass; nothing full-field is held.
        for label, subset in (("train_states", train), ("temporal_holdout", test)):
            coefficients = {index: np.zeros(rank) for index in subset}
            for start, stop in fields.bands():
                modes = basis_band(rank, start, stop)
                local = mask[start:stop]
                for index in subset:
                    residual = fields.rows(index, start, stop)[local] - centre_bands[start]
                    coefficients[index] += modes @ residual
            errors = np.zeros(len(subset))
            norms = np.zeros(len(subset))
            for start, stop in fields.bands():
                modes = basis_band(rank, start, stop)
                local = mask[start:stop]
                for position, index in enumerate(subset):
                    observed = fields.rows(index, start, stop)[local] - centre_bands[start]
                    difference = coefficients[index] @ modes - observed
                    errors[position] += float((difference**2).sum())
                    norms[position] += float(
                        ((observed + centre_bands[start]) ** 2).sum()
                    )
            entry[label] = {
                "field_error": float(np.mean(np.sqrt(errors / np.maximum(norms, 1e-30)))),
            }
        entry["seen_window"] = _window_errors(
            fields, centre_bands, basis_band, mask, train + test,
            WINDOW_CORNER, WINDOW_SIDE, step, rank, shape,
        )
        entry["spatial_holdout"] = (
            "undefined: POD modes are tied to absolute positions and say nothing "
            "about pixels excluded from the fit"
        )
        results.append(entry)
        print(
            f"  rank {rank:3d}: train {entry['train_states']['field_error']:.4f}  "
            f"temporal holdout {entry['temporal_holdout']['field_error']:.4f}  "
            f"window {entry['seen_window']['field_error']:.4f} "
            f"(gradient {entry['seen_window']['gradient_error']:.4f})",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "status": "completed_pod_morphology_baseline",
        "normalisation": "H_t = EVM_t / mean(EVM_t), frozen",
        "history": str(arguments.history),
        "decimation": step,
        "field_shape": list(shape),
        "states": indices,
        "train_states": [indices[index] for index in train],
        "temporal_holdout_states": [indices[index] for index in test],
        "split_rationale": (
            "the holdout is spread over the loading in two-state blocks so "
            "training spans every mechanical regime; withholding states 31-40 "
            "would test out-of-distribution extrapolation instead of the "
            "compactness of the representation"
        ),
        "spatial_holdout_regions": [list(region) for region in HOLDOUT_REGIONS],
        "spatial_holdout_fraction": float(1.0 - mask.mean()),
        "pixels_in_fit": pixels,
        "model_parameter_note": (
            "a POD mode carries one value per pixel, so the model cost is "
            "(rank + 1) times the pixel count, the mean field included; it is "
            "reported per rank in the results because a single number that "
            "does not depend on the rank is not a model cost"
        ),
        "results": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    handle.close()
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
