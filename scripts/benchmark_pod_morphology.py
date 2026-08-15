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


def _window_errors(fields, centre, basis, mask, subset, corner, side, step):
    """POD reconstruction error restricted to one window of the fitted domain."""

    row, column = corner[0] // step, corner[1] // step
    extent = side // step
    shape = fields.shape[1:]
    window = np.zeros(shape, dtype=bool)
    window[row : row + extent, column : column + extent] = True
    if not mask[window].all():
        raise SystemExit("the comparison window overlaps a spatial holdout")
    selector = window[mask]
    errors, gradients = [], []
    for index in subset:
        observed = fields[index][mask]
        reconstruction = centre + (basis @ (observed - centre)) @ basis
        errors.append(_relative(reconstruction[selector], observed[selector]))
        full = np.zeros(shape)
        reference = np.zeros(shape)
        full[mask] = reconstruction
        reference[mask] = observed
        gradients.append(
            _gradient_error(
                full[row : row + extent, column : column + extent],
                reference[row : row + extent, column : column + extent],
            )
        )
    return {
        "field_error": float(np.mean(errors)),
        "gradient_error": float(np.mean(gradients)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DATA)
    parser.add_argument("--ranks", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--decimate", type=int, default=3)
    parser.add_argument("--train-states", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    step = arguments.decimate
    with h5py.File(arguments.history, "r") as handle:
        evm = handle["evm"]
        states = int(evm.shape[0])
        means = np.asarray(handle.attrs["mean_evm"], dtype=np.float64)
        # State 0 is the undeformed reference and its mean is zero, so it has no
        # normalised morphology; the history proper starts at one.
        indices = list(range(1, states))
        fields = np.stack(
            [np.asarray(evm[state][::step, ::step], dtype=np.float64) / means[state]
             for state in indices]
        )
    shape = fields.shape[1:]
    print(f"{len(indices)} states, decimated shape {shape}", flush=True)

    mask = np.ones(shape, dtype=bool)
    for row, column, size in HOLDOUT_REGIONS:
        mask[row // step : (row + size) // step, column // step : (column + size) // step] = False
    print(f"spatial holdout covers {100.0 * (1.0 - mask.mean()):.1f} % of the field", flush=True)

    train = [index for index, state in enumerate(indices) if state <= arguments.train_states]
    test = [index for index, state in enumerate(indices) if state > arguments.train_states]

    # The basis is fitted on the training states over the training pixels only,
    # so a temporal holdout is not contaminated by the states it must predict.
    matrix = fields[train][:, mask]
    centre = matrix.mean(axis=0)
    _, _, right = np.linalg.svd(matrix - centre, full_matrices=False)

    results = []
    for rank in arguments.ranks:
        if rank > right.shape[0]:
            continue
        basis = right[:rank]
        entry: dict[str, object] = {
            "rank": rank,
            "coefficients_per_state": rank,
            "model_parameters": int((rank + 1) * right.shape[1]),
        }
        for label, subset in (("train_states", train), ("temporal_holdout", test)):
            errors, gradients = [], []
            for index in subset:
                observed = fields[index][mask]
                coefficients = basis @ (observed - centre)
                reconstruction = centre + coefficients @ basis
                errors.append(_relative(reconstruction, observed))
                full = np.zeros(shape)
                full[mask] = reconstruction
                reference = np.zeros(shape)
                reference[mask] = observed
                gradients.append(_gradient_error(full, reference))
            entry[label] = {
                "field_error": float(np.mean(errors)),
                "gradient_error": float(np.mean(gradients)),
            }
        # The same windows the convolutional benchmark reports on. Comparing a
        # domain-wide POD error against a CNN error measured on one window
        # compares different populations of pixels, and a locally easy or hard
        # region would then decide the verdict.
        entry["seen_window"] = _window_errors(fields, centre, basis, mask, train + test,
                                              WINDOW_CORNER, WINDOW_SIDE, step)
        entry["spatial_holdout"] = (
            "undefined: POD modes are tied to absolute positions and say nothing "
            "about pixels excluded from the fit"
        )
        results.append(entry)
        print(
            f"  rank {rank:3d}: train {entry['train_states']['field_error']:.4f}  "
            f"temporal holdout {entry['temporal_holdout']['field_error']:.4f}  "
            f"(gradient {entry['temporal_holdout']['gradient_error']:.4f})",
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
        "spatial_holdout_regions": [list(region) for region in HOLDOUT_REGIONS],
        "spatial_holdout_fraction": float(1.0 - mask.mean()),
        "pixels_in_fit": int(right.shape[1]),
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
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
