#!/usr/bin/env python3
"""Convolutional autoencoder against the POD baseline, on the same holdouts.

The question is compactness: does a shared convolutional generator describe the
measured morphology with far fewer numbers per state than a global linear basis
does, and does it keep doing so on a region and on states it never saw?

The architecture is deliberately small and entirely convolutional -- no
attention, no pretrained weights, no U-Net skips, since a skip connection would
let the decoder copy the input around the bottleneck and make the latent size
meaningless. The latent is spatial, `(H/s, W/s, d)`, not one global vector: a
handful of numbers cannot describe a specimen, and the claim being tested is
that local patterns recur, which is what a shared decoder over a coarse latent
grid expresses.

Patches are samples of one large field used to fit shared weights, never
independent experiments. The model stays fully convolutional so it applies to
any size at evaluation, and evaluation runs on whole holdout regions rather
than on patches, with a halo trimmed so that patch borders never enter a
reported number.

Three holdouts, and the third is the one that matters: a region excluded from
training, states excluded from training, and both at once. Reconstruction of
seen regions at seen states measures compression; the last measures whether a
generator was learned.

Two error measures, field and gradient, because the POD baseline reaches a
field error of 0.18 on the temporal holdout while its gradient error stays near
0.76 -- it captures the smooth part and none of the texture. Beating it on the
field error alone would prove nothing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from benchmark_pod_morphology import (  # type: ignore[import-not-found]
    HOLDOUT_REGIONS,
    _gradient_error,
    _relative,
)
from morphology_benchmark_split import split_states  # type: ignore[import-not-found]
from torch import nn

DATA = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/p0043_evm_history.h5")


class ConvolutionalAutoencoder(nn.Module):
    """Fully convolutional, stride-2 encoder and mirrored decoder.

    `GroupNorm` rather than `BatchNorm`: batch statistics would couple patches
    that are only samples of one field, and would behave differently at
    evaluation where whole regions are pushed through at batch size one.
    """

    def __init__(self, *, downsampling: int, latent_channels: int, width: int = 24) -> None:
        super().__init__()
        depth = int(np.log2(downsampling))
        if 2**depth != downsampling:
            raise ValueError("downsampling must be a power of two")
        encoder: list[nn.Module] = []
        channels = 1
        for level in range(depth):
            out = width * (2 ** min(level, 2))
            encoder += [
                nn.Conv2d(channels, out, 4, stride=2, padding=1),
                nn.GroupNorm(4, out),
                nn.GELU(),
            ]
            channels = out
        encoder.append(nn.Conv2d(channels, latent_channels, 3, padding=1))
        self.encoder = nn.Sequential(*encoder)

        decoder: list[nn.Module] = []
        channels = latent_channels
        for level in reversed(range(depth)):
            out = width * (2 ** min(level, 2))
            decoder += [
                nn.ConvTranspose2d(channels, out, 4, stride=2, padding=1),
                nn.GroupNorm(4, out),
                nn.GELU(),
            ]
            channels = out
        decoder.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))


def _training_mask(shape: tuple[int, int], step: int) -> np.ndarray:
    """Holdout regions are quoted in full-resolution pixels, the grid is decimated.

    Forgetting the conversion silently triples the excluded area, which makes
    the spatial holdout look harder than it is and steals training data.
    """

    mask = np.ones(shape, dtype=bool)
    for row, column, size in HOLDOUT_REGIONS:
        mask[row // step : (row + size) // step, column // step : (column + size) // step] = False
    return mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DATA)
    parser.add_argument("--downsampling", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3000)
    # Full resolution by default. The spectra showed signal above noise down to a
    # two-pixel wavelength, and taking every third pixel with no anti-aliasing
    # filter folds everything between two and six pixels back into the low
    # frequencies -- destroying exactly the texture whose representability is
    # the question. Training is on patches anyway, so decimation buys nothing.
    parser.add_argument("--decimate", type=int, default=1)
    # Scaled so the two terms are of comparable magnitude on this data rather
    # than tuned; the point is that the gradient is optimised at all, not that
    # the trade-off is optimal.
    parser.add_argument("--gradient-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=None)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    generator = np.random.default_rng(arguments.seed)
    step = arguments.decimate

    with h5py.File(arguments.history, "r") as handle:
        evm = handle["evm"]
        states = int(evm.shape[0])
        means = np.asarray(handle.attrs["mean_evm"], dtype=np.float64)
        indices = list(range(1, states))
        fields = np.stack(
            [
                (np.asarray(evm[state][::step, ::step], dtype=np.float32)
                 / np.float32(means[state]))
                for state in indices
            ]
        )
    shape = fields.shape[1:]
    mask = _training_mask(shape, step)
    train_states, test_states = split_states(indices)
    print(f"{len(indices)} states, shape {shape}, holdout "
          f"{100 * (1 - mask.mean()):.1f} % of pixels", flush=True)

    model = ConvolutionalAutoencoder(
        downsampling=arguments.downsampling,
        latent_channels=arguments.latent_channels,
        width=arguments.width,
    )
    parameters = sum(p.numel() for p in model.parameters())
    decoder_parameters = sum(p.numel() for p in model.decoder.parameters())
    print(f"model {parameters} parameters, decoder {decoder_parameters}", flush=True)

    patch = arguments.patch
    optimiser = torch.optim.Adam(model.parameters(), lr=2.0e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, arguments.steps)

    def sample_batch() -> torch.Tensor:
        chosen = []
        while len(chosen) < arguments.batch:
            state = train_states[generator.integers(len(train_states))]
            row = int(generator.integers(0, shape[0] - patch))
            column = int(generator.integers(0, shape[1] - patch))
            # A patch is rejected outright if it touches a holdout region, so no
            # excluded pixel ever reaches a gradient update.
            if not mask[row : row + patch, column : column + patch].all():
                continue
            chosen.append(fields[state, row : row + patch, column : column + patch])
        return torch.from_numpy(np.stack(chosen)[:, None])

    def gradient_loss(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """First differences along both axes.

        Judging a network on a morphological measure it was never asked to
        optimise is not a fair test: plain MSE is free to smooth fine structure
        whenever that lowers the mean square, and the POD baseline shows the
        verdict lives in the gradient, 0.181 in field against 0.761 in
        gradient. So the training objective now contains the quantity the
        verdict uses.
        """

        total = candidate.new_zeros(())
        for axis in (2, 3):
            total = total + nn.functional.mse_loss(
                torch.diff(candidate, dim=axis), torch.diff(reference, dim=axis)
            )
        return total

    started = time.perf_counter()
    for iteration in range(1, arguments.steps + 1):
        batch = sample_batch()
        prediction = model(batch)
        field_term = nn.functional.mse_loss(prediction, batch)
        gradient_term = gradient_loss(prediction, batch)
        loss = field_term + arguments.gradient_weight * gradient_term
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        schedule.step()
        if iteration % 250 == 0 or iteration == 1:
            print(
                f"  step {iteration:5d}  field {field_term.item():.4e}  "
                f"gradient {gradient_term.item():.4e}  "
                f"({time.perf_counter() - started:.0f} s)",
                flush=True,
            )

    # Evaluation on whole regions, with a halo trimmed so patch borders never
    # enter a reported number.
    halo = 16
    model.eval()

    # Equal-sized windows so the four numbers are comparable: the first holdout
    # region, and a seen window of the same size taken from training pixels.
    size = HOLDOUT_REGIONS[0][2] // step
    side = (size // arguments.downsampling) * arguments.downsampling
    seen_corner = (0, 700 // step)
    holdout_corners = [(row // step, column // step) for row, column, _ in HOLDOUT_REGIONS]
    seen_window = mask[
        seen_corner[0] : seen_corner[0] + side, seen_corner[1] : seen_corner[1] + side
    ]
    if not seen_window.all():
        raise SystemExit("the seen evaluation window overlaps a holdout region")

    def evaluate(state_subset: list[int], corners: list[tuple[int, int]]) -> dict:
        """Averaged over every requested window: two holdout regions were
        defined precisely so a lucky local texture could not carry the verdict,
        and reporting only the first would waste that."""

        errors, gradients, per_window = [], [], []
        for corner_index, (row, column) in enumerate(corners):
            window_errors = []
            for index in state_subset:
                window = fields[index, row : row + side, column : column + side]
                with torch.no_grad():
                    output = model(torch.from_numpy(window[None, None]))[0, 0].numpy()
                reference = window[halo:-halo, halo:-halo]
                candidate = output[halo:-halo, halo:-halo]
                errors.append(_relative(candidate, reference))
                window_errors.append(errors[-1])
                gradients.append(_gradient_error(candidate, reference))
            per_window.append({"corner": [row, column],
                               "field_error": float(np.mean(window_errors))})
            del corner_index
        return {
            "field_error": float(np.mean(errors)),
            "gradient_error": float(np.mean(gradients)),
            "per_window": per_window,
        }

    evaluations = {
        "seen_region_seen_states": evaluate(train_states, [seen_corner]),
        "spatial_holdout_seen_states": evaluate(train_states, holdout_corners),
        "seen_region_temporal_holdout": evaluate(test_states, [seen_corner]),
        "spatial_and_temporal_holdout": evaluate(test_states, holdout_corners),
    }
    for label, entry in evaluations.items():
        print(f"  {label:34s} field {entry['field_error']:.4f}  "
              f"gradient {entry['gradient_error']:.4f}", flush=True)

    latent_per_state = (
        (shape[0] // arguments.downsampling)
        * (shape[1] // arguments.downsampling)
        * arguments.latent_channels
    )
    report = {
        "schema_version": 1,
        "status": "completed_cnn_morphology_benchmark",
        "normalisation": "H_t = EVM_t / mean(EVM_t), frozen",
        "architecture": {
            "downsampling": arguments.downsampling,
            "latent_channels": arguments.latent_channels,
            "width": arguments.width,
            "total_parameters": parameters,
            "decoder_parameters": decoder_parameters,
        },
        "training": {
            "patch": patch,
            "batch": arguments.batch,
            "steps": arguments.steps,
            "seconds": time.perf_counter() - started,
            "states": [indices[i] for i in train_states],
        },
        "compactness": {
            "latent_values_per_state": latent_per_state,
            "field_values_per_state": int(shape[0] * shape[1]),
            "compression": float(shape[0] * shape[1]) / latent_per_state,
        },
        "evaluations": evaluations,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    # The weights are the artefact, not the error table: the next step wraps
    # this decoder in the mechanical adjoint, and a benchmark whose network
    # vanishes with the process would have to be rerun to be used.
    weights = arguments.weights or arguments.output.with_suffix(".pt")
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": report["architecture"],
            "normalisation": report["normalisation"],
            "seed": arguments.seed,
            "decimate": step,
            "train_states": [indices[i] for i in train_states],
            "holdout_regions": [list(region) for region in HOLDOUT_REGIONS],
        },
        weights,
    )
    print(f"wrote {arguments.output} and {weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
