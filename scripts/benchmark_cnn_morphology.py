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
    parser.add_argument("--train-states", type=int, default=30)
    parser.add_argument("--decimate", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output", type=Path, required=True)
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
    train_states = [i for i, s in enumerate(indices) if s <= arguments.train_states]
    test_states = [i for i, s in enumerate(indices) if s > arguments.train_states]
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

    started = time.perf_counter()
    for iteration in range(1, arguments.steps + 1):
        batch = sample_batch()
        loss = nn.functional.mse_loss(model(batch), batch)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        schedule.step()
        if iteration % 250 == 0 or iteration == 1:
            print(
                f"  step {iteration:5d}  loss {loss.item():.5e}  "
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
    seen_corner = (0, 700)
    holdout_corner = (HOLDOUT_REGIONS[0][0] // step, HOLDOUT_REGIONS[0][1] // step)
    seen_window = mask[
        seen_corner[0] : seen_corner[0] + side, seen_corner[1] : seen_corner[1] + side
    ]
    if not seen_window.all():
        raise SystemExit("the seen evaluation window overlaps a holdout region")

    def evaluate(state_subset: list[int], corner: tuple[int, int]) -> dict:
        errors, gradients = [], []
        for index in state_subset:
            row, column = corner
            window = fields[index, row : row + side, column : column + side]
            with torch.no_grad():
                output = model(torch.from_numpy(window[None, None]))[0, 0].numpy()
            reference = window[halo:-halo, halo:-halo]
            candidate = output[halo:-halo, halo:-halo]
            errors.append(_relative(candidate, reference))
            gradients.append(_gradient_error(candidate, reference))
        return {
            "field_error": float(np.mean(errors)),
            "gradient_error": float(np.mean(gradients)),
        }

    evaluations = {
        "seen_region_seen_states": evaluate(train_states, seen_corner),
        "spatial_holdout_seen_states": evaluate(train_states, holdout_corner),
        "seen_region_temporal_holdout": evaluate(test_states, seen_corner),
        "spatial_and_temporal_holdout": evaluate(test_states, holdout_corner),
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
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
