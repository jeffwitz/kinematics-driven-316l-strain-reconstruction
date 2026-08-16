#!/usr/bin/env python3
"""A CROM-style neural field against POD: r coordinates per state, not an image.

The convolutional autoencoder answered the wrong question. At s32/d8 it carried
86016 numbers per state and still destroyed everything below thirty-two pixels
in its bottleneck -- too many state degrees of freedom and too little
resolution at once, and its error curve was flat from step six thousand to
twenty thousand at a gradient error of 0.97. What a reduced model needs is the
opposite: a few generalised coordinates parameterising a field that keeps its
full spatial resolution.

So the field becomes a continuous function of position, shared across states,
conditioned on a small latent:

```text
H_t(x, y) = D_theta(gamma(x, y), z_t),     z_t in R^r,  r = 4, 8, 16, 32
```

`gamma` is a random Fourier feature map, because a plain MLP has a spectral
bias and learns high frequencies badly, and the fine scales are exactly what
the previous architecture lost. There is no encoder: `theta` and every `z_t`
are optimised jointly, the auto-decoder form, so nothing has to compress an
image and the latent is not forced to be an image.

The shared decoder is also, by construction, the static component the fine-band
test called for: the part of the field common to all states is what `theta`
represents when `z` barely moves, so no term has to be subtracted by hand and
POD keeps its mean field for an identical comparison.

## Why this architecture and not a better-tuned autoencoder

Its tangent map is what a mechanical adjoint needs. A later tensor decoder
`D_p(x, z)` giving the Kelvin plastic triple has `J_D : R^r -> R^{3N}`, so the
reduced operator is `A J_D` and its adjoint `J_D^T A^T`, with r of order ten.
The convolutional latent would have made that map `R^{172000} -> R^{3N}`, which
is not a reduced model in any useful sense.

## Held-out states are identified, not encoded

For a state outside training, `theta` is frozen and only its `z_t` is fitted,
on pixels away from the evaluation windows; the error is then measured
elsewhere. That asks whether a few coordinates place an unseen state on the
learned manifold -- the question a reduced model has to answer -- rather than
whether an encoder can copy an image it is shown.
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
    WINDOW_CORNER,
    WINDOW_SIDE,
    _gradient_error,
    _relative,
)
from morphology_benchmark_split import split_states  # type: ignore[import-not-found]
from torch import nn

DATA = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/p0043_evm_history.h5")


class FourierFeatures(nn.Module):
    """Random Fourier map on normalised coordinates.

    The frequencies are drawn once and frozen; training them adds parameters
    without changing what the map is for, which is to remove the spectral bias
    that keeps a plain MLP from representing fine structure.
    """

    def __init__(self, count: int, scale: float, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "frequencies", torch.randn(2, count, generator=generator) * scale
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        projected = 2.0 * np.pi * coordinates @ self.frequencies
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class NeuralField(nn.Module):
    def __init__(
        self, *, latent: int, features: int, scale: float, width: int, depth: int, seed: int
    ) -> None:
        super().__init__()
        self.encoding = FourierFeatures(features, scale, seed)
        layers: list[nn.Module] = [nn.Linear(2 * features + latent, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers.append(nn.Linear(width, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, coordinates: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        encoded = self.encoding(coordinates)
        if latent.dim() == 1:
            latent = latent.expand(encoded.shape[0], -1)
        return self.body(torch.cat([encoded, latent], dim=-1)).squeeze(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DATA)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--features", type=int, default=128)
    # The scale sets the bandwidth, and getting it wrong disables the whole
    # point of the encoding. Coordinates span [-1, 1], so a frequency b gives
    # 2b cycles across the field; reaching Nyquist on an N-pixel axis needs
    # b ~ N/4. At scale 12 the map topped out near forty cycles, resolving
    # nothing finer than about forty-five pixels -- the same failure as the
    # convolutional bottleneck, reached by a different route. Default None
    # derives it from the grid instead of guessing.
    parser.add_argument("--fourier-scale", type=float, default=None)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch-coordinates", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--identify-steps", type=int, default=1500)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--decimate", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    generator = np.random.default_rng(arguments.seed)
    step = arguments.decimate

    with h5py.File(arguments.history, "r") as handle:
        evm = handle["evm"]
        means = np.asarray(handle.attrs["mean_evm"], dtype=np.float64)
        indices = list(range(1, int(evm.shape[0])))
        fields = np.stack(
            [
                (np.asarray(evm[state][::step, ::step], dtype=np.float32)
                 / np.float32(means[state]))
                for state in indices
            ]
        )
    shape = fields.shape[1:]
    train_states, test_states = split_states(indices)
    if arguments.fourier_scale is None:
        # Three sigma reaches Nyquist, so the tail of the distribution covers
        # the finest resolvable structure rather than stopping short of it.
        arguments.fourier_scale = float(min(shape)) / 12.0
        print(f"fourier scale set to {arguments.fourier_scale:.1f} from the grid", flush=True)

    mask = np.ones(shape, dtype=bool)
    for row, column, size in HOLDOUT_REGIONS:
        mask[row // step : (row + size) // step, column // step : (column + size) // step] = False
    print(f"{len(indices)} states, shape {shape}, "
          f"spatial holdout {100 * (1 - mask.mean()):.1f} %", flush=True)

    rows = (np.arange(shape[0], dtype=np.float32) / (shape[0] - 1)) * 2.0 - 1.0
    columns = (np.arange(shape[1], dtype=np.float32) / (shape[1] - 1)) * 2.0 - 1.0
    train_pixels = np.flatnonzero(mask.ravel())

    model = NeuralField(
        latent=arguments.latent, features=arguments.features,
        scale=arguments.fourier_scale, width=arguments.width,
        depth=arguments.depth, seed=arguments.seed,
    )
    shared = sum(p.numel() for p in model.parameters() if p.requires_grad)
    latents = nn.Parameter(torch.zeros(len(indices), arguments.latent))
    with torch.no_grad():
        latents.normal_(0.0, 0.01)
    print(f"shared parameters {shared}, {arguments.latent} coordinates per state", flush=True)

    def coordinates_of(flat: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            np.stack([rows[flat // shape[1]], columns[flat % shape[1]]], axis=-1)
        )

    def window_grid(corner: tuple[int, int], side: int):
        row, column = corner[0] // step, corner[1] // step
        extent = side // step
        grid_rows, grid_columns = np.meshgrid(
            np.arange(row, row + extent), np.arange(column, column + extent), indexing="ij"
        )
        flat = (grid_rows * shape[1] + grid_columns).ravel()
        return flat, extent

    def predict(flat: np.ndarray, latent: torch.Tensor) -> np.ndarray:
        pieces = []
        with torch.no_grad():
            for start in range(0, flat.size, 65536):
                chunk = flat[start : start + 65536]
                pieces.append(model(coordinates_of(chunk), latent).numpy())
        return np.concatenate(pieces)

    def identify(index: int, exclude: list[np.ndarray]) -> torch.Tensor:
        """Fit only `z` for a held-out state, on pixels away from the windows."""

        blocked = np.zeros(shape[0] * shape[1], dtype=bool)
        for flat in exclude:
            blocked[flat] = True
        available = train_pixels[~blocked[train_pixels]]
        latent = torch.zeros(arguments.latent, requires_grad=True)
        optimiser = torch.optim.Adam([latent], lr=1.0e-2)
        target = fields[index].ravel()
        for _ in range(arguments.identify_steps):
            picked = available[generator.integers(0, available.size, arguments.batch_coordinates)]
            loss = nn.functional.mse_loss(
                model(coordinates_of(picked), latent),
                torch.from_numpy(target[picked]),
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        return latent.detach()

    seen_flat, extent = window_grid(WINDOW_CORNER, WINDOW_SIDE)
    holdout_flat, _ = window_grid((HOLDOUT_REGIONS[0][0], HOLDOUT_REGIONS[0][1]), WINDOW_SIDE)

    def evaluate(subset: list[int], flat: np.ndarray, coefficients: dict) -> dict:
        errors, gradients = [], []
        for index in subset:
            prediction = predict(flat, coefficients[index]).reshape(extent, extent)
            reference = fields[index].ravel()[flat].reshape(extent, extent)
            errors.append(_relative(prediction, reference))
            gradients.append(_gradient_error(prediction, reference))
        return {"field_error": float(np.mean(errors)),
                "gradient_error": float(np.mean(gradients))}

    optimiser = torch.optim.Adam(
        [{"params": model.parameters(), "lr": 1.0e-3},
         {"params": [latents], "lr": 1.0e-2}]
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, arguments.steps)
    flat_fields = fields.reshape(len(indices), -1)
    progress: list[dict] = []
    started = time.perf_counter()
    for iteration in range(1, arguments.steps + 1):
        index = train_states[generator.integers(len(train_states))]
        picked = train_pixels[
            generator.integers(0, train_pixels.size, arguments.batch_coordinates)
        ]
        loss = nn.functional.mse_loss(
            model(coordinates_of(picked), latents[index]),
            torch.from_numpy(flat_fields[index][picked]),
        )
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        schedule.step()
        if iteration % 500 == 0 or iteration == 1:
            print(f"  step {iteration:6d}  loss {loss.item():.5e}  "
                  f"({time.perf_counter() - started:.0f} s)", flush=True)
        if iteration % arguments.checkpoint_every == 0:
            trained = {i: latents[i].detach() for i in train_states}
            entry = {
                "step": iteration,
                "seconds": time.perf_counter() - started,
                "seen_region_seen_states": evaluate(train_states, seen_flat, trained),
                "spatial_holdout_seen_states": evaluate(train_states, holdout_flat, trained),
            }
            progress.append(entry)
            arguments.output.with_suffix(".progress.json").write_text(
                json.dumps(progress, indent=2) + "\n", "utf-8"
            )
            print(f"    checkpoint {iteration}: seen "
                  f"{entry['seen_region_seen_states']['field_error']:.4f} / "
                  f"{entry['seen_region_seen_states']['gradient_error']:.4f}  "
                  f"spatial holdout "
                  f"{entry['spatial_holdout_seen_states']['field_error']:.4f}",
                  flush=True)

    coefficients = {i: latents[i].detach() for i in train_states}
    print("identifying the held-out states with theta frozen", flush=True)
    for index in test_states:
        coefficients[index] = identify(index, [seen_flat, holdout_flat])

    evaluations = {
        "seen_region_seen_states": evaluate(train_states, seen_flat, coefficients),
        "spatial_holdout_seen_states": evaluate(train_states, holdout_flat, coefficients),
        "seen_region_temporal_holdout": evaluate(test_states, seen_flat, coefficients),
        "spatial_and_temporal_holdout": evaluate(test_states, holdout_flat, coefficients),
    }
    for label, entry in evaluations.items():
        print(f"  {label:32s} field {entry['field_error']:.4f}  "
              f"gradient {entry['gradient_error']:.4f}", flush=True)

    report = {
        "schema_version": 1,
        "status": "completed_neural_field_benchmark",
        "normalisation": "H_t = EVM_t / mean(EVM_t), frozen",
        "architecture": {
            "kind": "CROM-style implicit neural representation, auto-decoder",
            "latent": arguments.latent,
            "fourier_features": arguments.features,
            "fourier_scale": arguments.fourier_scale,
            "width": arguments.width,
            "depth": arguments.depth,
            "shared_parameters": shared,
        },
        "compactness": {
            "coordinates_per_state": arguments.latent,
            "field_values_per_state": int(shape[0] * shape[1]),
        },
        "decimation": step,
        "train_states": [indices[i] for i in train_states],
        "temporal_holdout_states": [indices[i] for i in test_states],
        "training": {"steps": arguments.steps, "seconds": time.perf_counter() - started},
        "evaluations": evaluations,
        "progress": progress,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    torch.save(
        {"state_dict": model.state_dict(),
         "latents": {indices[i]: coefficients[i] for i in coefficients},
         "architecture": report["architecture"]},
        arguments.output.with_suffix(".pt"),
    )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
