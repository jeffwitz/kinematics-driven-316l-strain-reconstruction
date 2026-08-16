#!/usr/bin/env python3
"""Is there a local rule that transfers to a region it was never trained on?

The reduction campaign refuted the global hypothesis: no fixed basis spans the
morphologies of held-out states at any scale. A shared *local* rule has no such
constraint, and this measures whether one exists.

The task is deliberately morphological, not yet mechanical. A network sees a
square context with its centre blanked and predicts that centre; the weights
are shared across every position and state, and the evaluation happens inside
two regions from which no training sample was ever drawn.

## The baseline is not optional

A network shown a large context and asked for a small core can simply
interpolate: the core is largely determined by continuity with its
surroundings. Measuring only the network error would report an inpainting
ability and read it as a transferable law. So every number is quoted against
harmonic inpainting -- the Laplace solution in the core with Dirichlet data
from its immediate ring, which is the natural "smooth continuation" answer --
and against the ring mean. Beating those is the minimum evidence that anything
was learned; not beating them refutes the hypothesis outright at this scale.

## The radius is measured rather than chosen

The same task is run at several context sizes with the core fixed. Where the
network stops improving estimates the informative range of the local rule, and
that plateau is what should set the architecture later, instead of a patch size
picked by hand.

**This is morphology, not a constitutive law.** Succeeding here shows a
transferable local regularity in the measured field and nothing more; the
mechanical closure is a separate experiment in which the network never sees the
interior measurement.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from benchmark_pod_morphology import HOLDOUT_REGIONS  # type: ignore[import-not-found]
from morphology_benchmark_split import split_states  # type: ignore[import-not-found]
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized
from torch import nn

DATA = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/p0043_evm_history.h5")


def harmonic_operator(core: int):
    """Factorised Laplacian for inpainting a `core`-square from its ring.

    The smooth continuation of the surroundings, which is exactly what a
    network could produce without having learned anything about the material.
    """

    size = core * core
    matrix = lil_matrix((size, size))
    for i in range(core):
        for j in range(core):
            k = i * core + j
            matrix[k, k] = 4.0
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < core and 0 <= nj < core:
                    matrix[k, ni * core + nj] = -1.0
    return factorized(matrix.tocsc())


def harmonic_inpaint(solve, patch: np.ndarray, core: int, offset: int) -> np.ndarray:
    """Laplace solution inside the core, Dirichlet from the surrounding ring."""

    right = np.zeros((core, core))
    right[0] += patch[offset - 1, offset : offset + core]
    right[-1] += patch[offset + core, offset : offset + core]
    right[:, 0] += patch[offset : offset + core, offset - 1]
    right[:, -1] += patch[offset : offset + core, offset + core]
    return solve(right.ravel()).reshape(core, core)


class CorePredictor(nn.Module):
    """Context to core: convolutional trunk, global pooling, dense head.

    Pooling rather than a decoder over the full context, because the output is
    a small core and the point is what the surroundings say about it, not a
    reconstruction of the surroundings themselves.
    """

    def __init__(self, *, context: int, core: int, width: int = 32) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = 2
        for _ in range(int(np.log2(context // 8))):
            layers += [nn.Conv2d(channels, width, 4, stride=2, padding=1),
                       nn.GroupNorm(4, width), nn.GELU()]
            channels = width
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(width, 256), nn.GELU(), nn.Linear(256, core * core)
        )
        self.core = core

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        pooled = self.trunk(patch).flatten(1)
        return self.head(pooled).view(-1, self.core, self.core)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DATA)
    parser.add_argument("--core", type=int, default=16)
    parser.add_argument("--contexts", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--evaluations", type=int, default=400)
    parser.add_argument("--decimate", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    generator = np.random.default_rng(arguments.seed)
    step = arguments.decimate
    core = arguments.core

    with h5py.File(arguments.history, "r") as handle:
        evm = handle["evm"]
        means = np.asarray(handle.attrs["mean_evm"], dtype=np.float64)
        indices = list(range(1, int(evm.shape[0])))
        fields = np.stack(
            [(np.asarray(evm[s][::step, ::step], dtype=np.float32) / np.float32(means[s]))
             for s in indices]
        )
    shape = fields.shape[1:]
    train_states, _ = split_states(indices)

    # Held-out regions, in the decimated grid. No training sample may overlap
    # them, and every reported number is measured inside them.
    regions = [
        (row // step, column // step, size // step) for row, column, size in HOLDOUT_REGIONS
    ]
    excluded = np.zeros(shape, dtype=bool)
    for row, column, size in regions:
        excluded[row : row + size, column : column + size] = True
    print(f"{len(indices)} states, shape {shape}, core {core}, "
          f"held-out {100 * excluded.mean():.1f} % of pixels", flush=True)

    solve = harmonic_operator(core)

    def sample(context: int, inside: bool, count: int):
        half = context // 2
        offset = half - core // 2
        patches, cores = [], []
        while len(patches) < count:
            state = (train_states[generator.integers(len(train_states))]
                     if not inside else generator.integers(len(indices)))
            row = int(generator.integers(0, shape[0] - context))
            column = int(generator.integers(0, shape[1] - context))
            window = excluded[row : row + context, column : column + context]
            centre = excluded[row + offset : row + offset + core,
                              column + offset : column + offset + core]
            # Training patches must not touch a held-out region at all;
            # evaluation patches must have their core strictly inside one.
            if inside:
                if not centre.all():
                    continue
            elif window.any():
                continue
            patches.append(fields[state, row : row + context, column : column + context])
            cores.append(patches[-1][offset : offset + core, offset : offset + core].copy())
        return np.stack(patches), np.stack(cores), offset

    def masked(patches: np.ndarray, offset: int) -> torch.Tensor:
        blanked = patches.copy()
        blanked[:, offset : offset + core, offset : offset + core] = 0.0
        flag = np.zeros_like(blanked)
        flag[:, offset : offset + core, offset : offset + core] = 1.0
        return torch.from_numpy(np.stack([blanked, flag], axis=1))

    def relative(candidate: np.ndarray, reference: np.ndarray) -> float:
        # Relative to the fluctuation of the core, not to its level: predicting
        # the mean of a nearly constant patch is not a success.
        centred = reference - reference.mean(axis=(1, 2), keepdims=True)
        return float(
            np.linalg.norm(candidate - reference) / max(np.linalg.norm(centred), 1e-30)
        )

    results = []
    for context in arguments.contexts:
        model = CorePredictor(context=context, core=core)
        parameters = sum(p.numel() for p in model.parameters())
        optimiser = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, arguments.steps)
        started = time.perf_counter()
        for iteration in range(1, arguments.steps + 1):
            patches, cores, offset = sample(context, False, arguments.batch)
            loss = nn.functional.mse_loss(
                model(masked(patches, offset)), torch.from_numpy(cores)
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
            if iteration % 1000 == 0:
                print(f"  context {context} step {iteration}: loss {loss.item():.4e} "
                      f"({time.perf_counter() - started:.0f} s)", flush=True)

        patches, cores, offset = sample(context, True, arguments.evaluations)
        model.eval()
        with torch.no_grad():
            predicted = model(masked(patches, offset)).numpy()
        harmonic = np.stack(
            [harmonic_inpaint(solve, patch, core, offset) for patch in patches]
        )
        ring = patches.copy()
        ring[:, offset : offset + core, offset : offset + core] = np.nan
        constant = np.broadcast_to(
            np.nanmean(ring, axis=(1, 2))[:, None, None], cores.shape
        )
        entry = {
            "context": context,
            "parameters": parameters,
            "network": relative(predicted, cores),
            "harmonic_inpainting": relative(harmonic, cores),
            "ring_mean": relative(np.asarray(constant), cores),
            "seconds": time.perf_counter() - started,
        }
        entry["gain_over_harmonic"] = (
            1.0 - entry["network"] / max(entry["harmonic_inpainting"], 1e-30)
        )
        results.append(entry)
        print(f"context {context}: network {entry['network']:.4f}  harmonic "
              f"{entry['harmonic_inpainting']:.4f}  ring mean {entry['ring_mean']:.4f}  "
              f"gain {100 * entry['gain_over_harmonic']:+.1f} %", flush=True)

    report = {
        "schema_version": 1,
        "status": "completed_local_transferability",
        "core": core,
        "decimation": step,
        "evaluation": "cores strictly inside the two spatial holdout regions",
        "metric": "error relative to the fluctuation of the core",
        "results": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
