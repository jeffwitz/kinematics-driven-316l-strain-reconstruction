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
from scipy.ndimage import distance_transform_edt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized
from torch import nn

DATA = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/p0043_evm_history.h5")
CRYSTAL = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")

#: The twelve FCC octahedral systems, {111}<110>, as plane normals and slip
#: directions in the crystal frame.
FCC_SYSTEMS = [
    ((1, 1, 1), d) for d in ((-1, 1, 0), (0, -1, 1), (1, 0, -1))
] + [
    ((-1, 1, 1), d) for d in ((1, 1, 0), (0, -1, 1), (-1, 0, -1))
] + [
    ((1, -1, 1), d) for d in ((1, 1, 0), (0, 1, 1), (1, 0, -1))
] + [
    ((1, 1, -1), d) for d in ((-1, 1, 0), (0, 1, 1), (1, 0, 1))
]

# A slip direction lies in its slip plane. The check is one line and it is not
# decorative: a mistyped direction in this table stays plausible everywhere
# except in the bound it breaks. `(1,-1,1)[-1,0,-1]` sat here with a dot
# product of -2, and it pushed the Schmid factors to 0.905 -- exactly the
# (1 + |cos theta|) / 2 = 0.908 that a non-orthogonal pair allows.
for _normal, _direction in FCC_SYSTEMS:
    assert np.dot(_normal, _direction) == 0, (_normal, _direction)

#: Loading along <111> minimises the largest Schmid factor of the twelve FCC
#: octahedral systems. No orientation whatsoever can fall below it, so a map
#: that does is reporting something other than a Schmid factor.
MINIMUM_MAX_SCHMID = 0.2721655269759087


def schmid_channels(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Sorted Schmid factors per pixel, plus a validity flag, from the EBSD.

    Neither Euler angles nor a rotation matrix is used as the network input.
    316L is cubic, so an orientation is defined up to twenty-four rotations and
    both representations jump across symmetry boundaries -- a network would see
    artificial discontinuities where the crystal is unchanged. Sorting the
    twelve Schmid factors is invariant under those symmetries by construction,
    and it is the mechanically meaningful quantity: how easily each system can
    be driven by the applied axis.

    Two conventions are fixed by measurement rather than asserted.

    The loading axis is the first sample axis. `g` maps sample coordinates to
    crystal coordinates, so the loading direction seen by the crystal is a
    *column* of `g`, not a row. Taking column 0, 1 and 2 in turn and comparing
    the resulting largest factor against the archived `max_schmid_factor` gives
    0.77, 0.18 and 0.06; the choice is the measurement, not a reading of the
    acquisition notes, which do not record it.

    That 0.77 is also the honest limit of this reconstruction. The archived map
    and this one share their distribution almost exactly -- medians 0.4633 and
    0.4615 -- but agree to 1e-3 on only 18 % of pixels, and no integer shift
    improves the correlation, so the residual is a convention inside their
    calculation rather than a registration offset. What is used downstream is
    this computation, which is bounded correctly by construction; the archived
    map is used only as the outlier detector below.

    Corrupt pixels are found through that archived map. Non-indexed EBSD points
    carry the sentinel 1449 in all three Euler angles, and a sentinel still
    produces a perfectly valid rotation, hence a perfectly plausible Schmid
    factor -- this computation cannot detect its own bad inputs. The archived
    map can: 827 of its pixels fall outside the physically attainable
    [0.2722, 0.5], and they are exactly the sentinel ones. They are refilled
    from their nearest valid neighbour rather than dropped, since dropping
    pixels would punch holes into every patch that overlaps them, and the flag
    channel tells the network which pixels were invented.
    """

    with h5py.File(path, "r") as handle:
        phi1 = np.deg2rad(np.asarray(handle["orientation/phi1"], dtype=np.float64))
        capital = np.deg2rad(np.asarray(handle["orientation/Phi"], dtype=np.float64))
        phi2 = np.deg2rad(np.asarray(handle["orientation/phi2"], dtype=np.float64))
        archived = np.asarray(handle["schmid/max_schmid_factor"], dtype=np.float64)
    phi1, capital, phi2, archived = (
        a[: shape[0], : shape[1]] for a in (phi1, capital, phi2, archived)
    )
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c2, s2 = np.cos(capital), np.sin(capital)
    c3, s3 = np.cos(phi2), np.sin(phi2)
    # Column 0 of the Bunge ZXZ matrix: the first sample axis in crystal frame.
    axis = np.stack(
        [
            c1 * c3 - s1 * s3 * c2,
            -c1 * s3 - s1 * c3 * c2,
            s1 * s2,
        ],
        axis=-1,
    )
    axis = axis / np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    factors = []
    for normal, direction in FCC_SYSTEMS:
        n = np.asarray(normal, dtype=np.float64) / np.sqrt(3.0)
        d = np.asarray(direction, dtype=np.float64) / np.sqrt(2.0)
        factors.append(np.abs((axis @ n) * (axis @ d)))
    stacked = np.sort(np.stack(factors, axis=0), axis=0)[::-1].astype(np.float32)

    corrupt = ~(
        (archived >= MINIMUM_MAX_SCHMID - 1e-6) & (archived <= 0.5 + 1e-6)
    )
    if corrupt.any():
        _, nearest = distance_transform_edt(corrupt, return_indices=True)
        stacked = stacked[:, nearest[0], nearest[1]]
        print(f"EBSD: refilled {int(corrupt.sum())} non-indexed pixels "
              f"({100 * corrupt.mean():.4f} %) from their nearest valid neighbour",
              flush=True)
    top = stacked.max(axis=0)
    assert top.min() >= MINIMUM_MAX_SCHMID - 1e-6 and top.max() <= 0.5 + 1e-6, (
        float(top.min()), float(top.max())
    )
    return np.concatenate([stacked, (~corrupt)[None].astype(np.float32)], axis=0)


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
    """Context to core, keeping a spatial map all the way to the output.

    An earlier version pooled the trunk to one vector per channel before a
    dense head. That destroys exactly what the task needs: whether a band
    arrives from the left or the right, whether it is inclined one way or the
    other, whether a structure three pixels outside the core continues into it.
    Two patches with the same motifs in different places pool to nearly the
    same vector, so the head could only ever emit a position-blind average, and
    a relative error of one was the arithmetic consequence rather than a
    statement about the data.

    The trunk therefore reduces to a coarse feature map, never to a point, and
    a small decoder reads the core out of it. There is no skip across the
    blanked centre, which would let the network copy an answer it is supposed
    to infer.
    """

    def __init__(self, *, context: int, core: int, width: int = 32, extra: int = 0) -> None:
        super().__init__()
        # Down to a map of about a quarter of the core, so position survives.
        levels = max(1, int(np.log2(context // max(core // 4, 2))))
        layers: list[nn.Module] = []
        channels = 2 + extra
        for level in range(levels):
            out = width * (2 ** min(level, 2))
            layers += [nn.Conv2d(channels, out, 4, stride=2, padding=1),
                       nn.GroupNorm(4, out), nn.GELU()]
            channels = out
        self.trunk = nn.Sequential(*layers)
        self.reduced = context // (2**levels)
        head: list[nn.Module] = []
        size = self.reduced
        while size < core:
            head += [nn.ConvTranspose2d(channels, width, 4, stride=2, padding=1),
                     nn.GroupNorm(4, width), nn.GELU()]
            channels = width
            size *= 2
        head.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.head = nn.Sequential(*head)
        self.core = core

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        features = self.trunk(patch)
        decoded = self.head(features)[:, 0]
        centre = (decoded.shape[-1] - self.core) // 2
        return decoded[:, centre : centre + self.core, centre : centre + self.core]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DATA)
    parser.add_argument("--core", type=int, default=16)
    parser.add_argument("--contexts", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--evaluations", type=int, default=400)
    # Full resolution: decimating without an anti-aliasing filter has already
    # cost this campaign twice, and a small core on a decimated grid would sit
    # at exactly the scales the subsampling altered.
    parser.add_argument("--decimate", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--ebsd", action="store_true")
    parser.add_argument("--schmid-channels", type=int, default=6)
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

    crystal = None
    if arguments.ebsd:
        crystal = schmid_channels(CRYSTAL, (shape[0] * step, shape[1] * step))
        # The last channel is the validity flag and is always kept; the leading
        # ones are the sorted factors, of which the largest carry the signal.
        crystal = np.concatenate(
            [crystal[: arguments.schmid_channels], crystal[-1:]]
        )[:, ::step, ::step]
        print(f"EBSD: {crystal.shape[0] - 1} sorted Schmid channels plus a "
              f"validity flag, range {crystal[:-1].min():.3f} to "
              f"{crystal[:-1].max():.3f}", flush=True)

    solve = harmonic_operator(core)

    def sample(context: int, inside: bool, count: int):
        half = context // 2
        offset = half - core // 2
        patches, cores, corners = [], [], []
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
            corners.append((row, column))
        return np.stack(patches), np.stack(cores), offset, corners

    def masked(patches: np.ndarray, offset: int, corners: list) -> torch.Tensor:
        blanked = patches.copy()
        blanked[:, offset : offset + core, offset : offset + core] = 0.0
        flag = np.zeros_like(blanked)
        flag[:, offset : offset + core, offset : offset + core] = 1.0
        channels = [blanked, flag]
        if crystal is not None:
            # The crystallography is *not* blanked in the core: it is a static
            # material property, known everywhere, unlike the measurement.
            extent = patches.shape[1]
            for index in range(crystal.shape[0]):
                channels.append(
                    np.stack([crystal[index, r : r + extent, c : c + extent]
                              for r, c in corners])
                )
        return torch.from_numpy(np.stack(channels, axis=1))

    def scores(candidate: np.ndarray, reference: np.ndarray) -> dict:
        """Level and morphology reported separately.

        The combined figure conflates two failures. A prediction that recovers
        the texture but misses the level by a little would otherwise be graded
        with one that emits a constant, and the morphology is the question.
        """

        reference_mean = reference.mean(axis=(1, 2), keepdims=True)
        candidate_mean = candidate.mean(axis=(1, 2), keepdims=True)
        fluctuation = reference - reference_mean
        norm = max(float(np.linalg.norm(fluctuation)), 1e-30)
        return {
            "total": float(np.linalg.norm(candidate - reference) / norm),
            "morphology": float(
                np.linalg.norm((candidate - candidate_mean) - fluctuation) / norm
            ),
            "level": float(np.linalg.norm(candidate_mean - reference_mean) / norm),
        }

    results = []
    for context in arguments.contexts:
        model = CorePredictor(context=context, core=core,
                              extra=0 if crystal is None else crystal.shape[0])
        parameters = sum(p.numel() for p in model.parameters())
        optimiser = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, arguments.steps)
        started = time.perf_counter()
        for iteration in range(1, arguments.steps + 1):
            patches, cores, offset, corners = sample(context, False, arguments.batch)
            loss = nn.functional.mse_loss(
                model(masked(patches, offset, corners)), torch.from_numpy(cores)
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
            if iteration % 1000 == 0:
                print(f"  context {context} step {iteration}: loss {loss.item():.4e} "
                      f"({time.perf_counter() - started:.0f} s)", flush=True)

        patches, cores, offset, corners = sample(context, True, arguments.evaluations)
        model.eval()
        with torch.no_grad():
            predicted = model(masked(patches, offset, corners)).numpy()
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
            "network": scores(predicted, cores),
            "harmonic_inpainting": scores(harmonic, cores),
            "ring_mean": scores(np.asarray(constant), cores),
            "seconds": time.perf_counter() - started,
        }
        entry["morphology_gain_over_harmonic"] = 1.0 - entry["network"]["morphology"] / max(
            entry["harmonic_inpainting"]["morphology"], 1e-30
        )
        results.append(entry)
        print(f"context {context}: network morph {entry['network']['morphology']:.4f} "
              f"(level {entry['network']['level']:.4f})  harmonic morph "
              f"{entry['harmonic_inpainting']['morphology']:.4f}  ring mean "
              f"{entry['ring_mean']['morphology']:.4f}  gain "
              f"{100 * entry['morphology_gain_over_harmonic']:+.1f} %", flush=True)

    report = {
        "schema_version": 1,
        "status": "completed_local_transferability",
        "core": core,
        "ebsd": bool(arguments.ebsd),
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
