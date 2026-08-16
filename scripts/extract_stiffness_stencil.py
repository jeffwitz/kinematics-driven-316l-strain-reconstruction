#!/usr/bin/env python3
"""Identify `K = B^T C B` as a fixed local stencil, from its impulse response.

The profile says 84 % of a PCG iteration is spent in `K`, and inside it the
constitutive product costs 1.3 ms against 23.5 for `B` and 54.4 for `B^T`. On a
regular grid with homogeneous elasticity that composition is a constant local
operator, so the three passes

```text
u -> B u -> C B u -> B^T C B u
```

buy nothing but generality we do not need here. There is no implementation
pathology to fix first -- no `add.at`, no indirect scatter -- and rewiring onto
the buffered `_into` variants measured *slower*, 95 ms against 53. The cost is
the successive passes themselves.

The stencil is not derived by hand. The generic operator is already qualified
against the assembled sparse one at 4e-13, so it is used as the oracle: place a
unit impulse on one interior node, apply `K`, and read the coefficients off the
response. Translation invariance then extends that to the whole grid. Nothing
is assumed about the support or about how many stencils there are -- the mesh
may alternate with pixel parity, so all four classes are extracted and compared,
and the non-zero support of the response decides its own size.

The NumPy implementation below is a **validation prototype, not the verdict on
what a stencil can do**. Each accumulated neighbour rereads and rewrites the
whole output array, so a nine-point stencil makes several full passes through
memory where a fused kernel would make one. Its throughput is a lower bound.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from qualify_full_field_plastic_operator import FullFieldPlasticOperator

from fem_inhouse.spectral2d.grid import StructuredGrid2D

PIXEL_SIZE_MM = 0.00184


def impulse_response(operator: FullFieldPlasticOperator, node: tuple[int, int],
                     component: int) -> np.ndarray:
    """`K e` for a unit displacement on one interior node, as a nodal field."""

    shape = operator.interior_shape
    field = np.zeros(shape)
    field[node[0], node[1], component] = 1.0
    return operator.stiffness(field.reshape(-1)).reshape(shape)


def extract(operator: FullFieldPlasticOperator, parity: tuple[int, int],
            radius: int, tolerance: float) -> tuple[dict, float]:
    """The `2x2` blocks around a node of the given parity class."""

    shape = operator.interior_shape
    centre = (shape[0] // 2, shape[1] // 2)
    node = (centre[0] + (parity[0] - centre[0] % 2) % 2,
            centre[1] + (parity[1] - centre[1] % 2) % 2)
    columns = [impulse_response(operator, node, c) for c in (0, 1)]
    largest = max(float(np.abs(c).max()) for c in columns)
    blocks: dict[tuple[int, int], np.ndarray] = {}
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            # Response at the neighbour is the column of the block that the
            # *neighbour* applies to this node; by symmetry of K the block at
            # (di, dj) read this way is the transpose of the one acting from
            # (di, dj) onto the centre, and the check below settles it.
            block = np.array(
                [[columns[c][node[0] + di, node[1] + dj, r] for c in (0, 1)]
                 for r in (0, 1)]
            )
            if np.abs(block).max() > tolerance * largest:
                blocks[(di, dj)] = block
    return blocks, largest


def apply_stencil(blocks: dict[tuple[int, int], np.ndarray],
                  field: np.ndarray) -> np.ndarray:
    """Slice-based accumulation. Views, never `roll`, which copies and wraps.

    The fluctuation vanishes outside the interior, so a zero pad is exactly the
    Dirichlet condition and no special boundary branch is needed.
    """

    radius = max(max(abs(i), abs(j)) for i, j in blocks)
    padded = np.zeros((field.shape[0] + 2 * radius, field.shape[1] + 2 * radius, 2))
    padded[radius:-radius, radius:-radius] = field
    out = np.zeros_like(field)
    height, width = field.shape[0], field.shape[1]
    for (di, dj), block in blocks.items():
        window = padded[radius + di : radius + di + height,
                        radius + dj : radius + dj + width]
        out += window @ block.T
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=64)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    n = arguments.pixels
    grid = StructuredGrid2D(n, n, PIXEL_SIZE_MM * n, PIXEL_SIZE_MM * n)
    operator = FullFieldPlasticOperator(grid, backend="fftw", workers=1)

    # 1. Does the stencil depend on the parity of the node?
    classes = {}
    for parity in ((0, 0), (0, 1), (1, 0), (1, 1)):
        blocks, largest = extract(operator, parity, arguments.radius, arguments.tolerance)
        classes[parity] = blocks
        print(f"parity {parity}: {len(blocks)} non-zero blocks, "
              f"largest coefficient {largest:.6e}", flush=True)

    reference = classes[(0, 0)]
    spread = 0.0
    for parity, blocks in classes.items():
        if set(blocks) != set(reference):
            spread = float("inf")
            print(f"parity {parity} has a different support", flush=True)
            continue
        for key, block in blocks.items():
            spread = max(spread, float(np.abs(block - reference[key]).max()))
    scale = max(float(np.abs(b).max()) for b in reference.values())
    print(f"largest disagreement between parity classes: {spread / scale:.3e} relative",
          flush=True)
    uniform = spread / scale < 1e-10
    print(f"-> {'one stencil suffices' if uniform else 'the stencil depends on parity'}",
          flush=True)
    if not uniform:
        raise SystemExit("parity-dependent stencils are not handled by this prototype")

    offsets = sorted(reference)
    extent = max(max(abs(i), abs(j)) for i, j in offsets)
    print(f"support: {len(offsets)} blocks within radius {extent}", flush=True)

    # 2. Does it reproduce the operator it came from?
    report: dict[str, object] = {
        "identified_on_pixels": n,
        "block_count": len(offsets),
        "support_radius": extent,
        "offsets": [list(k) for k in offsets],
        "parity_disagreement": spread / scale,
        "sizes": {},
    }
    generator = np.random.default_rng(20260816)
    for size in arguments.sizes:
        other = StructuredGrid2D(size, size, PIXEL_SIZE_MM * size, PIXEL_SIZE_MM * size)
        checked = FullFieldPlasticOperator(other, backend="fftw", workers=1)
        shape = checked.interior_shape
        u = generator.standard_normal(shape)
        v = generator.standard_normal(shape)
        exact = checked.stiffness(u.reshape(-1)).reshape(shape)
        mine = apply_stencil(reference, u)
        error = float(np.linalg.norm(mine - exact) / np.linalg.norm(exact))
        # Symmetry and positivity matter as much as agreement: CG needs both.
        symmetry = abs(float((u * apply_stencil(reference, v)).sum())
                       - float((apply_stencil(reference, u) * v).sum()))
        symmetry /= max(abs(float((u * apply_stencil(reference, v)).sum())), 1e-300)
        energy = float((u * apply_stencil(reference, u)).sum())

        def timed(function, argument, repeats=6):
            function(argument)
            started = time.time()
            for _ in range(repeats):
                function(argument)
            return (time.time() - started) / repeats

        generic = timed(checked.stiffness, u.reshape(-1))
        stencil = timed(lambda x, b=reference: apply_stencil(b, x), u)
        pixels = shape[0] * shape[1]
        report["sizes"][str(size)] = {
            "relative_error": error,
            "symmetry": symmetry,
            "energy": energy,
            "generic_seconds": generic,
            "stencil_seconds": stencil,
            "speedup": generic / stencil,
            "megapixels_per_second": pixels / stencil / 1e6,
            "minimum_equivalent_gigabytes_per_second": 32.0 * pixels / stencil / 1e9,
        }
        print(
            f"{size:4d}: error {error:.3e}  symmetry {symmetry:.3e}  energy {energy:+.4e}"
            f"  |  generic {1000 * generic:7.1f} ms  stencil {1000 * stencil:7.1f} ms"
            f"  x{generic / stencil:.2f}  {pixels / stencil / 1e6:.1f} Mpix/s"
            f"  >= {32.0 * pixels / stencil / 1e9:.2f} GB/s",
            flush=True,
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\nwrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
