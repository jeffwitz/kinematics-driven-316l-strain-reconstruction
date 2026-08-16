#!/usr/bin/env python3
"""A fused CPU kernel for the seven-block elastic stencil.

Not a Numba transcription of the NumPy version. The NumPy path accumulates one
neighbour at a time, so seven blocks mean seven passes over the whole field,
each rereading and rewriting the output. This does the opposite: one pass, and
for each pixel the seven neighbours are loaded, contracted into two scalar
accumulators, and written once.

Inside the hot loop there is no strain array, no stress array, no temporary
slice, no `stack`, no `einsum`, and no allocation of any kind.

The seven offsets are those the impulse response actually produced --
`(-1,0) (-1,1) (0,-1) (0,0) (0,1) (1,-1) (1,0)`. The two absent corners are
`(-1,-1)` and `(1,1)`: the TRI2 split cuts along the other diagonal. They are
baked in rather than looped over, since the stencil does not change across the
millions of applications a campaign performs.

Layout is `(2, H, W)`. I predicted `(H, W, 2)` would win, reasoning that a
neighbour's two components sit on one cache line, and measured the opposite --
11.4 ms against 8.1 at 2048 square. The stride is what matters: `(2, H, W)`
gives the inner loop unit stride per component and lets it vectorise, while two
interleaved doubles defeat that. The `einsum` experiment had said the same thing
and I should have believed it.

There is **no halo**. A padded buffer costs a copy in and a copy out on every
application -- 46 ms around an 8 ms kernel, a fused loop wrapped in exactly the
traffic it exists to avoid. Instead the kernel reads the solver's own array
directly, so a CG vector is used as a `reshape` view at no cost, and the outer
ring is handled by a separate pass. The fluctuation vanishes outside the
interior, so a missing neighbour contributes nothing; the ring is four thousandths
of the pixels, and a slow loop there costs nothing.

Only the outer loop is a `prange`: every row carries the same work, so a static
split gives each thread a contiguous band of rows and its own region of `out`,
with no atomics, no reduction and no per-pixel synchronisation.

`fastmath` stays **off**. The operator currently reproduces the generic one to
1.95e-16 with symmetry at 5e-15, and reassociating floating-point operations is
exactly what would move those last bits. It is an experiment for afterwards,
gated on the whole qualification passing again.
"""

from __future__ import annotations

import numpy as np

try:  # Performance extra. The NumPy path below is the reference and must work
    from numba import njit, prange  # without it, so nothing scientific depends
    HAS_NUMBA = True                # on a compiler being present.
except ImportError:  # pragma: no cover - exercised only where numba is absent
    HAS_NUMBA = False

    def njit(*_args, **_kwargs):
        def decorate(function):
            return function

        return decorate

    prange = range

#: The support, as measured. Order fixes the meaning of the coefficient rows.
OFFSETS = ((-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0))


@njit(parallel=True, cache=True, fastmath=False, boundscheck=False)
def apply_stencil_fused(u, out, s):
    """`out = K u` in one pass. `u`, `out` are `(2, H, W)`; `s` is `(7, 2, 2)`.

    Coefficients are hoisted into scalars before the loops so the inner body
    holds no array indexing beyond the neighbour loads themselves.
    """

    height = u.shape[1]
    width = u.shape[2]
    a00, a01, a10, a11 = s[0, 0, 0], s[0, 0, 1], s[0, 1, 0], s[0, 1, 1]
    b00, b01, b10, b11 = s[1, 0, 0], s[1, 0, 1], s[1, 1, 0], s[1, 1, 1]
    c00, c01, c10, c11 = s[2, 0, 0], s[2, 0, 1], s[2, 1, 0], s[2, 1, 1]
    d00, d01, d10, d11 = s[3, 0, 0], s[3, 0, 1], s[3, 1, 0], s[3, 1, 1]
    e00, e01, e10, e11 = s[4, 0, 0], s[4, 0, 1], s[4, 1, 0], s[4, 1, 1]
    f00, f01, f10, f11 = s[5, 0, 0], s[5, 0, 1], s[5, 1, 0], s[5, 1, 1]
    g00, g01, g10, g11 = s[6, 0, 0], s[6, 0, 1], s[6, 1, 0], s[6, 1, 1]
    for i in prange(1, height - 1):
        for j in range(1, width - 1):
            ax, ay = u[0, i - 1, j], u[1, i - 1, j]
            bx, by = u[0, i - 1, j + 1], u[1, i - 1, j + 1]
            cx, cy = u[0, i, j - 1], u[1, i, j - 1]
            dx, dy = u[0, i, j], u[1, i, j]
            ex, ey = u[0, i, j + 1], u[1, i, j + 1]
            fx_, fy_ = u[0, i + 1, j - 1], u[1, i + 1, j - 1]
            gx, gy = u[0, i + 1, j], u[1, i + 1, j]
            out[0, i, j] = (
                a00 * ax + a01 * ay + b00 * bx + b01 * by
                + c00 * cx + c01 * cy + d00 * dx + d01 * dy
                + e00 * ex + e01 * ey + f00 * fx_ + f01 * fy_
                + g00 * gx + g01 * gy
            )
            out[1, i, j] = (
                a10 * ax + a11 * ay + b10 * bx + b11 * by
                + c10 * cx + c11 * cy + d10 * dx + d11 * dy
                + e10 * ex + e11 * ey + f10 * fx_ + f11 * fy_
                + g10 * gx + g11 * gy
            )


@njit(cache=True, fastmath=False, boundscheck=False)
def apply_stencil_ring(u, out, s, offsets):
    """The outer ring, where a neighbour may fall outside and contribute zero.

    Four thousandths of the pixels, so clarity beats speed and the generic
    neighbour loop stays.
    """

    height = u.shape[1]
    width = u.shape[2]
    count = offsets.shape[0]
    # Walk the ring itself. Sweeping the whole grid and skipping the interior
    # costs four million serial iterations to touch sixteen thousand pixels,
    # which measured 18 ms against the kernel's 8.
    for step in range(2 * height + 2 * width - 4):
        if step < width:
            i, j = 0, step
        elif step < 2 * width:
            i, j = height - 1, step - width
        elif step < 2 * width + height - 2:
            i, j = step - 2 * width + 1, 0
        else:
            i, j = step - 2 * width - height + 3, width - 1
        fx = 0.0
        fy = 0.0
        for k in range(count):
            p = i + offsets[k, 0]
            q = j + offsets[k, 1]
            if p < 0 or p >= height or q < 0 or q >= width:
                continue
            px = u[0, p, q]
            py = u[1, p, q]
            fx += s[k, 0, 0] * px + s[k, 0, 1] * py
            fy += s[k, 1, 0] * px + s[k, 1, 1] * py
        out[0, i, j] = fx
        out[1, i, j] = fy


@njit(parallel=True, cache=True, fastmath=False, boundscheck=False)
def apply_stencil_indexed(u, out, s, offsets):
    """The same, keeping the neighbour loop. Kept only to be benchmarked."""

    height = u.shape[0]
    width = u.shape[1]
    count = offsets.shape[0]
    for i in prange(1, height - 1):
        for j in range(1, width - 1):
            fx = 0.0
            fy = 0.0
            for k in range(count):
                px = u[0, i + offsets[k, 0], j + offsets[k, 1]]
                py = u[1, i + offsets[k, 0], j + offsets[k, 1]]
                fx += s[k, 0, 0] * px + s[k, 0, 1] * py
                fy += s[k, 1, 0] * px + s[k, 1, 1] * py
            out[0, i, j] = fx
            out[1, i, j] = fy


def apply_stencil_slices(blocks, field, out):
    """The NumPy reference: seven passes, correct everywhere, no compiler.

    Kept as the fallback and as the oracle the kernel is checked against, so a
    machine without the performance extra still runs every campaign, slower.
    """

    out[...] = 0.0
    height, width = field.shape[1], field.shape[2]
    for (di, dj), block in blocks.items():
        source = field[
            :,
            max(0, -di) : height - max(0, di),
            max(0, -dj) : width - max(0, dj),
        ]
        target = out[
            :,
            max(0, di) : height - max(0, -di),
            max(0, dj) : width - max(0, -dj),
        ]
        target[0] += block[0, 0] * source[0] + block[0, 1] * source[1]
        target[1] += block[1, 0] * source[0] + block[1, 1] * source[1]
    return out


class FusedStencil:
    """Buffers allocated once, reused for every application inside the solver."""

    def __init__(self, blocks: dict[tuple[int, int], np.ndarray],
                 interior_shape: tuple[int, int, int]) -> None:
        if tuple(sorted(blocks)) != OFFSETS:
            raise ValueError(f"unexpected stencil support: {sorted(blocks)}")
        self.coefficients = np.ascontiguousarray(
            np.stack([blocks[o] for o in OFFSETS]), dtype=np.float64
        )
        self.offsets = np.asarray(OFFSETS, dtype=np.int64)
        self.blocks = dict(blocks)
        height, width = interior_shape[0], interior_shape[1]
        # One halo ring, which *is* the Dirichlet condition: the fluctuation
        # vanishes outside, so the padding stays zero for ever and the kernel
        # needs no boundary branch.
        self.output = np.zeros((2, height, width), dtype=np.float64)
        self.shape = (height, width)

    def apply(self, field: np.ndarray) -> np.ndarray:
        """`field` is `(2, H, W)` or its flat view. No copy, no allocation."""

        height, width = self.shape
        view = np.asarray(field, dtype=np.float64).reshape(2, height, width)
        if not HAS_NUMBA:
            return apply_stencil_slices(self.blocks, view, self.output).reshape(-1)
        apply_stencil_fused(view, self.output, self.coefficients)
        apply_stencil_ring(view, self.output, self.coefficients, self.offsets)
        return self.output.reshape(-1)
