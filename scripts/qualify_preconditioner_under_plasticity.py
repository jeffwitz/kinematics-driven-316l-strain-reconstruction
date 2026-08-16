#!/usr/bin/env python3
"""Milestone 1.0: does the homogeneous preconditioner survive a plastic tangent?

Milestone 0 qualified the linear operator with a homogeneous `C`, and the
preconditioner turned out to be mesh independent -- 21 iterations without
padding, 29 with, from 24 to 3599 pixels square. None of that says what happens
once a plastic zone appears and the tangent becomes spatially variable.

The elastic answer does not transfer, and an earlier note of mine implied it
did. Cubic 316L has a Voigt/Reuss bracket of 1.41 on `mu`, which is why
crystalline *elasticity* will not trouble a homogeneous reference. The plastic
tangent is another matter: its deviatoric modulus collapses toward zero as
hardening falls, so the contrast is bounded only by the hardening modulus and
can be enormous.

This is testable before a single line of Newton loop, which is the point. The
tangent takes the form the return mapping actually produces -- a rank-one
softening along the flow direction,

```text
C_alg = C - beta (C N) (C N)^T / (N^T C N),      0 <= beta < 1
```

symmetric, positive definite short of `beta = 1`, and singular along `N` in the
perfectly plastic limit. `beta` is set from a requested contrast and applied to
a chosen fraction of points with random flow directions, since what matters here
is the spectral spread the preconditioner sees rather than any particular
loading path.

What is measured is `n_CG` against plastic fraction and contrast, with the
homogeneous preconditioner left exactly as it is. If 29 becomes 300 at a
contrast of 100, the preconditioner is the next milestone and nothing else gets
written. If it stays modest, the nonlinear bench can be built on the machinery
qualified so far.
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


def softened_tangent(elasticity: np.ndarray, points: int, fraction: float,
                     contrast: float, generator: np.random.Generator) -> np.ndarray:
    """Per-point `C_alg`, rank-one softened along a random flow direction.

    The direction is deviatoric in the plane-stress Kelvin sense, since a
    plastic flow direction is, and `beta` is chosen so that the stiffness along
    it is divided by the requested contrast.
    """

    tangent = np.broadcast_to(elasticity, (points, 3, 3)).copy()
    if fraction <= 0.0 or contrast <= 1.0:
        return tangent
    chosen = generator.random(points) < fraction
    count = int(chosen.sum())
    if count == 0:
        return tangent
    direction = generator.standard_normal((count, 3))
    # Deviatoric in three dimensions, with eps_zz implied: remove the part that
    # a purely volumetric increment would carry.
    volumetric = np.array([1.0, 1.0, 0.0])
    volumetric = volumetric / np.linalg.norm(volumetric)
    direction -= (direction @ volumetric)[:, None] * volumetric[None, :]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    stressed = direction @ elasticity
    denominator = np.einsum("pi,pi->p", direction, stressed)
    beta = 1.0 - 1.0 / contrast
    tangent[chosen] -= beta * (
        stressed[:, :, None] * stressed[:, None, :] / denominator[:, None, None]
    )
    return tangent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=256)
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.0, 0.05, 0.2, 0.5, 1.0])
    parser.add_argument("--contrasts", nargs="+", type=float,
                        default=[2.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--pad-transform", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    n = arguments.pixels
    grid = StructuredGrid2D(n, n, PIXEL_SIZE_MM * n, PIXEL_SIZE_MM * n)
    operator = FullFieldPlasticOperator(
        grid, backend="fftw", workers=8, kernel="generic",
        pad_transform=arguments.pad_transform, planner="estimate",
        tolerance=arguments.tolerance, maximum_iterations=4000,
    )
    generator = np.random.default_rng(20260817)
    homogeneous = operator.elasticity.copy()
    load = operator.divergence(
        operator.stress_of(generator.standard_normal((operator.points, 3)))
    )
    print(f"{n}x{n}, {operator.size} unknowns, preconditioner unchanged\n", flush=True)

    def measure(tangent: np.ndarray) -> tuple[int, float]:
        operator.elasticity = tangent
        operator.iterations.clear()
        started = time.time()
        try:
            operator.solve(load)
            return operator.iterations[-1], time.time() - started
        except RuntimeError:
            return -1, time.time() - started

    baseline, _ = measure(np.broadcast_to(homogeneous, (operator.points, 3, 3)).copy())
    print(f"elastic reference: {baseline} iterations\n", flush=True)

    report: dict[str, object] = {
        "pixels": n, "baseline_iterations": baseline, "grid": {},
    }
    header = "  ".join(f"{c:>9.0f}" for c in arguments.contrasts)
    print(f"{'fraction':>9}  {header}   (iterations; -1 means no convergence)", flush=True)
    for fraction in arguments.fractions:
        row = []
        for contrast in arguments.contrasts:
            tangent = softened_tangent(
                homogeneous, operator.points, fraction, contrast, generator
            )
            count, seconds = measure(tangent)
            row.append(count)
            report["grid"][f"f{fraction}_c{contrast}"] = {
                "iterations": count, "seconds": seconds
            }
        print(f"{fraction:>9.2f}  " + "  ".join(f"{v:>9d}" for v in row), flush=True)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\nwrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
