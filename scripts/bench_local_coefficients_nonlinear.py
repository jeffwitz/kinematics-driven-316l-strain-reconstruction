#!/usr/bin/env python3
"""Milestone 1A/1B: the nonlinear loop with many local coefficients.

Milestone 0 qualified the linear operator and its adjoint at full field, and
milestone 1.0 showed the homogeneous preconditioner survives a plastic tangent,
saturating near 79 iterations even at a contrast of 10 000. Neither says what
the **global nonlinear loop** costs once internal variables, an algorithmic
tangent, Newton-Krylov and many spatial coefficients arrive together.

Almost none of this is new code. `DrivenJ2PlaneStressBatch` integrates
associated J2 at a *prescribed* `Delta p` with a consistent tangent, and
`solve_fixed_plastic_increment_equilibrium` already drives the global Newton
loop with a matrix-free Jacobian `J v = -B^T C_alg(x) B v`. What was missing is
that its GMRES ran unpreconditioned, which is the 1398-iteration regime measured
in milestone 0; it now accepts the spectral preconditioner qualified there.

The prescribed increment comes from a **coarse grid of local coefficients**
joined by a bilinear partition of unity:

```text
Delta p(x) = sum_j w_j(x) a_j,      a_j >= 0,      sum_j w_j = 1
```

The property under test is architectural, not physical:

```text
the coefficient count must never multiply the number of global solves
```

The field is assembled first and the mechanics solved once, so eight thousand
coefficients must cost interpolation, not eight thousand equilibria. Anything
else and the local representation cannot scale, whatever it does for accuracy.

Material parameters stay homogeneous and known. Locality lives in the plastic
representation, not in an invented map of yield stresses, which would confound
a nonlinear-solver question with a heterogeneous-identification one. The
loading is a clean synthetic uniaxial Dirichlet rather than the DIC lifting,
whose repair across `scripts/*_p43.py` is still outstanding and has no business
contaminating a solver benchmark.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from qualify_full_field_plastic_operator import FullFieldPlasticOperator
from scipy.sparse.linalg import LinearOperator

from fem_inhouse.core.driven_j2 import DrivenJ2PlaneStressBatch
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.workflows.experimental_mechanical_oracle import (
    solve_fixed_plastic_increment_equilibrium,
)

PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30


def assemble_increment(grid: StructuredGrid2D, patches: int,
                       coefficients: np.ndarray) -> np.ndarray:
    """`sum_j w_j(x) a_j` for a bilinear partition of unity, without the matrix.

    The weights are separable, `w_j(x, y) = wx_i(x) wy_k(y)`, so the field is
    two small contractions. Materialising the `(pixels, patches^2)` matrix is
    what killed the 1024-square, 4096-coefficient case: 1 048 576 x 4096 in
    float64 is 34 GB, and it has no business existing.
    """

    x = np.linspace(0.0, patches - 1.0, grid.nx)
    y = np.linspace(0.0, patches - 1.0, grid.ny)
    nodes = np.arange(patches, dtype=np.float64)
    wx = np.clip(1.0 - np.abs(x[:, None] - nodes[None, :]), 0.0, None)
    wy = np.clip(1.0 - np.abs(y[:, None] - nodes[None, :]), 0.0, None)
    wx /= wx.sum(axis=1, keepdims=True)
    wy /= wy.sum(axis=1, keepdims=True)
    square = coefficients.reshape(patches, patches)
    return np.einsum("xi,ij,yj->xy", wx, square, wy).reshape(-1)


def uniaxial_boundary(grid: StructuredGrid2D, strain: float) -> np.ndarray:
    """Affine uniaxial Dirichlet data on every node: clean, and not the DIC."""

    rows = np.arange(grid.nx + 1, dtype=np.float64) * (grid.length_x / grid.nx)
    columns = np.arange(grid.ny + 1, dtype=np.float64) * (grid.length_y / grid.ny)
    field = np.zeros((grid.nx + 1, grid.ny + 1, 2))
    field[:, :, 0] = -POISSON * strain * rows[:, None]
    field[:, :, 1] = strain * columns[None, :]
    return field


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--patches", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--increments", type=int, default=3)
    parser.add_argument("--strain-per-increment", type=float, default=1.5e-3)
    parser.add_argument("--peak-increment", type=float, default=6.0e-4)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report: dict[str, object] = {"cases": {}}
    for pixels in arguments.pixels:
        grid = StructuredGrid2D(
            pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels
        )
        kinematics = TwoSubcellDiagnostic2D(grid)
        points = kinematics.material_point_count
        # The spectral preconditioner from milestone 0, on the same grid. Its
        # reference is elastic and homogeneous; milestone 1.0 measured what a
        # plastic tangent costs it.
        spectral = FullFieldPlasticOperator(
            grid, backend="fftw", workers=8, kernel="generic",
            pad_transform=0, planner="estimate",
        )
        preconditioner = LinearOperator(
            (spectral.size, spectral.size), matvec=spectral.precondition, dtype=np.float64
        )
        for patches in arguments.patches:
            generator = np.random.default_rng(20260817)
            # Coefficients that localise: a smooth random field over the patch
            # grid, non-negative, peaking near the registered amplitude.
            coefficients = np.abs(generator.standard_normal(patches * patches))
            coefficients *= arguments.peak_increment / coefficients.max()
            per_pixel = assemble_increment(grid, patches, coefficients)
            increment = np.repeat(per_pixel, points // (grid.nx * grid.ny))

            # The unpreconditioned arm runs only at the smallest size. At 64
            # square it already took 6972 Krylov iterations against 111, and the
            # cost per iteration grows with the grid, so above that it would
            # spend the night reconfirming a settled point.
            arms = [("preconditioned", preconditioner)]
            if pixels <= min(arguments.pixels):
                arms.append(("plain", None))
            for label, conditioner in arms:
                material = DrivenJ2PlaneStressBatch(
                    points, young_modulus_mpa=YOUNG_MPA, poisson_ratio=POISSON
                )
                newton, krylov, seconds = [], [], []
                displacement = None
                failed = None
                for step in range(1, arguments.increments + 1):
                    boundary = uniaxial_boundary(
                        grid, step * arguments.strain_per_increment
                    )
                    started = time.time()
                    try:
                        result = solve_fixed_plastic_increment_equilibrium(
                            material=material,
                            kinematics=kinematics,
                            boundary_displacement=boundary,
                            equivalent_plastic_increment=increment.reshape(
                                grid.nx, grid.ny, 2
                            ),
                            initial_displacement=displacement,
                            equilibrium_rms_tolerance=1e-7,
                            maximum_krylov_iterations=3000,
                            preconditioner=conditioner,
                        )
                    except Exception as error:
                        failed = f"{type(error).__name__}: {error}"
                        break
                    seconds.append(time.time() - started)
                    newton.append(result.newton_iterations)
                    krylov.append(sum(result.krylov_iterations))
                    displacement = result.displacement
                    material.commit()
                entry = {
                    "newton": newton, "krylov_total": krylov, "seconds": seconds,
                    "coefficients": patches * patches, "failure": failed,
                }
                report["cases"][f"{pixels}_{patches}_{label}"] = entry
                summary = (
                    f"{sum(newton)} Newton, {sum(krylov)} Krylov, "
                    f"{sum(seconds):.1f} s"
                    if failed is None
                    else f"FAILED {failed[:60]}"
                )
                print(f"{pixels:>5}px  {patches * patches:>5} coeff  "
                      f"{label:>14}: {summary}", flush=True)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\nwrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
