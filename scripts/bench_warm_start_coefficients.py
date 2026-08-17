#!/usr/bin/env python3
"""What a coefficient perturbation costs in the warm regime.

The nonlinear bench measured cold solves: eight Newton iterations from an
undeformed start. That is not what an optimisation loop does. There,

```text
a^(k+1) = a^k + alpha da,      |alpha da| << |a|
```

and the previous displacement is an excellent opening guess, so the honest
budget for training is the cost of a *perturbation*, not of a cold solve.

The committed material state is deliberately not advanced between the reference
solve and the perturbed one: both start from the same history, which is exactly
the situation an optimiser is in when it tries a new coefficient vector at the
same load step.

Measured against `alpha`, with and without the warm start, on the same
increment: Newton iterations, Krylov totals and wall time. If eight Newton
become two, the arithmetic of full-field training changes character.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from bench_local_coefficients_nonlinear import assemble_increment, uniaxial_boundary
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--patches", type=int, default=16)
    parser.add_argument("--settle-increments", type=int, default=3)
    parser.add_argument("--strain-per-increment", type=float, default=1.5e-3)
    parser.add_argument("--peak-increment", type=float, default=6.0e-4)
    parser.add_argument("--alphas", nargs="+", type=float, default=[1e-1, 1e-2, 1e-3])
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report: dict[str, object] = {"cases": {}}
    for pixels in arguments.pixels:
        grid = StructuredGrid2D(
            pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels
        )
        kinematics = TwoSubcellDiagnostic2D(grid)
        points = kinematics.material_point_count
        spectral = FullFieldPlasticOperator(
            grid, backend="fftw", workers=8, kernel="generic",
            pad_transform=0, planner="estimate",
        )
        preconditioner = LinearOperator(
            (spectral.size, spectral.size), matvec=spectral.precondition, dtype=np.float64
        )
        generator = np.random.default_rng(20260817)
        patches = arguments.patches
        coefficients = np.abs(generator.standard_normal(patches * patches))
        coefficients *= arguments.peak_increment / coefficients.max()
        direction = generator.standard_normal(patches * patches)

        def field(values: np.ndarray, g=grid, p=patches, n=points) -> np.ndarray:
            per_pixel = np.maximum(assemble_increment(g, p, values), 0.0)
            return np.repeat(per_pixel, n // (g.nx * g.ny)).reshape(g.nx, g.ny, 2)

        material = DrivenJ2PlaneStressBatch(
            points, young_modulus_mpa=YOUNG_MPA, poisson_ratio=POISSON
        )
        displacement = None
        for step in range(1, arguments.settle_increments + 1):
            result = solve_fixed_plastic_increment_equilibrium(
                material=material, kinematics=kinematics,
                boundary_displacement=uniaxial_boundary(
                    grid, step * arguments.strain_per_increment
                ),
                equivalent_plastic_increment=field(coefficients),
                initial_displacement=displacement,
                equilibrium_rms_tolerance=1e-7, maximum_krylov_iterations=3000,
                preconditioner=preconditioner,
            )
            displacement = result.displacement
            material.commit()
        settled = displacement
        boundary = uniaxial_boundary(
            grid, (arguments.settle_increments + 1) * arguments.strain_per_increment
        )
        # The reference solve at the next load step, itself warm from the last
        # increment. Everything below perturbs the coefficients around it
        # without advancing the committed state.
        reference = solve_fixed_plastic_increment_equilibrium(
            material=material, kinematics=kinematics,
            boundary_displacement=boundary,
            equivalent_plastic_increment=field(coefficients),
            initial_displacement=settled,
            equilibrium_rms_tolerance=1e-7, maximum_krylov_iterations=3000,
            preconditioner=preconditioner,
        )
        print(f"{pixels}px reference solve: {reference.newton_iterations} Newton, "
              f"{sum(reference.krylov_iterations)} Krylov", flush=True)
        print(f"{'alpha':>8} {'warm N':>7} {'warm K':>8} {'warm s':>8} "
              f"{'cold N':>7} {'cold K':>8} {'cold s':>8}", flush=True)
        rows: dict[str, object] = {}
        for alpha in arguments.alphas:
            trial = coefficients + alpha * arguments.peak_increment * direction
            increment = field(trial)
            row = {}
            for label, start in (("warm", reference.displacement), ("cold", None)):
                started = time.time()
                outcome = solve_fixed_plastic_increment_equilibrium(
                    material=material, kinematics=kinematics,
                    boundary_displacement=boundary,
                    equivalent_plastic_increment=increment,
                    initial_displacement=start,
                    equilibrium_rms_tolerance=1e-7, maximum_krylov_iterations=3000,
                    preconditioner=preconditioner,
                )
                row[label] = {
                    "newton": outcome.newton_iterations,
                    "krylov": sum(outcome.krylov_iterations),
                    "seconds": time.time() - started,
                }
            rows[f"alpha_{alpha:g}"] = row
            print(f"{alpha:>8.0e} {row['warm']['newton']:>7} {row['warm']['krylov']:>8} "
                  f"{row['warm']['seconds']:>8.1f} {row['cold']['newton']:>7} "
                  f"{row['cold']['krylov']:>8} {row['cold']['seconds']:>8.1f}", flush=True)
        report["cases"][str(pixels)] = {
            "reference_newton": reference.newton_iterations,
            "reference_krylov": sum(reference.krylov_iterations),
            "perturbations": rows,
        }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\nwrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
