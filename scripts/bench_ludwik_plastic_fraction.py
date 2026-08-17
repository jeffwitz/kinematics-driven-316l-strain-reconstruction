#!/usr/bin/env python3
"""Milestone 1A proper: what a real yield criterion costs the preconditioner.

The local-coefficient bench prescribed `Delta p` and never asked a point
whether it yields. A real return mapping does, and the domain then carries a
*mixture* of tangents,

```text
C_alg(x) = C_e            where f_trial <= 0
           C_ep(x)        where the return map fires
```

with moving elastic-plastic boundaries. Milestone 1.0 measured a synthetic
softening applied uniformly at random and found the iteration count saturating
near 79 even at a contrast of 10 000. That is reassuring but it is not the same
object: real plasticity is spatially organised, the transition front is sharp,
and the tangent jumps across it.

So the question is no longer whether many coefficients cost much -- they cost
nothing, measured -- but:

```text
what happens to the Krylov count as the plastic fraction sweeps 5 % to 80 %?
```

If it holds near its elastic value the homogeneous reference is far more robust
than one would dare assume. If it climbs, the next lock is the plastic tangent
contrast, and neither the problem size nor the representation.

Everything here is assembled from qualified parts: `PythonJ2PlaneStressBatch` is
the Ludwik return mapping, and `solve_two_state_dirichlet_plane_stress` is the
production Newton-GMRES with its DST-I preconditioner.

One deliberate departure from "homogeneous and known". With a uniform yield
stress under affine uniaxial loading every point yields at the same instant,
the plastic fraction jumps from nothing to everything, and the sweep does not
exist. A modest dispersion of the yield stress -- ten per cent, smooth -- is the
least that makes the question answerable. It is not a map to be identified; the
elastic constants and the hardening law stay uniform.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fem_inhouse.core.mfront_native import MFrontNativePlaneStressBatch
from fem_inhouse.core.plane_stress_material import PythonJ2PlaneStressBatch
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_ebi import EBISpectralSolverConfig
from fem_inhouse.spectral2d.newton_two_state import (
    TwoStateIncrementFields,
    solve_two_state_dirichlet_plane_stress,
)

PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30
YIELD_MPA = 260.0
HARDENING_MPA = 900.0
LUDWIK_EXPONENT = 0.32


def smooth_yield_map(grid: StructuredGrid2D, spread: float, seed: int) -> np.ndarray:
    """A gently varying yield stress, so points do not all yield at once."""

    generator = np.random.default_rng(seed)
    coarse = generator.standard_normal((8, 8))
    x = np.linspace(0.0, 7.0, grid.nx)
    y = np.linspace(0.0, 7.0, grid.ny)
    nodes = np.arange(8, dtype=np.float64)
    wx = np.clip(1.0 - np.abs(x[:, None] - nodes[None, :]), 0.0, None)
    wy = np.clip(1.0 - np.abs(y[:, None] - nodes[None, :]), 0.0, None)
    wx /= wx.sum(axis=1, keepdims=True)
    wy /= wy.sum(axis=1, keepdims=True)
    field = np.einsum("xi,ij,yj->xy", wx, coarse, wy)
    field /= max(float(np.abs(field).max()), 1e-30)
    return (YIELD_MPA * (1.0 + spread * field)).reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--peak-strain", type=float, default=2.6e-3)
    parser.add_argument("--start-strain", type=float, default=6.0e-4)
    # Ten per cent of dispersion put every point past yield in one increment and
    # there was no sweep to measure. With forty, the yield strain spans 0.76e-3
    # to 1.78e-3, so increments crossing that band take the plastic fraction up
    # gradually, which is the whole object of the experiment.
    parser.add_argument("--spread", type=float, default=0.40)
    # MFront overtakes the vectorised Python batch at four threads and is 1.9x
    # faster at eight on the plastic branch, which is where the time goes. It
    # loses on the elastic branch, where the per-point work is trivial.
    parser.add_argument("--backend", default="mfront", choices=("mfront", "python"))
    parser.add_argument("--mfront-threads", type=int, default=8)
    parser.add_argument("--library", type=Path,
                        default=Path("build/mfront/src/libBehaviour.so"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report: dict[str, object] = {"cases": {}}
    for pixels in arguments.pixels:
        grid = StructuredGrid2D(
            pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels
        )
        yield_map = smooth_yield_map(grid, arguments.spread, 20260817)
        yields = np.repeat(yield_map, 2)
        hardening = np.repeat(np.full(grid.nx * grid.ny, HARDENING_MPA), 2)
        if arguments.backend == "mfront":
            material = MFrontNativePlaneStressBatch(
                arguments.library, yields, hardening,
                np.full(yields.size, LUDWIK_EXPONENT),
                behaviour_name="PixelLudwikJ2Plasticity",
                thread_count=arguments.mfront_threads,
            )
        else:
            material = PythonJ2PlaneStressBatch(
                yields, hardening, LUDWIK_EXPONENT,
                young_modulus_mpa=YOUNG_MPA, poisson_ratio=POISSON,
            )
        columns = np.arange(grid.ny + 1, dtype=np.float64) * (grid.length_y / grid.ny)
        rows = np.arange(grid.nx + 1, dtype=np.float64) * (grid.length_x / grid.nx)
        history = np.zeros((arguments.increments + 1, *grid.node_shape, 2))
        for step in range(arguments.increments + 1):
            strain = 0.0 if step == 0 else arguments.start_strain + (
                arguments.peak_strain - arguments.start_strain
            ) * (step - 1) / max(arguments.increments - 1, 1)
            history[step, :, :, 0] = -POISSON * strain * rows[:, None]
            history[step, :, :, 1] = strain * columns[None, :]

        observations: list[dict[str, float]] = []

        def watch(fields: TwoStateIncrementFields, sink=observations) -> None:
            plastic = fields.plastic_strain_tensor
            active = 0.0
            if plastic is not None:
                magnitude = np.abs(np.asarray(plastic)).reshape(-1, 3).max(axis=1)
                active = float((magnitude > 1e-12).mean())
            sink.append({"increment": int(fields.increment), "plastic_fraction": active})

        config = EBISpectralSolverConfig()
        started = time.time()
        result = solve_two_state_dirichlet_plane_stress(
            grid=grid, material=material,
            boundary_displacement_history=history, config=config,
            increment_observer=watch,
        )
        seconds = time.time() - started
        payload = {
            "seconds": seconds,
            "plastic_fraction": [o["plastic_fraction"] for o in observations],
        }
        solves = list(getattr(result.diagnostics, "linear_solves", ()))
        if solves:
            payload["gmres_per_newton"] = [int(s.gmres_iterations) for s in solves]
            payload["gmres_seconds"] = sum(float(s.gmres_seconds) for s in solves)
            payload["jacobian_seconds"] = sum(float(s.jacobian_seconds) for s in solves)
            payload["preconditioner_seconds"] = sum(
                float(s.preconditioner_seconds) for s in solves
            )
            payload["krylov_overhead_seconds"] = sum(
                float(s.krylov_overhead_seconds) for s in solves
            )
            payload["increment_of_solve"] = [int(s.increment) for s in solves]
        diagnostics = result.diagnostics
        for name in dir(diagnostics):
            if name.startswith("_"):
                continue
            value = getattr(diagnostics, name)
            if isinstance(value, int | float | bool):
                payload[name] = value
            elif isinstance(value, (list, tuple)) and value and isinstance(
                value[0], int | float
            ):
                payload[name] = list(value)
        report["cases"][str(pixels)] = payload
        fractions = " ".join(f"{o['plastic_fraction']:.2f}" for o in observations)
        payload["backend"] = arguments.backend
        payload["mfront_threads"] = arguments.mfront_threads
        print(f"{pixels:>5}px  {arguments.backend}  {seconds:7.1f} s   "
              f"plastic fraction per increment: {fractions}",
              flush=True)
        if "gmres_per_newton" in payload:
            # GMRES per increment, beside the plastic fraction it was paid for.
            by_increment: dict[int, list[int]] = {}
            for which, count in zip(payload["increment_of_solve"],
                                    payload["gmres_per_newton"], strict=True):
                by_increment.setdefault(which, []).append(count)
            print("        increment  plastic  newton  gmres", flush=True)
            for which in sorted(by_increment):
                counts = by_increment[which]
                fraction = observations[which - 1]["plastic_fraction"] if (
                    0 < which <= len(observations)
                ) else float("nan")
                print(f"        {which:>9} {fraction:>8.2f} {len(counts):>7} "
                      f"{sum(counts):>6}", flush=True)
            print(f"        time: gmres {payload['gmres_seconds']:.1f} s, "
                  f"jacobian {payload['jacobian_seconds']:.1f} s, "
                  f"preconditioner {payload['preconditioner_seconds']:.1f} s, "
                  f"overhead {payload['krylov_overhead_seconds']:.1f} s", flush=True)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\nwrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
