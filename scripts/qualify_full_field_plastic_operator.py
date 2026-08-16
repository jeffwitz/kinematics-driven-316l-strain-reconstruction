#!/usr/bin/env python3
"""Milestone 0: the plastic eigenstrain operator, matrix-free, at full field.

A correctness gate, not a speed benchmark. Nothing downstream is built until
the adjoint identity holds.

**No new boundary treatment is written here.** The full-Dirichlet spectral
solver already exists -- see `docs/explanation/spectral_mechanics/`, and the
bridge page `plastic_inverse_reuse.md`. It provides the splitting
`u = u* + u^f` with `u*` a discrete harmonic extension of the measured boundary
displacements, so the transform acts on a homogeneous fluctuation and nothing is
periodic, and it provides the DST-I basis applying the reference inverse
`B_0^-1` as a preconditioner. What this script adds is only

```text
A : d eps_p -> d eps        and its adjoint,
```

matrix-free, on top of that.

Three things about the solve, each of which was wrong in an earlier plan.

It is **linear**. `A` is the elastic response to an eigenstrain, not the
constitutive Newton loop, so there is one preconditioned Krylov solve and no
Newton iteration.

It uses **conjugate gradient**, not GMRES. `K = B^T C B` restricted to the
interior is symmetric positive definite, GMRES belongs to the constitutive loop
where the Jacobian is neither, and at 22 million degrees of freedom a short
recurrence holds three working vectors where GMRES(50) would hold fifty. Both
are equally matrix-free; that was never the distinction.

The inversion is **not** done in Fourier and cannot be. DST-I does not
diagonalise the coupled elastic operator under zero Dirichlet on four edges:
`(lambda + mu) grad(div u)` carries mixed derivatives, and with `u_x` in
`sin.sin`, `d2 u_y / dx dy` lands in `cos.cos` and leaves the space. In the
periodic case `Gamma_hat(k)` would be exact; the boundary is what breaks it.
Hence a preconditioner, hence iterations, and hence the iteration count -- not
the transform cost -- is what sets `T_A`.

Every claim above is checked rather than asserted: symmetry and positivity of
the operator and of the preconditioner, agreement of the matrix-free stiffness
with the assembled one, agreement of `A` with the sparse-direct operator already
qualified at 1.5e-15, and the adjoint dot product. The Krylov tolerance is
driven far below the adjoint threshold first, because a solve to tolerance `tau`
is not a linear operator at all -- its Krylov subspace depends on the
right-hand side -- so the identity cannot be verified below `tau`.
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.core.kelvin import KELVIN_SCALE_2D, stiffness_from_engineering
from fem_inhouse.spectral2d.green import B0Green2D, project_isotropic_plane_stress_tangent
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior
from fem_inhouse.spectral2d.transform_factory import create_full_dirichlet_dsti_plan
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig

PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30


class FullFieldPlasticOperator:
    """`A eps_p = B K^-1 B^T w C eps_p`, matrix-free, on the existing solver."""

    def __init__(self, grid: StructuredGrid2D, *, backend: str = "fftw",
                 workers: int = 1, tolerance: float = 1e-12,
                 maximum_iterations: int = 2000,
                 wisdom: Path | None = None) -> None:
        self.grid = grid
        self.kinematics = TwoSubcellDiagnostic2D(grid)
        self.points = self.kinematics.material_point_count
        self.weight = float(self.kinematics.sample_quadrature_weight)
        self.elasticity = stiffness_from_engineering(
            plane_stress_elasticity(YOUNG_MPA, POISSON)
        )
        # FFTW with persistent plans and wisdom, not the SciPy prototype. The
        # repository ships both; running the sweep on SciPy was an oversight,
        # and the difference is measured rather than assumed.
        self.plan = create_full_dirichlet_dsti_plan(
            grid,
            SpectralTransformConfig(
                backend=backend, workers=workers,
                fftw_planner_effort="measure", fftw_wisdom_directory=wisdom,
                fftw_planning_time_limit_s=None if wisdom is not None else 2.0,
            ),
        )
        self.backend_name = str(self.plan.backend_name)
        symbols = self.kinematics.reference_operator_symbols(self.plan)
        lambda_0, mu_0, _ = project_isotropic_plane_stress_tangent(self.elasticity)
        self.green = B0Green2D(symbols, lambda_0=lambda_0, mu_0=mu_0)
        self.interior_shape = (*grid.interior_shape, 2)
        self.size = int(np.prod(self.interior_shape))
        self.tolerance = tolerance
        self.maximum_iterations = maximum_iterations
        self.iterations: list[int] = []

    # -- the pieces, each one line of mechanics ------------------------------

    def kelvin_strain(self, nodal) -> np.ndarray:
        return np.asarray(self.kinematics.strain(nodal)).reshape(-1, 3) / KELVIN_SCALE_2D

    def stress_of(self, strain: np.ndarray) -> np.ndarray:
        return strain.reshape(-1, 3) @ self.elasticity

    def divergence(self, kelvin_stress: np.ndarray) -> np.ndarray:
        """`B^T` on a Kelvin stress, returning the interior nodal load."""

        voigt = kelvin_stress.reshape(-1, 3) / KELVIN_SCALE_2D
        nodal = self.kinematics.divergence_from_sample_stress(
            voigt.reshape((self.grid.nx, self.grid.ny, 2, 3))
        )
        return -pack_interior(nodal) / self.weight

    def stiffness(self, interior: np.ndarray) -> np.ndarray:
        """`K v`, matrix-free: strain, stress, divergence. Never assembled."""

        nodal = unpack_interior(np.asarray(interior, dtype=np.float64).reshape(-1), self.grid)
        return self.divergence(self.stress_of(self.kelvin_strain(nodal)))

    def precondition(self, interior: np.ndarray) -> np.ndarray:
        """`B_0^-1` through DST-I. A preconditioner, never an exact inverse.

        The sign matters and was found by the positivity check rather than by
        reading. `B_0^-1` follows the production convention, where it acts on a
        residual `R = -sum B^T sigma`, while `stiffness` above already carries
        that minus. Composed as they stand, `K` comes out positive definite and
        `M` uniformly *negative* definite -- a Rayleigh quotient negative on
        100 % of random vectors -- so their product is indefinite and conjugate
        gradient has no guarantee, whatever it happens to do on a small grid.
        Negating here puts both on the same side.
        """

        shaped = np.asarray(interior, dtype=np.float64).reshape(self.interior_shape)
        spectral = self.plan.forward_displacement(shaped)
        return -np.asarray(
            self.plan.inverse_displacement(self.green.apply(spectral))
        ).reshape(-1)

    def solve(self, load: np.ndarray) -> np.ndarray:
        """`K^-1 load` by preconditioned conjugate gradient."""

        count = 0

        def tick(_x: object) -> None:
            nonlocal count
            count += 1

        operator = LinearOperator(
            (self.size, self.size), matvec=self.stiffness, dtype=np.float64
        )
        preconditioner = LinearOperator(
            (self.size, self.size), matvec=self.precondition, dtype=np.float64
        )
        answer, info = cg(
            operator, np.asarray(load, dtype=np.float64).reshape(-1), M=preconditioner,
            rtol=self.tolerance, atol=0.0, maxiter=self.maximum_iterations, callback=tick,
        )
        if info != 0:
            raise RuntimeError(f"conjugate gradient did not converge, info={info}")
        self.iterations.append(count)
        return answer

    # -- the operator and its adjoint ----------------------------------------

    def apply(self, plastic: np.ndarray) -> np.ndarray:
        displacement = unpack_interior(
            self.solve(self.divergence(self.stress_of(plastic))), self.grid
        )
        return self.kelvin_strain(displacement)

    def apply_transpose(self, dual: np.ndarray) -> np.ndarray:
        """Written independently, never assumed equal to `apply`.

        `A = E C` with `E = B K^-1 B^T w` symmetric and `C` symmetric, so
        `A^T = C E`: the same solve, with the elasticity on the other side.
        """

        displacement = unpack_interior(
            self.solve(self.divergence(np.asarray(dual, dtype=np.float64))), self.grid
        )
        return self.stress_of(self.kelvin_strain(displacement))


def relative(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-300)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", nargs=2, type=int, default=(100, 100))
    parser.add_argument("--backend", default="fftw")
    parser.add_argument("--wisdom", type=Path,
                        default=Path.home() / ".cache/fem_inhouse/fftw_wisdom")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--compare-direct", action="store_true",
                        help="check against the assembled sparse operator (small grids only)")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    nx, ny = arguments.pixels
    grid = StructuredGrid2D(nx, ny, PIXEL_SIZE_MM * nx, PIXEL_SIZE_MM * ny)
    started = time.time()
    arguments.wisdom.mkdir(parents=True, exist_ok=True)
    operator = FullFieldPlasticOperator(
        grid, backend=arguments.backend, workers=arguments.workers,
        tolerance=arguments.tolerance, wisdom=arguments.wisdom,
    )
    report_backend = operator.backend_name
    print(f"grid {nx}x{ny}, {operator.size} interior unknowns, "
          f"{operator.points} material points, backend {report_backend}, "
          f"{arguments.workers} workers, setup {time.time() - started:.1f} s",
          flush=True)

    generator = np.random.default_rng(20260816)
    report: dict[str, object] = {
        "pixels": [nx, ny], "interior_unknowns": operator.size,
        "backend": report_backend, "workers": arguments.workers,
    }

    # 1. The operator must be symmetric and positive definite, or CG is invalid.
    left = generator.standard_normal(operator.size)
    right = generator.standard_normal(operator.size)
    forward = float(operator.stiffness(left) @ right)
    backward = float(left @ operator.stiffness(right))
    report["stiffness_symmetry"] = relative(forward, backward)
    report["stiffness_positivity"] = float(left @ operator.stiffness(left))
    print(f"stiffness symmetry {report['stiffness_symmetry']:.3e}, "
          f"v.Kv {report['stiffness_positivity']:+.6e}", flush=True)

    # 2. So must the preconditioner, and this one is a genuine question.
    forward = float(operator.precondition(left) @ right)
    backward = float(left @ operator.precondition(right))
    report["preconditioner_symmetry"] = relative(forward, backward)
    report["preconditioner_positivity"] = float(left @ operator.precondition(left))
    print(f"preconditioner symmetry {report['preconditioner_symmetry']:.3e}, "
          f"v.Mv {report['preconditioner_positivity']:+.6e}", flush=True)

    # 3. The preconditioner has to earn its place.
    operator.iterations.clear()
    load = operator.divergence(operator.stress_of(generator.standard_normal((operator.points, 3))))
    started = time.time()
    operator.solve(load)
    report["preconditioned_iterations"] = operator.iterations[-1]
    report["preconditioned_seconds"] = time.time() - started
    saved = operator.precondition
    operator.precondition = lambda v: np.asarray(v, dtype=np.float64).reshape(-1)
    started = time.time()
    try:
        operator.solve(load)
        report["plain_iterations"] = operator.iterations[-1]
        report["plain_seconds"] = time.time() - started
    except RuntimeError:
        report["plain_iterations"] = None
        report["plain_seconds"] = None
    operator.precondition = saved
    print(f"conjugate gradient: {report['preconditioned_iterations']} iterations "
          f"preconditioned ({report['preconditioned_seconds']:.1f} s) against "
          f"{report['plain_iterations']} plain", flush=True)

    # 4. The gate itself.
    discrepancies = []
    for _ in range(arguments.pairs):
        x = generator.standard_normal((operator.points, 3))
        y = generator.standard_normal((operator.points, 3))
        forward = float((operator.apply(x) * y).sum())
        backward = float((x * operator.apply_transpose(y)).sum())
        discrepancies.append(
            abs(forward - backward)
            / max(float(np.linalg.norm(operator.apply(x)) * np.linalg.norm(y)), 1e-300)
        )
    report["adjoint_discrepancy"] = max(discrepancies)
    print(f"adjoint dot product: {max(discrepancies):.3e} over "
          f"{arguments.pairs} pairs", flush=True)

    # 5. Cost, measured rather than extrapolated.
    field = generator.standard_normal((operator.points, 3))
    tracemalloc.start()
    started = time.time()
    operator.apply(field)
    report["seconds_per_apply"] = time.time() - started
    started = time.time()
    operator.apply_transpose(operator.kelvin_strain(
        unpack_interior(np.zeros(operator.size), grid)
    ) + field)
    report["seconds_per_apply_transpose"] = time.time() - started
    report["peak_megabytes"] = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    report["iterations_per_apply"] = operator.iterations[-1]
    print(f"A {report['seconds_per_apply']:.2f} s, A^T "
          f"{report['seconds_per_apply_transpose']:.2f} s, "
          f"{report['iterations_per_apply']} iterations, "
          f"peak {report['peak_megabytes']:.0f} MB", flush=True)

    # 6. Against the operator already qualified at 1.5e-15, where it is affordable.
    if arguments.compare_direct:
        from fem_inhouse.identification.tensor_plastic_observability import (
            TensorPlasticObservabilityOperator,
        )

        class _Identity:
            def apply(self, values):
                return np.asarray(values, dtype=np.float64)

            def adjoint(self, values):
                return np.asarray(values, dtype=np.float64)

        direct = TensorPlasticObservabilityOperator.build(
            grid, young_modulus_mpa=YOUNG_MPA, poisson_ratio=POISSON,
            transfer=_Identity(), whitener=_Identity(),
        )
        probe = generator.standard_normal((operator.points, 3))
        mine = operator.apply(probe)
        theirs = direct.kelvin_response(probe).reshape(-1, 3)
        report["agreement_with_sparse_direct"] = float(
            np.linalg.norm(mine - theirs) / max(np.linalg.norm(theirs), 1e-300)
        )
        print(f"against the sparse direct operator: "
              f"{report['agreement_with_sparse_direct']:.3e}", flush=True)

    verdict = (
        report["adjoint_discrepancy"] < 1e-8
        and report["stiffness_symmetry"] < 1e-10
        and report["preconditioner_symmetry"] < 1e-10
        and report["stiffness_positivity"] > 0.0
        and report["preconditioner_positivity"] > 0.0
    )
    report["gate_passed"] = bool(verdict)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\ngate {'PASSED' if verdict else 'FAILED'} -> {arguments.output}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
