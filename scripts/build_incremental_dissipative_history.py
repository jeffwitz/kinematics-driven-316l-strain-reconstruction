#!/usr/bin/env python3
"""Is there a locally dissipative plastic history that reproduces the DIC?

One question, and nothing else: does there exist an eigenstrain history that is
plastically incompressible, in equilibrium, dissipative **at every material
point of every increment**, and that reproduces the measured kinematics? No
Ludwik, no yield map, no hardening law, no shared subspace, no learned
potential. Those are later milestones and they contaminate this one.

The previous attempt could not answer it. Forcing twenty states to live in one
rank-16 spatial subspace meant a poor constrained fit was equally consistent
with "dissipative plasticity is impossible here" and with "the shared basis is
too poor", and leave-one-out had already shown the second was true. The shared
block-Krylov space was a useful reduction experiment; it is the wrong spatial
constraint for a history.

So the history is built the way an elastoplastic integrator builds one --
forward, one increment at a time, with the constraint present from birth rather
than imposed afterwards on a free field.

```text
eps_p(n-1)                      already accumulated, kept
eps_hat(n) = eps_el(n) + A eps_p(n-1)      predictor: DIC Dirichlet at n,
                                           no new plasticity
g_n = eps_DIC(n) - eps_hat(n)              what the new increment must produce
V_n = Krylov_r(A^T A, A^T g_n)             a basis of THIS increment
d eps_p(n) = V_n q_n,  eps_p(n) = eps_p(n-1) + V_n q_n
```

Each increment carries only `r` unknowns, sixteen to a hundred and twenty
eight, against the hundreds of thousands of the global formulation.

## The constraint is local and strong

In Kelvin coordinates the Euclidean dot product *is* the tensor contraction,
and plane stress makes `sigma_zz` vanish, so the in-plane dot product is the
whole of `sigma : d eps_p`. Plastic incompressibility needs no constraint
either: the plane-stress plastic triple already implies
`eps_zz = -(eps_xx + eps_yy)`, which is what `PLANE_STRESS_PLASTIC_GAUGE`
encodes. What remains is the second principle at the mid-point,

```text
D(n,g) = (sigma(n-1,g) + sigma(n,g)) / 2 . d eps_p(n,g) >= 0     for every g,
```

imposed pointwise -- not as a penalty, not as a domain average.

`sigma(n)` depends on `q_n`, so the constraint is solved by sequential
convexification: freeze the mid-point stress, solve a genuine convex QP in `r`
variables with the inequalities `G q >= 0`, recompute the mechanical stress,
measure the *true* non-linearised dissipation, iterate. Active constraints are
never turned into equalities -- that is what over-determined the earlier
active-set attempt and drove it to the trivial solution. The first pass is
unconstrained, because starting from `q = 0` is a fixed point of
freeze-then-solve: there are no violations to constrain on.

## The verdict is a curve, not a number

For each increment the rank is raised, 8, 16, 32, 64. Either the DIC error
falls with enrichment, and a dissipative history reproduces the measurement, or
it plateaus, and then -- and only then -- the conclusion is available that under
these elasticity, plane-stress and incompressibility assumptions the measured
kinematics is not reachable by a locally dissipative plastic evolution.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import osqp
from numpy.typing import NDArray
from scipy import sparse

from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_ebi import unpack_interior

FloatArray = NDArray[np.float64]

ROOT = Path(__file__).resolve().parents[1]
HISTORY_ROOT = ROOT / "validation/reference_data/dic_multistep_history_p0043_repaired_v1"
PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30


class _Identity:
    def apply(self, values: Any) -> FloatArray:
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values: Any) -> FloatArray:
        return np.asarray(values, dtype=np.float64)


def _load_history(origin: tuple[int, int], pixels: int) -> FloatArray:
    report = json.loads((HISTORY_ROOT / "report.json").read_text(encoding="utf-8"))
    bounds = list(map(int, report["solve_bounds"]))
    x0, y0 = origin
    source = np.load(HISTORY_ROOT / "repaired_history_mm.npy", mmap_mode="r", allow_pickle=False)
    return np.asarray(
        source[
            :,
            x0 - bounds[0] : x0 + pixels - bounds[0] + 1,
            y0 - bounds[2] : y0 + pixels - bounds[2] + 1,
            :,
        ],
        dtype=np.float64,
    )


def _solve_constrained(
    normal: FloatArray,
    gradient: FloatArray,
    rows: FloatArray | None,
    ridge: float,
) -> FloatArray:
    """`min 1/2 q^T P q + c^T q` subject to `rows q >= 0`, or unconstrained.

    The ridge keeps `P` numerically positive definite when the reduced normal
    matrix is nearly singular, which happens as soon as the Krylov basis has
    captured everything the increment can reach.
    """

    size = normal.shape[0]
    matrix = normal + ridge * np.trace(normal) / max(size, 1) * np.eye(size)
    if rows is None or rows.shape[0] == 0:
        return np.linalg.solve(matrix, -gradient)
    problem = osqp.OSQP()
    problem.setup(
        P=sparse.csc_matrix(matrix),
        q=gradient,
        A=sparse.csc_matrix(rows),
        l=np.zeros(rows.shape[0]),
        u=np.full(rows.shape[0], np.inf),
        verbose=False,
        eps_abs=1.0e-9,
        eps_rel=1.0e-9,
        max_iter=40_000,
        polish=True,
    )
    result = problem.solve()
    if result.x is None or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"the increment QP failed: {result.info.status}")
    return np.asarray(result.x, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", nargs=2, type=int, default=(1580, 1030))
    parser.add_argument("--pixels", type=int, default=20)
    parser.add_argument("--increments", type=int, default=10)
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--convexification-iterations", type=int, default=12)
    parser.add_argument("--ridge", type=float, default=1.0e-10)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    pixels = arguments.pixels
    origin = (int(arguments.origin[0]), int(arguments.origin[1]))
    grid = StructuredGrid2D(pixels, pixels, pixels * PIXEL_SIZE_MM, pixels * PIXEL_SIZE_MM)
    operator = TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=_Identity(),
        whitener=_Identity(),
    )
    points = operator.kinematics.material_point_count
    history = _load_history(origin, pixels)

    def elastic_extension(field: FloatArray) -> FloatArray:
        strain = operator.kelvin_strain(field)
        stress = np.einsum("pi,pij->pj", strain, operator.elasticity)
        interior = operator.solve_stiffness(operator._strain_transpose(stress.reshape(-1)))
        result = field.copy()
        result[1:-1, 1:-1, :] -= interior.reshape(pixels - 1, pixels - 1, 2)
        return result

    def response(plastic: FloatArray) -> FloatArray:
        return operator.kelvin_response(plastic)

    def transpose(observation: FloatArray) -> FloatArray:
        dual = np.asarray(observation, dtype=np.float64).reshape(-1, 3)
        interior = operator.solve_stiffness(operator._strain_transpose(dual.reshape(-1)))
        strain = operator.kelvin_strain(unpack_interior(interior, grid))
        return np.einsum("pi,pij->pj", strain, operator.elasticity).reshape(-1)

    def krylov_basis(seed: FloatArray, rank: int) -> FloatArray:
        """`span{A^T g, (A^T A) A^T g, ...}` with full re-orthogonalisation.

        Ranks stay below a few hundred, so the quadratic cost of orthogonalising
        against every previous column is irrelevant next to one stiffness solve,
        and it keeps the basis usable where a Lanczos recurrence would have lost
        orthogonality long before the last column.
        """

        vector = transpose(seed)
        scale = float(np.linalg.norm(vector))
        columns: list[FloatArray] = []
        for _ in range(rank):
            for existing in columns:
                vector = vector - (existing @ vector) * existing
            norm = float(np.linalg.norm(vector))
            if norm <= 1.0e-13 * max(scale, 1.0e-30):
                break
            vector = vector / norm
            columns.append(vector)
            vector = transpose(response(vector))
        return np.asarray(columns).T

    measured = {
        n: operator.kelvin_strain(history[n]).reshape(-1)
        for n in range(arguments.increments + 1)
    }
    elastic = {
        n: operator.kelvin_strain(elastic_extension(history[n])).reshape(-1)
        for n in range(arguments.increments + 1)
    }

    per_rank: dict[str, Any] = {}
    for rank in arguments.ranks:
        started = time.perf_counter()
        plastic = np.zeros(3 * points)
        simulated = elastic[0]
        stress = np.einsum(
            "pi,pij->pj", (simulated - plastic).reshape(-1, 3), operator.elasticity
        )
        increments: list[dict[str, Any]] = []
        path_length = 0.0
        for n in range(1, arguments.increments + 1):
            predictor = elastic[n] + response(plastic)
            gap = measured[n] - predictor
            basis = krylov_basis(gap, rank)
            if basis.shape[1] == 0:
                raise SystemExit(f"the Krylov basis collapsed at increment {n}")
            applied = np.asarray(
                [response(basis[:, k]) for k in range(basis.shape[1])]
            ).T
            normal = applied.T @ applied
            gradient = -(applied.T @ gap)

            rows: FloatArray | None = None
            coefficients = np.zeros(basis.shape[1])
            iterations = 0
            negative_fraction = 1.0
            minimum_dissipation = -np.inf
            unconstrained_misfit = float("nan")
            unconstrained_negative = float("nan")
            for attempt in range(1, arguments.convexification_iterations + 1):
                iterations = attempt
                coefficients = _solve_constrained(normal, gradient, rows, arguments.ridge)
                step = (basis @ coefficients).reshape(-1, 3)
                candidate_plastic = plastic.reshape(-1, 3) + step
                candidate_simulated = predictor + applied @ coefficients
                candidate_stress = np.einsum(
                    "pi,pij->pj",
                    candidate_simulated.reshape(-1, 3) - candidate_plastic,
                    operator.elasticity,
                )
                mid = 0.5 * (stress + candidate_stress)
                dissipation = np.einsum("pi,pi->p", mid, step)
                negative_fraction = float(np.mean(dissipation < 0.0))
                minimum_dissipation = float(dissipation.min())
                if attempt == 1:
                    # The free solution of this increment. Without it a q of
                    # zero is unreadable: it can mean the basis cannot reach the
                    # residual, or that the pointwise constraint leaves the cone
                    # with no interior. Those demand opposite responses.
                    unconstrained_misfit = float(
                        np.linalg.norm(applied @ coefficients - gap)
                        / max(float(np.linalg.norm(gap)), 1.0e-30)
                    )
                    unconstrained_negative = negative_fraction
                if negative_fraction == 0.0:
                    break
                # Freeze the mid-point stress and turn the second principle into
                # linear inequalities on q: one row per material point, the row
                # being that point's frozen stress contracted with its slice of
                # the basis. Inequalities only, never equalities.
                rows = np.einsum(
                    "pi,pir->pr", mid, basis.reshape(points, 3, basis.shape[1])
                )

            step = (basis @ coefficients).reshape(-1, 3)
            plastic = plastic + basis @ coefficients
            simulated = predictor + applied @ coefficients
            stress = np.einsum(
                "pi,pij->pj",
                simulated.reshape(-1, 3) - plastic.reshape(-1, 3),
                operator.elasticity,
            )
            step_norm = float(np.linalg.norm(step))
            path_length += step_norm
            increments.append(
                {
                    "increment": n,
                    "increment_misfit": float(
                        np.linalg.norm(applied @ coefficients - gap)
                        / max(float(np.linalg.norm(gap)), 1.0e-30)
                    ),
                    "cumulative_dic_error": float(
                        np.linalg.norm(simulated - measured[n])
                        / max(float(np.linalg.norm(measured[n])), 1.0e-30)
                    ),
                    "elastic_only_dic_error": float(
                        np.linalg.norm(elastic[n] - measured[n])
                        / max(float(np.linalg.norm(measured[n])), 1.0e-30)
                    ),
                    "unconstrained_misfit": unconstrained_misfit,
                    "unconstrained_negative_fraction": unconstrained_negative,
                    "constraint_rows": 0 if rows is None else int(rows.shape[0]),
                    "negative_dissipation_fraction": negative_fraction,
                    "minimum_dissipation": minimum_dissipation,
                    "convexification_iterations": iterations,
                    "plastic_increment_norm": step_norm,
                    "basis_columns": int(basis.shape[1]),
                }
            )
            print(
                f"rank {rank:3d} increment {n:2d}: free misfit "
                f"{unconstrained_misfit:.4f} (neg {unconstrained_negative:.3f}) "
                f"-> constrained {increments[-1]['increment_misfit']:.4f}  cumulative "
                f"{increments[-1]['cumulative_dic_error']:.4f}  negative "
                f"{negative_fraction:.4f}  iterations {iterations}",
                flush=True,
            )

        net = float(np.linalg.norm(plastic))
        per_rank[str(rank)] = {
            "increments": increments,
            "final_cumulative_dic_error": increments[-1]["cumulative_dic_error"],
            "worst_negative_fraction": max(
                entry["negative_dissipation_fraction"] for entry in increments
            ),
            "plastic_path_length": path_length,
            "plastic_net_norm": net,
            "path_over_net": path_length / max(net, 1.0e-30),
            "elapsed_seconds": time.perf_counter() - started,
        }
        print(
            f"rank {rank}: final DIC error "
            f"{per_rank[str(rank)]['final_cumulative_dic_error']:.4f}, worst negative "
            f"fraction {per_rank[str(rank)]['worst_negative_fraction']:.4f}, path/net "
            f"{per_rank[str(rank)]['path_over_net']:.2f}\n",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "status": "completed_incremental_dissipative_history",
        "question": (
            "does a locally dissipative, plastically incompressible eigenstrain "
            "history in equilibrium reproduce the measured kinematics"
        ),
        "constitutive_inputs": "none",
        "origin_nodes": list(origin),
        "mesh": [pixels, pixels],
        "increments": arguments.increments,
        "material_points": points,
        "dissipation": "mid-point, pointwise, Kelvin dot product, inequality only",
        "per_rank": per_rank,
        "verdict_curve": {
            str(rank): per_rank[str(rank)]["final_cumulative_dic_error"]
            for rank in arguments.ranks
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"verdict curve: {report['verdict_curve']}\nwrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
