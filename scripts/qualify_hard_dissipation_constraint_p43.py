#!/usr/bin/env python3
"""Is there a dissipative trajectory that still reproduces the DIC?

The free inversion reproduces each measured increment and wanders: its
dissipation splits near 50/50 and its path is four to five times longer than its
net displacement. That says the representative chosen by the kinematic criterion
alone is not a plastic history. It does not say no dissipative representative
exists, because the inverse problem is massively non-unique.

The penalty attempt could not answer it: a squared penalty on `min(D, 0)` has a
trivial minimiser at zero, so the optimiser shrank the field instead of
reorienting it. This imposes the constraint **hard**:

```text
min || A Phi a - r ||^2   subject to   D_k(q) >= 0 for every point and step,
```

with the dissipation of the previous correction --- mid-point rule, absolute
stress including `sigma_20`, computed on every state rather than in long jumps.

`sigma` depends on `a`, so the constraints are quadratic. They are handled by
sequential convexification: freeze `sigma` at the current iterate, solve the
linearly-constrained problem, refreeze. And there are four hundred thousand of
them for three hundred and twenty unknowns, so they are introduced by cutting
planes --- almost none will be active at the solution.

The number that matters is not feasibility itself but its **price**: how much
agreement with the DIC has to be given up to buy a thermodynamically admissible
history. A small price means the free solution was merely an unlucky
representative; a large one means the kinematics cannot be produced by a
dissipative plastic history in this subspace.

**This script does not yet produce that number, and no result should be quoted
from it.** Two failures, both recorded because they define what the working
version needs.

`trust-constr` did not finish in ten minutes on 320 unknowns with a few thousand
constraints. Replacing it by an active set solved through a dense KKT system was
fast for small sets but is cubic in the number of cuts, so admitting thousands
is slower again than the solver it replaced.

Worse, the add-only active set is wrong once the cuts outnumber the unknowns.
Treating every active inequality as an equality then over-determines the system
-- singular KKT, and in the limit it forces the trivial solution, which is
exactly the collapse the penalty formulation already produced for a different
reason. A correct method has to keep at most `n_unknowns` constraints active and
drop those whose multipliers turn negative.

What is needed is a genuine QP: `min 1/2 a^T H a - b^T a` subject to `G a >= 0`
with multiplier-based dropping, or an off-the-shelf solver. Feasibility itself is
never in doubt -- `a = 0` gives zero increments and zero dissipation -- so the
whole content is the price, and only a real QP measures it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fem_inhouse.core.constitutive import PLANE_STRESS_VON_MISES_METRIC
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior

ROOT = Path(__file__).resolve().parents[1]
HISTORY = (
    ROOT
    / "validation/reference_data/dic_multistep_history_p0043_repaired_v1"
    / "repaired_history_mm.npy"
)
PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30


class _Identity:
    def apply(self, values):
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values):
        return np.asarray(values, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", nargs=2, type=int, default=(1580, 1030))
    parser.add_argument("--pixels", type=int, default=100)
    parser.add_argument("--reference-state", type=int, default=20)
    parser.add_argument("--states", nargs="+", type=int, default=list(range(21, 41)))
    parser.add_argument("--basis-states", nargs="+", type=int, default=[25, 30, 35, 40])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--outer", type=int, default=3)
    parser.add_argument("--cutting-rounds", type=int, default=6)
    parser.add_argument("--constraints-per-round", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    pixels = arguments.pixels
    x0, y0 = arguments.origin
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    operator = TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=_Identity(),
        whitener=_Identity(),
    )
    points = pixels * pixels * 2

    def extension(field):
        strain = np.asarray(operator.kinematics.strain(field)).reshape(-1, 3)
        stress = np.einsum("pi,pij->pj", strain, operator.elasticity)
        forcing = (
            -pack_interior(
                operator.kinematics.divergence_from_sample_stress(
                    stress.reshape((pixels, pixels, 2, 3))
                )
            )
            / operator.quadrature_weight
        )
        result = field.copy()
        result[1:-1, 1:-1, :] -= operator.solve_stiffness(forcing).reshape(
            pixels - 1, pixels - 1, 2
        )
        return result

    def response(plastic):
        stress = np.einsum(
            "pi,pij->pj", np.asarray(plastic).reshape(-1, 3), operator.elasticity
        )
        displacement = unpack_interior(
            operator.solve_stiffness(operator._strain_transpose(stress.reshape(-1))), grid
        )
        return np.asarray(operator.kinematics.strain(displacement)).reshape(-1)

    def transpose(observation):
        dual = np.asarray(observation, dtype=np.float64).reshape(-1, 3)
        displacement = unpack_interior(
            operator.solve_stiffness(operator._strain_transpose(dual.reshape(-1))), grid
        )
        strain = np.asarray(operator.kinematics.strain(displacement)).reshape(-1, 3)
        return np.einsum("pi,pij->pj", strain, operator.elasticity).reshape(-1)

    report = json.loads((HISTORY.with_name("report.json")).read_text(encoding="utf-8"))
    bounds = list(map(int, report["solve_bounds"]))
    source = np.load(HISTORY, mmap_mode="r", allow_pickle=False)
    history = np.asarray(
        source[
            :,
            x0 - bounds[0] : x0 + pixels - bounds[0] + 1,
            y0 - bounds[2] : y0 + pixels - bounds[2] + 1,
            :,
        ],
        dtype=np.float64,
    )
    reference = history[arguments.reference_state]

    def target_of(state):
        field = history[state] - reference
        measured = np.asarray(operator.kinematics.strain(field)).reshape(-1)
        elastic = np.asarray(operator.kinematics.strain(extension(field))).reshape(-1)
        return measured - elastic, elastic

    block = np.asarray([transpose(target_of(state)[0]) for state in arguments.basis_states]).T
    basis, _ = np.linalg.qr(block)
    columns, total = [basis], basis.shape[1]
    while total < arguments.rank:
        current = np.asarray(
            [transpose(response(columns[-1][:, k])) for k in range(columns[-1].shape[1])]
        ).T
        orthonormal, _ = np.linalg.qr(np.concatenate([*columns, current], axis=1))
        addition = orthonormal[:, total:]
        if addition.shape[1] == 0:
            break
        columns.append(addition)
        total += addition.shape[1]
    subspace = np.concatenate(columns, axis=1)[:, : arguments.rank]
    rank = subspace.shape[1]
    design = np.asarray([response(subspace[:, k]) for k in range(rank)]).T

    states = arguments.states
    count = len(states)
    targets, elastics = [], []
    for state in states:
        target, elastic = target_of(state)
        targets.append(target)
        elastics.append(elastic)
    targets = np.asarray(targets)
    elastics = np.asarray(elastics)

    # Reduced normal equations: the objective is cheap in the 16-dimensional space.
    hessian = design.T @ design
    linear = targets @ design
    constants = float((targets**2).sum())
    references = np.linalg.norm(targets, axis=1)

    absolute = np.asarray(operator.kinematics.strain(reference)).reshape(-1, 3)
    stress_20 = np.einsum("pi,pij->pj", absolute, operator.elasticity)
    modes = subspace.reshape(points, 3, rank)

    def unconstrained_solution():
        return np.linalg.solve(hessian, linear.T).T

    def objective(flat):
        coefficients = flat.reshape(count, rank)
        value = constants + float(
            np.einsum("nr,rs,ns->", coefficients, hessian, coefficients)
        ) - 2.0 * float((coefficients * linear).sum())
        gradient = 2.0 * (coefficients @ hessian - linear)
        return value, gradient.reshape(-1)

    def stresses(coefficients):
        result = np.empty((count, points, 3), dtype=np.float64)
        for index in range(count):
            plastic = subspace @ coefficients[index]
            simulated = elastics[index] + design @ coefficients[index]
            result[index] = stress_20 + np.einsum(
                "pi,pij->pj", (simulated - plastic).reshape(-1, 3), operator.elasticity
            )
        return result

    def projections(coefficients):
        """Per-point, per-step linear form of the mid-point dissipation."""

        field = stresses(coefficients)
        previous = stress_20
        rows = np.empty((count, points, rank), dtype=np.float64)
        for index in range(count):
            mid = 0.5 * (previous + field[index])
            rows[index] = np.einsum("pi,pir->pr", mid, modes)
            previous = field[index]
        return rows

    def dissipation(coefficients, rows):
        increments = np.diff(
            np.concatenate([np.zeros((1, rank)), coefficients], axis=0), axis=0
        )
        return np.einsum("npr,nr->np", rows, increments)

    def constraint_matrix(rows, indices):
        """Rows of `D >= 0` for the selected (step, point) pairs, in flat coordinates."""

        matrix = np.zeros((len(indices), count * rank), dtype=np.float64)
        for row, (step, point) in enumerate(indices):
            matrix[row, step * rank : (step + 1) * rank] += rows[step, point]
            if step > 0:
                matrix[row, (step - 1) * rank : step * rank] -= rows[step, point]
        return matrix

    coefficients = unconstrained_solution()
    free_errors = [
        float(np.linalg.norm(design @ coefficients[index] - targets[index]) / references[index])
        for index in range(count)
    ]
    rows = projections(coefficients)
    power = dissipation(coefficients, rows)
    print(
        f"unconstrained: mean error {np.mean(free_errors):.5f}, "
        f"negative points {np.mean(power < 0.0):.4f}, "
        f"negative share of power {np.abs(np.minimum(power, 0.0)).sum() / np.abs(power).sum():.4f}"
    )

    # Active-set QP solved through its KKT system. The reduced Hessian is block
    # diagonal with the same block per state, so a few hundred active
    # constraints leave a dense solve of under a thousand unknowns -- instant,
    # where a general-purpose constrained optimiser did not finish in ten
    # minutes. The active set MUST stay capped: the KKT system is dense and
    # cubic in its size, so admitting thousands of cuts is slower than the
    # solver it replaced. Constraints are only added, never dropped, so the
    # result is a
    # feasible point and its cost is an UPPER bound on the price of
    # admissibility. An upper bound is what the question needs: if it is small,
    # the free solution was merely an unlucky representative.
    full = np.zeros((count * rank, count * rank), dtype=np.float64)
    for index in range(count):
        full[index * rank : (index + 1) * rank, index * rank : (index + 1) * rank] = hessian
    right = linear.reshape(-1)

    def solve_with(matrix):
        if matrix.shape[0] == 0:
            return np.linalg.solve(full, right).reshape(count, rank)
        size = full.shape[0] + matrix.shape[0]
        system = np.zeros((size, size), dtype=np.float64)
        system[: full.shape[0], : full.shape[0]] = full
        system[: full.shape[0], full.shape[0] :] = matrix.T
        system[full.shape[0] :, : full.shape[0]] = matrix
        vector = np.concatenate([right, np.zeros(matrix.shape[0])])
        solution = np.linalg.lstsq(system, vector, rcond=None)[0]
        return solution[: full.shape[0]].reshape(count, rank)

    active: list[tuple[int, int]] = []
    trace = []
    for outer in range(arguments.outer):
        rows = projections(coefficients)
        for _ in range(arguments.cutting_rounds):
            power = dissipation(coefficients, rows)
            violated = np.argwhere(power < 0.0)
            if violated.size == 0:
                break
            severity = power[violated[:, 0], violated[:, 1]]
            order = np.argsort(severity)[: arguments.constraints_per_round]
            for step, point in violated[order]:
                pair = (int(step), int(point))
                if pair not in active:
                    active.append(pair)
            coefficients = solve_with(constraint_matrix(rows, active))
        power = dissipation(coefficients, projections(coefficients))
        errors = [
            float(
                np.linalg.norm(design @ coefficients[index] - targets[index]) / references[index]
            )
            for index in range(count)
        ]
        entry = {
            "outer": outer,
            "active_constraints": len(active),
            "mean_error": float(np.mean(errors)),
            "error_at_the_last_state": errors[-1],
            "negative_points": float(np.mean(power < 0.0)),
            "negative_share_of_power": float(
                np.abs(np.minimum(power, 0.0)).sum() / np.abs(power).sum()
            ),
        }
        trace.append(entry)
        print(
            f"outer {outer}: {len(active):6d} active | mean error {entry['mean_error']:.5f} | "
            f"negative points {entry['negative_points']:.4f} | "
            f"negative power {entry['negative_share_of_power']:.4f}"
        )

    gauge = np.linalg.inv(PLANE_STRESS_VON_MISES_METRIC)
    previous = np.zeros((points, 3))
    path = np.zeros(points)
    for index in range(count):
        step = (subspace @ coefficients[index]).reshape(-1, 3) - previous
        path += np.sqrt(np.maximum(np.einsum("pi,ij,pj->p", step, gauge, step), 0.0))
        previous = (subspace @ coefficients[index]).reshape(-1, 3)
    final = (subspace @ coefficients[-1]).reshape(-1, 3)
    net = np.sqrt(np.maximum(np.einsum("pi,ij,pj->p", final, gauge, final), 0.0))

    output = {
        "schema_version": 1,
        "origin_nodes": [x0, y0],
        "states": states,
        "rank": rank,
        "unconstrained_mean_error": float(np.mean(free_errors)),
        "trace": trace,
        "path_over_net_ratio": float(
            np.sqrt((path**2).mean()) / np.sqrt((net**2).mean())
        ),
        "path_rms": float(np.sqrt((path**2).mean())),
        "net_rms": float(np.sqrt((net**2).mean())),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", "utf-8")
    print(
        f"\npath / net ratio {output['path_over_net_ratio']:.2f} "
        f"(free solution was 4.48)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
