#!/usr/bin/env python3
"""The price of a positively dissipative plastic history.

The free inversion reproduces every measured increment and wanders: it
dissipates 45/55 and its path is four to five times longer than its net
displacement. That says the representative picked by the kinematic criterion
alone is not a plastic history. It does not say no dissipative representative
exists, because the inverse is massively non-unique. This measures the
difference.

```text
min || A Phi a - r ||^2   subject to   sigma_k . (eps_p(k) - eps_p(k-1)) >= 0
```

at every material point and step. In Kelvin the contraction is a plain dot
product, so the constraint row is the projection of the mid-point stress on the
basis, with no metric and no factor of two.

Three things the two earlier attempts got wrong, each fixed here.

A squared penalty on `min(D, 0)` has a trivial minimiser at zero: it shrinks the
field instead of reorienting it. This is a hard constraint.

An add-only active set over-determines the system once the cuts outnumber the
unknowns, forcing the same collapse by another route. OSQP handles inequalities
as inequalities.

And a cut is built from a frozen `sigma`, so it is **not** a valid constraint of
the next linearisation. The constraint set is rebuilt from scratch at every
outer iteration rather than accumulated.

The start is `a = 0`, which is feasible by construction -- no increments, no
dissipation -- so the trajectory grows into the admissible set rather than being
dragged back into it.

What this shows, if the price is small, is the existence of a locally
dissipative eigenstrain history compatible with the DIC and with equilibrium.
Not plasticity: no yield surface, no normality, no consistency is imposed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import osqp
import scipy.sparse as sparse

from fem_inhouse.core.kelvin import equivalent_plastic_strain
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D

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
    parser.add_argument("--outer", type=int, default=4)
    parser.add_argument("--cuts", type=int, default=6)
    parser.add_argument("--cuts-per-round", type=int, default=3000)
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
        strain = operator.kelvin_strain(field)
        stress = np.einsum("pi,pij->pj", strain, operator.elasticity)
        forcing = operator._strain_transpose(stress.reshape(-1))
        result = field.copy()
        result[1:-1, 1:-1, :] -= operator.solve_stiffness(forcing).reshape(
            pixels - 1, pixels - 1, 2
        )
        return result

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

    def forward(coordinates_vector):
        """Gauge coordinates to Kelvin strain."""

        plastic = (
            np.asarray(coordinates_vector, dtype=np.float64).reshape(-1, 3)
            @ operator.inverse_gauge_root
        )
        return operator.kelvin_response(plastic.reshape(-1))

    def adjoint(strain_vector):
        """The transpose of `forward`, in strain space rather than observation space."""

        forcing = operator._strain_transpose(
            np.einsum(
                "pi,pij->pj",
                np.asarray(strain_vector, dtype=np.float64).reshape(-1, 3),
                operator.elasticity,
            ).reshape(-1)
        )
        from fem_inhouse.spectral2d.newton_ebi import unpack_interior

        displacement = unpack_interior(operator.solve_stiffness(forcing), grid)
        strain = operator.kelvin_strain(displacement)
        stress = np.einsum("pi,pij->pj", strain, operator.elasticity)
        return (stress @ operator.inverse_gauge_root).reshape(-1)

    def target_of(state):
        field = history[state] - reference
        measured = operator.kelvin_strain(field).reshape(-1)
        elastic = operator.kelvin_strain(extension(field)).reshape(-1)
        return measured - elastic, elastic

    block = np.asarray(
        [adjoint(target_of(state)[0]) for state in arguments.basis_states]
    ).T
    basis, _ = np.linalg.qr(block)
    columns, total = [basis], basis.shape[1]
    while total < arguments.rank:
        current = np.asarray(
            [
                adjoint(forward(columns[-1][:, k]))
                for k in range(columns[-1].shape[1])
            ]
        ).T
        orthonormal, _ = np.linalg.qr(np.concatenate([*columns, current], axis=1))
        addition = orthonormal[:, total:]
        if addition.shape[1] == 0:
            break
        columns.append(addition)
        total += addition.shape[1]
    coordinates = np.concatenate(columns, axis=1)[:, : arguments.rank]
    rank = coordinates.shape[1]
    # The basis in plastic units, and the strain each mode produces.
    subspace = np.asarray(
        [
            (coordinates[:, k].reshape(-1, 3) @ operator.inverse_gauge_root).reshape(-1)
            for k in range(rank)
        ]
    ).T
    design = np.asarray([forward(coordinates[:, k]) for k in range(rank)]).T

    states = arguments.states
    count = len(states)
    targets, elastics = [], []
    for state in states:
        target, elastic = target_of(state)
        targets.append(target)
        elastics.append(elastic)
    targets = np.asarray(targets)
    elastics = np.asarray(elastics)
    references = np.linalg.norm(targets, axis=1)

    hessian = design.T @ design
    linear = targets @ design
    quadratic = sparse.block_diag([2.0 * hessian] * count, format="csc")
    gradient = (-2.0 * linear).reshape(-1)

    absolute = operator.kelvin_strain(reference)
    stress_20 = np.einsum("pi,pij->pj", absolute, operator.elasticity)
    modes = subspace.reshape(points, 3, rank)

    def rows_of(coefficients):
        """Mid-point stress projected on the basis: the constraint's linear form."""

        previous = stress_20
        rows = np.empty((count, points, rank), dtype=np.float64)
        for index in range(count):
            plastic = subspace @ coefficients[index]
            simulated = elastics[index] + design @ coefficients[index]
            stress = stress_20 + np.einsum(
                "pi,pij->pj",
                (simulated - plastic).reshape(-1, 3),
                operator.elasticity,
            )
            rows[index] = np.einsum("pi,pir->pr", 0.5 * (previous + stress), modes)
            previous = stress
        return rows

    def dissipation(coefficients, rows):
        increments = np.diff(
            np.concatenate([np.zeros((1, rank)), coefficients], axis=0), axis=0
        )
        return np.einsum("npr,nr->np", rows, increments)

    def errors_of(coefficients):
        return [
            float(
                np.linalg.norm(design @ coefficients[index] - targets[index])
                / references[index]
            )
            for index in range(count)
        ]

    def solve_with(rows, selected):
        data, row_index, column_index = [], [], []
        for number, (step, point) in enumerate(selected):
            block_row = rows[step, point]
            for component in range(rank):
                data.append(block_row[component])
                row_index.append(number)
                column_index.append(step * rank + component)
                if step > 0:
                    data.append(-block_row[component])
                    row_index.append(number)
                    column_index.append((step - 1) * rank + component)
        constraint = sparse.csc_matrix(
            (data, (row_index, column_index)), shape=(len(selected), count * rank)
        )
        problem = osqp.OSQP()
        problem.setup(
            P=quadratic,
            q=gradient,
            A=constraint,
            l=np.zeros(len(selected)),
            u=np.full(len(selected), np.inf),
            verbose=False,
            eps_abs=1e-9,
            eps_rel=1e-9,
            max_iter=40_000,
            polish=True,
        )
        result = problem.solve()
        return np.asarray(result.x, dtype=np.float64).reshape(count, rank), result.info.status

    free = np.linalg.solve(hessian, linear.T).T
    free_errors = errors_of(free)
    free_power = dissipation(free, rows_of(free))
    print(
        f"free      : mean error {np.mean(free_errors):.5f} | "
        f"negative points {np.mean(free_power < 0.0):.4f} | "
        f"negative power {np.abs(np.minimum(free_power, 0.0)).sum() / np.abs(free_power).sum():.4f}"
    )

    coefficients = np.zeros((count, rank), dtype=np.float64)
    trace = []
    for outer in range(arguments.outer):
        rows = rows_of(coefficients)
        # Rebuilt, never accumulated: a cut is only valid for the sigma it was
        # linearised at.
        selected: list[tuple[int, int]] = []
        status = "unconstrained"
        # Solve BEFORE cutting. Starting from a = 0 is feasible, which makes it a
        # fixed point of a cut-then-solve loop: no violation to cut on, so the QP
        # never runs and the trajectory never leaves zero. The first solve of
        # each outer iteration therefore has an empty constraint set, and the
        # violations of its answer are what the cuts are built from.
        coefficients = np.linalg.solve(hessian, linear.T).T
        for _ in range(arguments.cuts):
            power = dissipation(coefficients, rows)
            violated = np.argwhere(power < 0.0)
            if violated.size == 0:
                break
            severity = power[violated[:, 0], violated[:, 1]]
            order = np.argsort(severity)[: arguments.cuts_per_round]
            known = set(selected)
            for step, point in violated[order]:
                pair = (int(step), int(point))
                if pair not in known:
                    selected.append(pair)
                    known.add(pair)
            coefficients, status = solve_with(rows, selected)
        true_power = dissipation(coefficients, rows_of(coefficients))
        errors = errors_of(coefficients)
        entry = {
            "outer": outer,
            "cuts": len(selected),
            "status": str(status),
            "mean_error": float(np.mean(errors)),
            "error_by_state": errors,
            "negative_points": float(np.mean(true_power < 0.0)),
            "negative_share_of_power": float(
                np.abs(np.minimum(true_power, 0.0)).sum() / np.abs(true_power).sum()
            ),
        }
        trace.append(entry)
        print(
            f"outer {outer} : {len(selected):6d} cuts | mean error {entry['mean_error']:.5f} | "
            f"negative points {entry['negative_points']:.4f} | "
            f"negative power {entry['negative_share_of_power']:.4f} | {status}"
        )

    previous = np.zeros((points, 3))
    path = np.zeros(points)
    for index in range(count):
        current = (subspace @ coefficients[index]).reshape(-1, 3)
        path += equivalent_plastic_strain(current - previous)
        previous = current
    net = equivalent_plastic_strain(previous)

    output = {
        "schema_version": 1,
        "origin_nodes": [x0, y0],
        "states": states,
        "rank": rank,
        "representation": "Kelvin; the dissipation is a plain dot product",
        "free_mean_error": float(np.mean(free_errors)),
        "free_negative_points": float(np.mean(free_power < 0.0)),
        "trace": trace,
        "price_of_admissibility": trace[-1]["mean_error"] if trace else None,
        "plastic_rms": float(np.sqrt((net**2).mean())),
        "path_over_net": float(np.sqrt((path**2).mean()) / np.sqrt((net**2).mean())),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", "utf-8")
    print(
        f"\nprice of admissibility {output['price_of_admissibility']:.5f} "
        f"against {output['free_mean_error']:.5f} free | "
        f"plastic RMS {output['plastic_rms']:.3e} | "
        f"path/net {output['path_over_net']:.2f} (free was 4.48)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
