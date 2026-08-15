#!/usr/bin/env python3
"""Does the shared subspace predict a state it never saw, and does it dissipate?

Two questions the previous run could not answer, one because the basis had seen
everything it was asked to reproduce, the other because the stress was wrong.

**Prediction.** The rank-16 subspace was built from the residuals of all four
states, so reproducing them is compression, not prediction. Building it from
three and asking it for the fourth is the test that separates the two.

**Dissipation.** The earlier check computed `sigma` from the strain *increment*
since state 20, which makes it the stress increment `d_sigma`, not the stress.
State 20 carries load, so the absolute term dominates and its omission can flip
the sign of `sigma : d_eps_p` outright. The "54.2 % of points thermodynamically
inadmissible" reported from it is withdrawn. Here

```text
sigma_n = sigma_20 + C : (d_eps_sim - d_eps_p),      sigma_20 = C : eps_20,
```

with `sigma_20` taken from the absolute measured strain at state 20 treated as
elastic, the increment taken from the **simulation** rather than the DIC -- they
coincide only while the fit is exact -- and the dissipation evaluated with the
mid-point rule over every available state rather than in four long jumps.
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
    parser.add_argument("--states", nargs="+", type=int, default=[25, 30, 35, 40])
    parser.add_argument("--dense-states", nargs="+", type=int, default=list(range(21, 41)))
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16, 32])
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

    def krylov(residuals, size):
        block = np.asarray([transpose(row) for row in residuals]).T
        basis, _ = np.linalg.qr(block)
        columns, total = [basis], basis.shape[1]
        while total < size:
            current = np.asarray(
                [transpose(response(columns[-1][:, k])) for k in range(columns[-1].shape[1])]
            ).T
            orthonormal, _ = np.linalg.qr(np.concatenate([*columns, current], axis=1))
            addition = orthonormal[:, total:]
            if addition.shape[1] == 0:
                break
            columns.append(addition)
            total += addition.shape[1]
        return np.concatenate(columns, axis=1)[:, :size]

    # Leave one state out: a basis that has never seen the state it is asked for.
    maximum = max(arguments.ranks)
    prediction = []
    for held_out in arguments.states:
        training = [state for state in arguments.states if state != held_out]
        subspace = krylov([target_of(state)[0] for state in training], maximum)
        responses = np.asarray(
            [response(subspace[:, k]) for k in range(subspace.shape[1])]
        ).T
        target, _ = target_of(held_out)
        entry = {"held_out": held_out, "errors": {}}
        for rank in arguments.ranks:
            coefficients, *_ = np.linalg.lstsq(responses[:, :rank], target, rcond=None)
            entry["errors"][str(rank)] = float(
                np.linalg.norm(responses[:, :rank] @ coefficients - target)
                / np.linalg.norm(target)
            )
        prediction.append(entry)
        print(
            f"held out {held_out}: "
            + "  ".join(
                f"rank {rank} -> {entry['errors'][str(rank)]:.4f}" for rank in arguments.ranks
            )
        )

    # Dissipation, with the absolute stress and the mid-point rule, on every state.
    dense = arguments.dense_states
    subspace = krylov([target_of(state)[0] for state in arguments.states], maximum)
    responses = np.asarray([response(subspace[:, k]) for k in range(subspace.shape[1])]).T
    rank = 16 if 16 in arguments.ranks else arguments.ranks[-1]
    plastic_fields, simulated_increments = [], []
    for state in dense:
        target, elastic = target_of(state)
        coefficients, *_ = np.linalg.lstsq(responses[:, :rank], target, rcond=None)
        plastic_fields.append(subspace[:, :rank] @ coefficients)
        simulated_increments.append(elastic + responses[:, :rank] @ coefficients)
    plastic_fields = np.asarray(plastic_fields)
    simulated_increments = np.asarray(simulated_increments)

    absolute_20 = np.asarray(operator.kinematics.strain(reference)).reshape(-1, 3)
    stress_20 = np.einsum("pi,pij->pj", absolute_20, operator.elasticity)
    stresses = np.asarray(
        [
            stress_20
            + np.einsum(
                "pi,pij->pj",
                (simulated_increments[index] - plastic_fields[index]).reshape(-1, 3),
                operator.elasticity,
            )
            for index in range(len(dense))
        ]
    )
    previous_plastic = np.zeros_like(plastic_fields[0]).reshape(-1, 3)
    previous_stress = stress_20
    powers = []
    for index in range(len(dense)):
        increment = plastic_fields[index].reshape(-1, 3) - previous_plastic
        mid = 0.5 * (previous_stress + stresses[index])
        powers.append((mid * increment).sum(axis=1))
        previous_plastic = plastic_fields[index].reshape(-1, 3)
        previous_stress = stresses[index]
    powers = np.asarray(powers)

    naive = []
    previous_plastic = np.zeros_like(plastic_fields[0]).reshape(-1, 3)
    for index in range(len(dense)):
        increment = plastic_fields[index].reshape(-1, 3) - previous_plastic
        incremental_stress = np.einsum(
            "pi,pij->pj",
            (simulated_increments[index] - plastic_fields[index]).reshape(-1, 3),
            operator.elasticity,
        )
        naive.append((incremental_stress * increment).sum(axis=1))
        previous_plastic = plastic_fields[index].reshape(-1, 3)
    naive = np.asarray(naive)

    gauge = np.linalg.inv(PLANE_STRESS_VON_MISES_METRIC)
    path = np.zeros(plastic_fields.shape[1] // 3)
    previous = np.zeros((path.size, 3))
    for index in range(len(dense)):
        step = plastic_fields[index].reshape(-1, 3) - previous
        path += np.sqrt(np.maximum(np.einsum("pi,ij,pj->p", step, gauge, step), 0.0))
        previous = plastic_fields[index].reshape(-1, 3)

    output = {
        "schema_version": 1,
        "origin_nodes": [x0, y0],
        "states": arguments.states,
        "dense_states": dense,
        "rank_used_for_dissipation": rank,
        "leave_one_state_out": prediction,
        "dissipation": {
            "definition": (
                "mid-point rule with the absolute stress, "
                "sigma_20 + C:(d_eps_sim - d_eps_p)"
            ),
            "negative_fraction": float(np.mean(powers < 0.0)),
            "negative_share_of_absolute_power": float(
                np.abs(np.minimum(powers, 0.0)).sum() / np.abs(powers).sum()
            ),
        },
        "dissipation_without_the_state_20_stress": {
            "definition": "the earlier, incorrect test: d_sigma : d_eps_p",
            "negative_fraction": float(np.mean(naive < 0.0)),
        },
        "accumulated_plastic_path_rms": float(np.sqrt((path**2).mean())),
        "accumulated_plastic_path_peak": float(path.max()),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", "utf-8")

    print(
        f"\ndissipation, absolute stress and mid-point rule over {len(dense)} states:"
        f"\n  negative points        : {output['dissipation']['negative_fraction']:.4f}"
        f"\n  negative share of power: "
        f"{output['dissipation']['negative_share_of_absolute_power']:.4f}"
        f"\n  the earlier d_sigma test: "
        f"{output['dissipation_without_the_state_20_stress']['negative_fraction']:.4f}"
    )
    print(
        f"accumulated plastic path: RMS {output['accumulated_plastic_path_rms']:.3e}, "
        f"peak {output['accumulated_plastic_path_peak']:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
