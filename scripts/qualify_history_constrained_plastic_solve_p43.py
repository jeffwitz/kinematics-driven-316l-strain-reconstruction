#!/usr/bin/env python3
"""Impose irreversibility on the shared subspace, and measure what it costs.

A rank-16 subspace shared by the four states reproduces every strain increment
to `1e-4`, but the amplitude it needs jumps and then decreases, where increments
from a common reference under monotonic loading should grow. Either the plastic
increment genuinely saturates, or the subspace is fitting something that is not
plastic. Nothing in that solve could tell them apart, because nothing tied one
state to the next.

The discriminator is to impose the constraint and see what it costs. Plasticity
dissipates, so between consecutive states

```text
sigma_k : (eps_p(k) - eps_p(k-1)) >= 0
```

at every material point. With the stress frozen at the unconstrained solution
this is **linear** in the reduced coefficients, and one outer update checks that
freezing it changed nothing that matters.

It is imposed by penalty rather than as a hard constraint, on purpose: the
quantity worth reading is not a feasible point but the **trade-off** — how much
agreement has to be given up to buy irreversibility. A constraint that costs
nothing was already satisfied; one that destroys the fit says the field being
reconstructed is not a plastic history.
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

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
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument(
        "--penalties", nargs="+", type=float, default=[0.0, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e4]
    )
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
    targets, measured_strains = [], []
    for state in arguments.states:
        field = history[state] - reference
        measured = np.asarray(operator.kinematics.strain(field)).reshape(-1)
        elastic = np.asarray(operator.kinematics.strain(extension(field))).reshape(-1)
        measured_strains.append(measured)
        targets.append(measured - elastic)
    targets = np.asarray(targets)
    measured_strains = np.asarray(measured_strains)
    references = np.linalg.norm(targets, axis=1)

    block = np.asarray([transpose(row) for row in targets]).T
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
    states = len(arguments.states)

    # Frozen stress from the unconstrained solution, which makes the
    # dissipation constraint linear in the coefficients.
    unconstrained, *_ = np.linalg.lstsq(design, targets.T, rcond=None)

    def stresses(coefficients):
        result = []
        for index in range(states):
            plastic = subspace @ coefficients[:, index]
            elastic_strain = measured_strains[index] - plastic
            result.append(
                np.einsum(
                    "pi,pij->pj", elastic_strain.reshape(-1, 3), operator.elasticity
                ).reshape(-1)
            )
        return np.asarray(result)

    gauge = np.linalg.inv(PLANE_STRESS_VON_MISES_METRIC)

    def amplitudes(coefficients):
        values = []
        for index in range(states):
            plastic = (subspace @ coefficients[:, index]).reshape(pixels, pixels, 2, 3)
            equivalent = np.sqrt(
                np.maximum(np.einsum("xyci,ij,xycj->xyc", plastic, gauge, plastic), 0.0)
            ).mean(axis=2)
            values.append(float(np.sqrt((equivalent**2).mean())))
        return values

    frozen = stresses(unconstrained)
    # Rows of the dissipation constraint: sigma_k . Phi (a_k - a_{k-1}) >= 0.
    projected = np.asarray(
        [
            (frozen[index].reshape(-1, 3)[:, None, :] * subspace.reshape(-1, 3, rank).transpose(
                0, 2, 1
            )).sum(axis=2)
            for index in range(states)
        ]
    )

    def dissipation(coefficients):
        increments = np.diff(
            np.concatenate([np.zeros((rank, 1)), coefficients], axis=1), axis=1
        )
        return np.asarray(
            [projected[index] @ increments[:, index] for index in range(states)]
        )

    records = []
    for penalty in arguments.penalties:

        def objective(flat, penalty=penalty):
            coefficients = flat.reshape(rank, states)
            misfit = design @ coefficients - targets.T
            value = float((misfit**2).sum())
            gradient = 2.0 * design.T @ misfit
            if penalty > 0.0:
                power = dissipation(coefficients)
                violation = np.minimum(power, 0.0)
                value += penalty * float((violation**2).sum())
                # The increments are Delta_k = a_k - a_{k-1}, so the adjoint of
                # that difference is a reverse difference, q_j - q_{j+1}. Using a
                # reverse cumulative sum here -- the adjoint of a cumulative sum,
                # not of a difference -- left the gradient 98 % wrong and the
                # optimiser motionless at every penalty.
                per_step = 2.0 * penalty * np.asarray(
                    [projected[index].T @ violation[index] for index in range(states)]
                ).T
                shifted = np.concatenate([per_step[:, 1:], np.zeros((rank, 1))], axis=1)
                gradient = gradient + per_step - shifted
            return value, gradient.reshape(-1)

        start = unconstrained if not records else records[-1]["solution"]
        result = minimize(
            objective,
            np.asarray(start).reshape(-1),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 4000, "ftol": 1e-18, "gtol": 1e-14},
        )
        coefficients = result.x.reshape(rank, states)
        misfit = design @ coefficients - targets.T
        power = dissipation(coefficients)
        amplitude = amplitudes(coefficients)
        records.append(
            {
                "penalty": penalty,
                "raw_error_by_state": [
                    float(np.linalg.norm(misfit[:, index]) / references[index])
                    for index in range(states)
                ],
                "plastic_rms_by_state": amplitude,
                "amplitude_is_monotone": bool(all(a < b for a, b in pairwise(amplitude))),
                "negative_dissipation_fraction": float(np.mean(power < 0.0)),
                "negative_dissipation_share_of_power": float(
                    np.abs(np.minimum(power, 0.0)).sum() / np.abs(power).sum()
                ),
                "solution": coefficients,
            }
        )
        entry = records[-1]
        print(
            f"penalty {penalty:9.3g} | err "
            + " ".join(f"{value:6.4f}" for value in entry["raw_error_by_state"])
            + " | p RMS "
            + " ".join(f"{value:.2e}" for value in amplitude)
            + f" | monotone {entry['amplitude_is_monotone']}"
            + f" | negative power {entry['negative_dissipation_share_of_power']:.4f}"
        )

    arguments.output.mkdir(parents=True, exist_ok=True)
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    penalties = [max(entry["penalty"], 1e-3) for entry in records]
    left.plot(
        penalties,
        [float(np.mean(entry["raw_error_by_state"])) for entry in records],
        "o-",
        label="mean raw error",
    )
    left.plot(
        penalties,
        [entry["negative_dissipation_share_of_power"] for entry in records],
        "s--",
        label="share of negative dissipation",
    )
    left.set_xscale("log")
    left.set_xlabel("dissipation penalty")
    left.legend()
    left.grid(alpha=0.3)
    for entry in records:
        right.plot(
            arguments.states,
            entry["plastic_rms_by_state"],
            "o-",
            label=f"penalty {entry['penalty']:g}",
        )
    right.set_xlabel("state")
    right.set_ylabel("plastic RMS")
    right.legend(fontsize=7)
    right.grid(alpha=0.3)
    figure.suptitle("What irreversibility costs on the shared subspace", fontsize=11)
    figure.savefig(arguments.output / "history_constraint.png", dpi=130)
    plt.close(figure)

    (arguments.output / "report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "origin_nodes": [x0, y0],
                "states": arguments.states,
                "reference_state": arguments.reference_state,
                "rank": rank,
                "constraint": "sigma_k : (eps_p(k) - eps_p(k-1)) >= 0, stress frozen",
                "records": [
                    {key: value for key, value in entry.items() if key != "solution"}
                    for entry in records
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
