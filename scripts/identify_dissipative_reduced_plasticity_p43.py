#!/usr/bin/env python3
"""A plastic history from equilibrium and the DIC alone -- and does it obey a law?

The Ludwik replay is kept as an archived negative control: a per-pixel yield map
localises the plastic field into places the measurement does not, and the
correction it applies is orthogonal to the defect it should close. So nothing
here uses a yield map, a hardening law, a flow direction or a threshold. The
only inputs are the measured kinematics, linear elasticity and equilibrium.

```text
r_n = eps_DIC(n) - eps_el(n)        the defect, with eps_el the elastic field
                                    carrying the same Dirichlet boundary data
eps_p(n) = Phi a_n                  a reduced plastic field
eps_sim(n) = eps_el(n) + A Phi a_n  equilibrium closes the loop
```

The basis is block-Krylov on `A^T r`, so its modes are the plastic directions
that can reach the measurement *through equilibrium* -- not directions a
property map suggests.

Thermodynamics needs no constitutive law. Once `a_n` is known,

```text
sigma_n = C : (eps_sim(n) - eps_p(n))
d_n(x)  = sigma_n(x) : (eps_p(n) - eps_p(n-1))(x)  >= 0
```

is the second principle at every integration point, and its reduced form is
`X_n . (a_n - a_{n-1})` with `X_n = Phi^T W C : (...)` the reduced driving
force. This pass measures how far the unconstrained trajectory is from that
requirement, and then asks the question that actually distinguishes a history
from a law:

**do the pairs `(delta a_n, X_n)` come from one convex potential?**

They do exactly when the sequence is cyclically monotone, and that has an exact
finite test -- no fitting, no tolerance. Build the complete digraph with edge
weights `w(i -> j) = X_i . (delta a_j - delta a_i)`; the subgradient
inequalities `rho_j >= rho_i + w(i -> j)` admit a solution if and only if the
graph carries no positive cycle. Bellman-Ford settles it on forty nodes.

## The noise floor is NOT established here, and the attempt is kept as a warning

Everything above divides by `|r|`, so how much of `r` is DIC noise decides
whether any of it means anything. The estimate implemented here fails, and
loudly enough to be worth keeping rather than deleting. Its amplitude comes
from the second time difference, `sigma ~ std(u[k+1] - 2 u[k] + u[k-1]) /
sqrt(6)`, which gives 0.045 pixel and is entirely plausible; but the synthetic
field it drives is spatially **white**, and pushing that through the same
elastic-defect chain produces a defect a hundred times larger than the measured
one. A hundredfold upper bound decides nothing.

The failure is instructive. Strain is a derivative, so its noise is dominated
by the highest spatial frequencies, and real correlation error is smooth over
the subset window rather than white. Any floor that ignores the spatial
covariance overestimates by orders of magnitude. The right instrument already
exists in `identification/dic_whitening.py`, which carries the measured
covariance, and the reference the earlier campaigns used is the propagated
`(I - E P_b) n` rather than the raw noise. Until that is wired here, every
ratio in this report has an unknown denominator, and the reported share is
published only to record that it is vacuous.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.core.kelvin import PLANE_STRESS_PLASTIC_GAUGE
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D

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


def _positive_cycle_exists(weights: FloatArray, tolerance: float) -> tuple[bool, float]:
    """Bellman-Ford on `-w`: a negative cycle there is a positive cycle here.

    Cyclic monotonicity of the pairs is exactly the absence of a positive
    cycle, so this is a decision rather than a fit. The returned slack is the
    largest violation found while relaxing, which says how far from a potential
    the trajectory sits when the answer is no.
    """

    count = weights.shape[0]
    potential = np.zeros(count)
    worst = 0.0
    for iteration in range(count):
        updated = False
        for i in range(count):
            candidate = potential[i] + weights[i]
            improvement = candidate - potential
            slack = float(improvement.max())
            if slack > tolerance:
                worst = max(worst, slack)
                potential = np.maximum(potential, candidate)
                updated = True
        if not updated:
            return False, worst
        if iteration == count - 1:
            return True, worst
    return False, worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", nargs=2, type=int, default=(1580, 1030))
    parser.add_argument("--pixels", type=int, default=100)
    parser.add_argument("--first-state", type=int, default=1)
    parser.add_argument("--last-state", type=int, default=40)
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--lsqr-iterations", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    pixels = arguments.pixels
    origin = (int(arguments.origin[0]), int(arguments.origin[1]))
    states = list(range(arguments.first_state, arguments.last_state + 1))
    grid = StructuredGrid2D(pixels, pixels, pixels * PIXEL_SIZE_MM, pixels * PIXEL_SIZE_MM)
    operator = TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=_Identity(),
        whitener=_Identity(),
    )
    points = operator.kinematics.material_point_count

    def elastic_extension(field: FloatArray) -> FloatArray:
        """The elastic field carrying the same Dirichlet data as `field`."""

        strain = operator.kelvin_strain(field)
        stress = np.einsum("pi,pij->pj", strain, operator.elasticity)
        interior = operator.solve_stiffness(operator._strain_transpose(stress.reshape(-1)))
        result = field.copy()
        result[1:-1, 1:-1, :] -= interior.reshape(pixels - 1, pixels - 1, 2)
        return result

    def transpose(observation: FloatArray) -> FloatArray:
        dual = np.asarray(observation, dtype=np.float64).reshape(-1, 3)
        displacement = operator.solve_stiffness(operator._strain_transpose(dual.reshape(-1)))
        from fem_inhouse.spectral2d.newton_ebi import unpack_interior

        strain = operator.kelvin_strain(unpack_interior(displacement, grid))
        return np.einsum("pi,pij->pj", strain, operator.elasticity).reshape(-1)

    # An affine field is already in elastic equilibrium, so the extension must
    # return it unchanged. This catches a sign error in the forcing, which is
    # otherwise invisible and poisons every residual below.
    x, y = grid.coordinates
    affine = np.stack(
        [1.0e-3 * x[:, None] + 0.0 * y[None, :], 0.0 * x[:, None] - 3.0e-4 * y[None, :]],
        axis=-1,
    )
    affine_error = float(np.linalg.norm(elastic_extension(affine) - affine))
    if affine_error > 1.0e-9 * float(np.linalg.norm(affine)):
        raise SystemExit(
            f"the elastic extension is not idempotent on affine fields: {affine_error}"
        )

    history = _load_history(origin, pixels)
    measured = {n: operator.kelvin_strain(history[n]).reshape(-1) for n in states}
    elastic = {
        n: operator.kelvin_strain(elastic_extension(history[n])).reshape(-1) for n in states
    }
    residual = {n: measured[n] - elastic[n] for n in states}

    # --- Noise floor -------------------------------------------------------
    second_difference = history[2:] - 2.0 * history[1:-1] + history[:-2]
    noise_mm = float(second_difference.std() / np.sqrt(6.0))
    generator = np.random.default_rng(20260815)
    synthetic = generator.normal(0.0, noise_mm, size=history[0].shape)
    synthetic[0, :, :] = synthetic[-1, :, :] = 0.0
    synthetic[:, 0, :] = synthetic[:, -1, :] = 0.0
    noise_defect = float(
        np.linalg.norm(
            operator.kelvin_strain(synthetic).reshape(-1)
            - operator.kelvin_strain(elastic_extension(synthetic)).reshape(-1)
        )
    )
    noise_share = {
        str(n): noise_defect / max(float(np.linalg.norm(residual[n])), 1.0e-30) for n in states
    }
    print(
        f"noise UPPER BOUND (white, therefore vacuous): sigma {noise_mm:.3e} mm "
        f"= {noise_mm / PIXEL_SIZE_MM:.3f} pixel, reaching |r| {noise_defect:.4e}, "
        f"i.e. {noise_share[str(states[-1])]:.1f} times the measured defect at "
        f"state {states[-1]}",
        flush=True,
    )

    # --- Block-Krylov basis from the data alone ----------------------------
    maximum_rank = max(arguments.ranks)
    seeds = np.asarray([transpose(residual[n]) for n in states]).T
    basis, _ = np.linalg.qr(seeds)
    columns, total = [basis], basis.shape[1]
    while total < maximum_rank:
        current = np.asarray(
            [
                transpose(operator.kelvin_response(columns[-1][:, k]))
                for k in range(columns[-1].shape[1])
            ]
        ).T
        orthonormal, _ = np.linalg.qr(np.concatenate([*columns, current], axis=1))
        addition = orthonormal[:, total:]
        if addition.shape[1] == 0:
            break
        columns.append(addition)
        total += addition.shape[1]
        print(f"  krylov rank {total}", flush=True)
    subspace = np.concatenate(columns, axis=1)[:, :maximum_rank]
    responses = np.asarray(
        [operator.kelvin_response(subspace[:, k]) for k in range(subspace.shape[1])]
    ).T

    # --- Joint trajectory, one solve per state at each rank ----------------
    per_rank: dict[str, Any] = {}
    trajectories: dict[int, FloatArray] = {}
    for rank in arguments.ranks:
        coefficients = np.asarray(
            [
                np.linalg.lstsq(responses[:, :rank], residual[n], rcond=None)[0]
                for n in states
            ]
        )
        errors = [
            float(
                np.linalg.norm(responses[:, :rank] @ coefficients[index] - residual[n])
                / np.linalg.norm(residual[n])
            )
            for index, n in enumerate(states)
        ]
        trajectories[rank] = coefficients
        per_rank[str(rank)] = {
            "mean_relative_error": float(np.mean(errors)),
            "final_relative_error": errors[-1],
            "relative_errors": errors,
        }
        print(f"rank {rank}: mean misfit {np.mean(errors):.4f}", flush=True)

    # --- Thermodynamics of the unconstrained trajectory --------------------
    rank = max(arguments.ranks)
    coefficients = trajectories[rank]
    plastic = coefficients @ subspace[:, :rank].T
    weight = operator.quadrature_weight
    previous_plastic = np.zeros((points, 3))
    powers, reduced_forces, reduced_increments = [], [], []
    for index, n in enumerate(states):
        field = plastic[index].reshape(-1, 3)
        simulated = elastic[n] + responses[:, :rank] @ coefficients[index]
        stress = np.einsum(
            "pi,pij->pj", simulated.reshape(-1, 3) - field, operator.elasticity
        )
        increment = field - previous_plastic
        powers.append((stress * increment).sum(axis=1))
        reduced_forces.append(weight * (subspace[:, :rank].T @ stress.reshape(-1)))
        reduced_increments.append(coefficients[index] - (coefficients[index - 1] if index else 0.0))
        previous_plastic = field
    powers = np.asarray(powers)
    reduced_forces = np.asarray(reduced_forces)
    reduced_increments = np.asarray(reduced_increments)
    reduced_dissipation = np.einsum("ni,ni->n", reduced_forces, reduced_increments)

    gauge = PLANE_STRESS_PLASTIC_GAUGE
    equivalent = np.sqrt(
        np.maximum(
            np.einsum("pi,ij,pj->p", plastic[-1].reshape(-1, 3), gauge, plastic[-1].reshape(-1, 3)),
            0.0,
        )
    )

    # --- Does a convex potential exist? ------------------------------------
    span = np.abs(reduced_forces).max() * np.abs(reduced_increments).max()
    weights_matrix = reduced_forces @ reduced_increments.T - np.einsum(
        "ni,ni->n", reduced_forces, reduced_increments
    )[:, None]
    positive_cycle, worst_slack = _positive_cycle_exists(
        weights_matrix, tolerance=1.0e-12 * max(span, 1.0)
    )

    report = {
        "schema_version": 1,
        "status": "completed_dissipative_reduced_plasticity_p43",
        "question": (
            "can equilibrium and the DIC alone carry a plastic history, and do "
            "its reduced force/increment pairs come from one convex potential"
        ),
        "origin_nodes": list(origin),
        "mesh": [pixels, pixels],
        "states": states,
        "constitutive_inputs": "none: no yield map, no hardening law, no flow rule",
        "affine_extension_error": affine_error,
        "noise_floor": {
            "status": (
                "NOT ESTABLISHED: the synthetic field is spatially white, which "
                "overestimates strain noise by orders of magnitude; wire the "
                "measured covariance from identification/dic_whitening.py and "
                "the propagated (I - E P_b) n reference instead"
            ),
            "method": "second temporal difference, white spatial synthetic",
            "sigma_mm": noise_mm,
            "defect_norm_reached": noise_defect,
            "share_of_the_defect": noise_share,
        },
        "subspace": {
            "seeds": len(states),
            "rank_built": int(subspace.shape[1]),
            "per_rank": per_rank,
        },
        "thermodynamics": {
            "rank": rank,
            "pointwise_negative_fraction": float(np.mean(powers < 0.0)),
            "negative_share_of_absolute_power": float(
                np.abs(np.minimum(powers, 0.0)).sum() / max(np.abs(powers).sum(), 1.0e-30)
            ),
            "reduced_dissipation_negative_states": [
                states[index]
                for index in range(len(states))
                if reduced_dissipation[index] < 0.0
            ],
            "equivalent_plastic_strain_rms": float(np.sqrt((equivalent**2).mean())),
            "equivalent_plastic_strain_peak": float(equivalent.max()),
        },
        "convex_potential": {
            "test": "cyclic monotonicity by positive-cycle detection",
            "potential_exists": not positive_cycle,
            "worst_violation": worst_slack,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    np.savez_compressed(
        arguments.output.with_suffix(".npz"),
        subspace=subspace,
        coefficients=coefficients,
        reduced_forces=reduced_forces,
        reduced_increments=reduced_increments,
        plastic_final=plastic[-1],
        pointwise_power=powers,
    )
    print(
        f"\nnegative dissipation points "
        f"{report['thermodynamics']['pointwise_negative_fraction']:.4f}"
        f"\nconvex potential exists: {report['convex_potential']['potential_exists']}"
        f"\nwrote {arguments.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
