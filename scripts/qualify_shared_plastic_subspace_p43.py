#!/usr/bin/env python3
"""One plastic subspace for all four states, instead of four independent fits.

Solving each state on its own is unconstrained enough that an exact fit is
guaranteed in advance, so the exactness carries no information. Requiring the
four states to build their plasticity inside the **same spatial subspace** is
the first real constraint that is not a constitutive law, not J2 and not
Tikhonov: it says the material has one place where it yields, and the load only
changes how much.

The subspace is a block Krylov space driven by the measured residuals of all
four states at once,

```text
Z_0 = A^T [r_25, r_30, r_35, r_40],   Z_{k+1} = A^T A Z_k,
```

kept in Krylov order. It is deliberately **not** reordered by the singular gain
of `A`: doing that reintroduces the observability ranking the residual-driven
construction exists to avoid, and it is what made an earlier attempt read as a
failure.

Then `eps_p(n) = Phi_r a_n` with one coefficient vector per state, and two
things are read off. Whether a shared subspace still closes the gap, compared
with four independent ones at the same parameter count. And whether the
coefficients grow with the load, which four independent fits cannot be asked --
nothing here imposes irreversibility or continuity, so a monotone trajectory is
a result.
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

from fem_inhouse.core.constitutive import PLANE_STRESS_VON_MISES_METRIC
from fem_inhouse.identification.pointwise_whitening import PointwiseFieldWhitener
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.measurement import image_flow_to_canonical
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior

ROOT = Path(__file__).resolve().parents[1]
NOISE = (
    ROOT
    / "validation/reference_data/dic_uncertainty_propagation_p0043_v1"
    / "centred_repeat_flow_pixels.npy"
)
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
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16, 32, 64, 128])
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
    shape = (pixels, pixels, 6)

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

    noise_source = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    canonical = np.asarray(
        image_flow_to_canonical(
            np.asarray(noise_source[:2100, :2100]), pixel_size_mm=PIXEL_SIZE_MM
        )
    )
    step = pixels + 1
    samples = []
    for row in range(0, canonical.shape[0] - step, step):
        for column in range(0, canonical.shape[1] - step, step):
            patch = np.ascontiguousarray(canonical[row : row + step, column : column + step, :])
            samples.append(
                np.asarray(operator.kinematics.strain(patch - extension(patch))).reshape(shape)
            )
    whitener = PointwiseFieldWhitener.fit(np.asarray(samples))

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
    targets = []
    for state in arguments.states:
        field = history[state] - reference
        measured = np.asarray(operator.kinematics.strain(field)).reshape(-1)
        elastic = np.asarray(operator.kinematics.strain(extension(field))).reshape(-1)
        targets.append(measured - elastic)
    targets = np.asarray(targets)
    references = np.linalg.norm(targets, axis=1)

    # Block Krylov, in Krylov order, no reordering by gain.
    maximum = max(arguments.ranks)
    block = np.asarray([transpose(row) for row in targets]).T
    basis, _ = np.linalg.qr(block)
    columns = [basis]
    total = basis.shape[1]
    while total < maximum:
        current = np.asarray(
            [transpose(response(columns[-1][:, k])) for k in range(columns[-1].shape[1])]
        ).T
        stacked = np.concatenate([*columns, current], axis=1)
        orthonormal, _ = np.linalg.qr(stacked)
        addition = orthonormal[:, total:]
        if addition.shape[1] == 0:
            break
        columns.append(addition)
        total += addition.shape[1]
    subspace = np.concatenate(columns, axis=1)[:, :maximum]
    gauge = np.linalg.inv(PLANE_STRESS_VON_MISES_METRIC)

    responses = np.asarray([response(subspace[:, k]) for k in range(subspace.shape[1])]).T
    records = []
    trajectories: dict[str, list[list[float]]] = {}
    for rank in arguments.ranks:
        if rank > subspace.shape[1]:
            continue
        design = responses[:, :rank]
        coefficients, *_ = np.linalg.lstsq(design, targets.T, rcond=None)
        errors, amplitudes, peaks, whitened = [], [], [], []
        for index in range(len(arguments.states)):
            difference = design @ coefficients[:, index] - targets[index]
            errors.append(float(np.linalg.norm(difference) / references[index]))
            whitened.append(
                float(
                    np.linalg.norm(whitener.apply(difference.reshape(shape)))
                    / np.linalg.norm(whitener.apply(targets[index].reshape(shape)))
                )
            )
            plastic = (subspace[:, :rank] @ coefficients[:, index]).reshape(
                pixels, pixels, 2, 3
            )
            equivalent = np.sqrt(
                np.maximum(np.einsum("xyci,ij,xycj->xyc", plastic, gauge, plastic), 0.0)
            ).mean(axis=2)
            amplitudes.append(float(np.sqrt((equivalent**2).mean())))
            peaks.append(float(equivalent.max()))
        trajectories[str(rank)] = coefficients[: min(rank, 6)].tolist()
        records.append(
            {
                "rank": rank,
                "raw_error_by_state": errors,
                "whitened_error_by_state": whitened,
                "plastic_rms_by_state": amplitudes,
                "plastic_peak_by_state": peaks,
                "amplitude_is_monotone": bool(
                    all(a < b for a, b in pairwise(amplitudes))
                ),
            }
        )
        print(
            f"rank {rank:4d} | raw err "
            + " ".join(f"{value:6.4f}" for value in errors)
            + " | p RMS "
            + " ".join(f"{value:.2e}" for value in amplitudes)
            + f" | monotone {records[-1]['amplitude_is_monotone']}"
        )

    arguments.output.mkdir(parents=True, exist_ok=True)
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    for index, state in enumerate(arguments.states):
        left.plot(
            [entry["rank"] for entry in records],
            [entry["raw_error_by_state"][index] for entry in records],
            "o-",
            label=f"state {state}",
        )
    left.set_xscale("log")
    left.set_xlabel("shared subspace rank")
    left.set_ylabel("raw tensor strain error")
    left.legend()
    left.grid(alpha=0.3)
    widest = str(max(int(key) for key in trajectories))
    for mode, series in enumerate(trajectories[widest][:4]):
        right.plot(arguments.states, series, "o-", label=f"mode {mode + 1}")
    right.set_xlabel("state")
    right.set_ylabel(f"coefficient, rank {widest} subspace")
    right.legend()
    right.grid(alpha=0.3)
    figure.suptitle(
        "One plastic subspace shared by four states: closure and coefficient history",
        fontsize=11,
    )
    figure.savefig(arguments.output / "shared_subspace.png", dpi=130)
    plt.close(figure)

    (arguments.output / "report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "origin_nodes": [x0, y0],
                "states": arguments.states,
                "reference_state": arguments.reference_state,
                "subspace_size": int(subspace.shape[1]),
                "records": records,
                "coefficient_trajectories": trajectories,
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
