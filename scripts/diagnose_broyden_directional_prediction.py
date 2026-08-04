"""Does the Broyden correction learn the operator, or memorise the path?

Section 23 of the 2026-08-04 specification, written to explain a negative
campaign rather than to look for a positive one.

The campaign result is unambiguous: on the registered SRIX case the correction
satisfies its secant conditions almost exactly -- the mean defect falls from
about `0.2` to `1e-4` -- and Newton needs MORE iterations, monotonically in the
memory. Those two facts are only compatible if the correction is right along the
directions it was fitted on and wrong along the direction the next step takes.

This script measures exactly that, on one element, with the constitutive law
re-integrated at every evaluation so `C` moves as it does in a real iteration:

- **in-sample**: predict `dr` along a direction already in the memory;
- **out-of-sample**: predict `dr` along a fresh direction of the same magnitude.

Both are reported for the base reduced Jacobian `G_0 = L K_stab T^+` and for the
corrected `G_0 + dG`. A correction that had learned the missing
`(df_stab/dC)(dC/du)` term would improve both. A correction that had fitted the
path improves the first and degrades the second.

Usage:
    MFRONT_BEHAVIOUR_LIBRARY=$PWD/build/mfront/src/libBehaviour.so \\
    python scripts/diagnose_broyden_directional_prediction.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from fem_inhouse.core.assumed_strain import batched_stabilisation, central_operators
from fem_inhouse.core.hourglass_modal_coordinates import modal_coordinates
from fem_inhouse.core.limited_memory_broyden import BroydenMemory, build_correction
from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch

ORIENTATION_BUNGE_DEG = (35.0, 20.0, 15.0)
NODES = np.array([[0.0, 0.0], [0.00184, 0.0], [0.00184, 0.00184], [0.0, 0.00184]])


def _batch(library: str) -> Any:
    from fem_inhouse.core.crystal_orientation import rotation_from_euler_bunge_deg

    return create_plane_stress_material_batch(
        "mfront-3d-condensed-plane-stress",
        np.full((1, 1), 250.0),
        np.full((1, 1), 500.0),
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.3,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1000,
        first_positive_plastic_strain=1e-6,
        mfront_library=library,
        mfront_threads=1,
        mfront_behaviour_id="fcc_forest_rubin_srix",
        constitutive_options={
            "crystal_orientation": {
                "mode": "homogeneous",
                "matrix": rotation_from_euler_bunge_deg(*ORIENTATION_BUNGE_DEG).tolist(),
            }
        },
    )


def _stabilisation(
    batch: Any, operators: Any, displacement: np.ndarray, *, floor: float, projection: str
) -> tuple[np.ndarray, np.ndarray]:
    """Stabilising force and tangent at one state, law re-integrated then reverted.

    The revert is what makes repeated evaluations independent, so a secant pair
    built from two of them is the pair a Newton iteration would have produced.
    """

    centre = operators.strain_displacement_centre
    trial = batch.evaluate((centre @ displacement)[None, :], time_increment=1.0)
    tangent = np.asarray(trial.tangent_in_plane_mpa[0], dtype=float)
    batch.revert()
    force, stiffness, _ = batched_stabilisation(
        operators,
        displacement[None, :],
        tangent[None, :, :],
        projection=projection,
        relative_floor=floor,
    )
    return force[0], stiffness[0]


def _relative(truth: np.ndarray, prediction: np.ndarray) -> float:
    scale = float(np.linalg.norm(truth))
    return float(np.linalg.norm(truth - prediction)) / (scale if scale > 0.0 else 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", default="asmd")
    parser.add_argument("--floor", type=float, default=1.0e-6)
    parser.add_argument("--memory", type=int, default=5)
    parser.add_argument("--preload-steps", type=int, default=20)
    parser.add_argument("--strain", type=float, default=0.012)
    #: Contraction of the simulated Newton sequence. Iterates approach the
    #: solution, so the stored directions shrink and align -- which is precisely
    #: the regime in which a multisecant fit can become ill-posed.
    parser.add_argument("--contraction", type=float, default=0.3)
    parser.add_argument("--iterates", type=int, default=6)
    parser.add_argument("--probes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--output", type=Path, default=Path("validation/_generated/cps4r_as")
    )
    arguments = parser.parse_args()

    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        raise SystemExit("MFRONT_BEHAVIOUR_LIBRARY must be set")
    arguments.output.mkdir(parents=True, exist_ok=True)

    operators = central_operators(NODES)
    coordinates = modal_coordinates(operators)
    centre = operators.strain_displacement_centre
    batch = _batch(library)
    generator = np.random.default_rng(arguments.seed)

    span = float(NODES[:, 0].max())
    axial = arguments.strain * span
    affine = np.array([0.0, 0.0, axial, 0.0, axial, -0.4 * axial, 0.0, -0.4 * axial])
    modal = np.zeros(8)
    scale = max(float(np.abs(operators.gamma).max()), 1e-30)
    modal[0::2] = operators.gamma * (0.15 * axial / scale)
    target = affine + modal
    # All but the last step committed, so every evaluation below carries a
    # genuine non-zero strain increment. Committing at the target would make the
    # trial increment exactly zero and SRIX would take its elastic branch.
    for index in range(arguments.preload_steps - 1):
        state = target * (index + 1) / arguments.preload_steps
        batch.evaluate((centre @ state)[None, :], time_increment=1.0)
        batch.commit()

    # A contracting sequence of iterates around the target, in the manner of a
    # Newton increment: the offsets shrink geometrically and change direction.
    offset = 0.05 * float(np.linalg.norm(target))
    iterates: list[np.ndarray] = []
    for index in range(arguments.iterates):
        direction = generator.standard_normal(8)
        direction /= np.linalg.norm(direction)
        iterates.append(target + offset * arguments.contraction**index * direction)

    memory = BroydenMemory(memory=arguments.memory)
    states: list[np.ndarray] = []
    forces: list[np.ndarray] = []
    for displacement in iterates:
        force, _ = _stabilisation(
            batch, operators, displacement, floor=arguments.floor,
            projection=arguments.projection,
        )
        states.append(coordinates.reduced_state(displacement))
        forces.append(coordinates.modal_force(force))
    for index in range(1, len(states)):
        memory.add(
            states[index] - states[index - 1],
            forces[index] - forces[index - 1],
            scale=float(np.linalg.norm(states[index])),
        )

    last = iterates[-1]
    last_force, last_tangent = _stabilisation(
        batch, operators, last, floor=arguments.floor, projection=arguments.projection
    )
    base = coordinates.reduced_jacobian(last_tangent)
    result = build_correction(memory, base)
    corrected = base + result.correction

    report: dict[str, Any] = {
        "projection": arguments.projection,
        "relative_floor": arguments.floor,
        "memory": arguments.memory,
        "pairs_stored": memory.pair_count,
        "rank": result.rank,
        "secant_defect_before": result.secant_defect_before,
        "secant_defect_after": result.secant_defect_after,
        "correction_relative_norm": float(
            np.linalg.norm(result.correction) / max(np.linalg.norm(base), 1e-30)
        ),
        "in_sample": {},
        "out_of_sample": {},
    }

    # In sample: the directions the correction was fitted on.
    in_base: list[float] = []
    in_corrected: list[float] = []
    for step, increment in zip(memory.steps, memory.increments, strict=True):
        in_base.append(_relative(increment, base @ step))
        in_corrected.append(_relative(increment, corrected @ step))
    report["in_sample"] = {
        "directions": len(in_base),
        "base_mean": float(np.mean(in_base)),
        "corrected_mean": float(np.mean(in_corrected)),
        "base_max": float(np.max(in_base)),
        "corrected_max": float(np.max(in_corrected)),
    }

    # Out of sample: fresh directions of the magnitude the NEXT step would have.
    step_length = float(np.linalg.norm(iterates[-1] - iterates[-2]))
    out_base: list[float] = []
    out_corrected: list[float] = []
    improved = 0
    for _ in range(arguments.probes):
        direction = generator.standard_normal(8)
        direction *= step_length / np.linalg.norm(direction)
        probe_force, _ = _stabilisation(
            batch, operators, last + direction, floor=arguments.floor,
            projection=arguments.projection,
        )
        truth = coordinates.modal_force(probe_force) - coordinates.modal_force(last_force)
        reduced_step = coordinates.reduced_state(direction)
        base_error = _relative(truth, base @ reduced_step)
        corrected_error = _relative(truth, corrected @ reduced_step)
        out_base.append(base_error)
        out_corrected.append(corrected_error)
        improved += int(corrected_error < base_error)
    report["out_of_sample"] = {
        "directions": arguments.probes,
        "step_length": step_length,
        "base_mean": float(np.mean(out_base)),
        "corrected_mean": float(np.mean(out_corrected)),
        "base_median": float(np.median(out_base)),
        "corrected_median": float(np.median(out_corrected)),
        "base_max": float(np.max(out_base)),
        "corrected_max": float(np.max(out_corrected)),
        "fraction_improved": improved / arguments.probes,
    }

    destination = arguments.output / "broyden_directional_prediction.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"pairs stored {memory.pair_count}, rank {result.rank}")
    print(
        f"secant defect      {result.secant_defect_before:.3e} -> "
        f"{result.secant_defect_after:.3e}"
    )
    print(
        f"in sample  mean    {report['in_sample']['base_mean']:.3e} -> "
        f"{report['in_sample']['corrected_mean']:.3e}"
    )
    print(
        f"out of sample mean {report['out_of_sample']['base_mean']:.3e} -> "
        f"{report['out_of_sample']['corrected_mean']:.3e}"
    )
    print(
        f"improved on {report['out_of_sample']['fraction_improved'] * 100:.0f} percent "
        "of fresh directions"
    )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
