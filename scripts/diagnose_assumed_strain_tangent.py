"""Is the assumed-strain element tangent consistent once `C` moves with `u`?

The priority test of
`validation/cps4r_assumed_strain_campaign2_preregistration.md`.

CPS4R-AS needs 47 Newton iterations on the SRIX case against CPS4's 37, which is
what pulls the constitutive speed-up below its bound while the number of
constitutive calls per element stays exactly one. The suspicion is that the
matrix handed to Newton differentiates `f_stab(u, C(u))` holding `C` fixed,
dropping `(df_stab/dC)(dC/du)`.

The existing element tests compare the stabilisation tangent against finite
differences **for a given tangent**, so they cannot see that term by
construction. This script perturbs the displacement and **re-integrates the
constitutive law at every perturbation**, so `C` moves as it does in a real
Newton step, and compares the finite-difference derivative of the *complete*
element force -- physical plus stabilisation -- against the matrix assembled.

The physical part, the stabilisation and the sum are reported separately. If the
physical part is consistent and the stabilisation is not, the missing term is
identified rather than hypothesised.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from fem_inhouse.core.assumed_strain import batched_stabilisation, central_operators
from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch

ORIENTATION_BUNGE_DEG = (35.0, 20.0, 15.0)
#: One element of the campaign mesh.
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


def element_force(
    batch: Any,
    operators: Any,
    displacement: np.ndarray,
    *,
    relative_floor: float | None,
    projection: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complete element force at `displacement`, with the law re-integrated.

    The batch is reverted before returning, so the committed state is untouched
    and repeated calls are independent -- which is what makes a finite
    difference meaningful here.
    """

    centre = operators.strain_displacement_centre
    trial = batch.evaluate((centre @ displacement)[None, :], time_increment=1.0)
    stress = np.asarray(trial.stress_in_plane_mpa[0], dtype=float)
    tangent = np.asarray(trial.tangent_in_plane_mpa[0], dtype=float)
    batch.revert()
    physical = operators.area * (centre.T @ stress)
    stabilising, stiffness, _ = batched_stabilisation(
        operators,
        displacement[None, :],
        tangent[None, :, :],
        projection=projection,
        relative_floor=relative_floor,
    )
    return physical, stabilising[0], stiffness[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", default="asmd")
    parser.add_argument("--floor", type=float, default=1.0e-6)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--strain", type=float, default=0.012)
    parser.add_argument(
        "--output", type=Path, default=Path("validation/_generated/cps4r_as")
    )
    arguments = parser.parse_args()

    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        raise SystemExit("MFRONT_BEHAVIOUR_LIBRARY must be set")
    arguments.output.mkdir(parents=True, exist_ok=True)

    operators = central_operators(NODES)
    centre = operators.strain_displacement_centre
    batch = _batch(library)

    # Drive into established plasticity, with a non-affine component so the
    # hourglass amplitudes are not zero and the stabilisation is actually doing
    # something at the state where the derivative is taken.
    span = float(NODES[:, 0].max())
    axial = arguments.strain * span
    affine = np.array([0.0, 0.0, axial, 0.0, axial, -0.4 * axial, 0.0, -0.4 * axial])
    modal = np.zeros(8)
    scale = max(float(np.abs(operators.gamma).max()), 1e-30)
    modal[0::2] = operators.gamma * (0.15 * axial / scale)
    base = affine + modal
    # Commit all but the LAST step, so every evaluation at `base` below carries a
    # genuine non-zero strain increment. Committing at `base` would make the
    # trial increment exactly zero, and SRIX then takes its guarded elastic
    # branch: the "tangent at the converged state" would be the elastic one
    # while the finite difference measures the plastic response. Measured first
    # -- that mistake reported 95 percent of error on the physical part alone.
    for index in range(arguments.steps - 1):
        state = base * (index + 1) / arguments.steps
        batch.evaluate((centre @ state)[None, :], time_increment=1.0)
        batch.commit()

    physical, stabilising, stab_tangent = element_force(
        batch, operators, base,
        relative_floor=arguments.floor, projection=arguments.projection,
    )
    trial = batch.evaluate((centre @ base)[None, :], time_increment=1.0)
    algorithmic = np.asarray(trial.tangent_in_plane_mpa[0], dtype=float)
    batch.revert()
    assembled_physical = operators.area * (centre.T @ algorithmic @ centre)
    assembled_total = assembled_physical + stab_tangent

    report: dict[str, Any] = {
        "projection": arguments.projection,
        "relative_floor": arguments.floor,
        "hourglass_amplitude": float(operators.gamma @ base[0::2]),
        "stabilisation_share_of_force": float(
            np.linalg.norm(stabilising) / max(float(np.linalg.norm(physical)), 1e-30)
        ),
        "stabilisation_share_of_tangent": float(
            np.abs(stab_tangent).max() / np.abs(assembled_total).max()
        ),
        "finite_difference": {},
    }
    best = {"physical": np.inf, "stabilisation": np.inf, "total": np.inf}
    for step in (1e-8, 1e-9, 1e-10, 1e-11):
        numerical_physical = np.zeros((8, 8))
        numerical_stab = np.zeros((8, 8))
        # ONE-SIDED, deliberately. At a converged plastic state the algorithmic
        # tangent is one-sided: unloading follows the elastic branch and loading
        # the plastic one, and a central difference averages the two. Measured
        # here first -- a central difference reported 16 percent of error on the
        # PHYSICAL part alone, which is not credible for A Bc^T C Bc and was the
        # signal that the probe, not the element, was wrong.
        reference = element_force(
            batch, operators, base,
            relative_floor=arguments.floor, projection=arguments.projection,
        )
        for column in range(8):
            forward = base.copy()
            forward[column] += step
            plus = element_force(
                batch, operators, forward,
                relative_floor=arguments.floor, projection=arguments.projection,
            )
            numerical_physical[:, column] = (plus[0] - reference[0]) / step
            numerical_stab[:, column] = (plus[1] - reference[1]) / step
        entry = {
            "physical": float(
                np.abs(numerical_physical - assembled_physical).max()
                / np.abs(assembled_physical).max()
            ),
            "stabilisation": float(
                np.abs(numerical_stab - stab_tangent).max() / np.abs(stab_tangent).max()
            ),
            "total": float(
                np.abs(numerical_physical + numerical_stab - assembled_total).max()
                / np.abs(assembled_total).max()
            ),
        }
        report["finite_difference"][f"{step:g}"] = entry
        for key, value in entry.items():
            best[key] = min(best[key], value)
    report["best_relative_error"] = best

    destination = arguments.output / "assumed_strain_tangent_consistency.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"hourglass amplitude            {report['hourglass_amplitude']:.4e}")
    print(f"stabilisation / physical force {report['stabilisation_share_of_force']:.4f}")
    print(f"stabilisation / total tangent  {report['stabilisation_share_of_tangent']:.4f}")
    print("best relative error over the finite-difference plateau:")
    for key, value in best.items():
        print(f"  {key:<14} {value:.3e}")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
