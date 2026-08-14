#!/usr/bin/env python3
"""Measure the archived oracle Delta-p history against the admissible wall.

Associated plane-stress J2 driven by a prescribed ``Delta p`` has a finite
admissible range: the flow relaxes the deviatoric stress towards the origin and
reaches it at ``Delta p_max``, beyond which no state with ``q > 0`` exists.
:meth:`DrivenJ2PlaneStressBatch.maximum_admissible_equivalent_plastic_increment`
gives that bound in closed form.

This walks the archived oracle history and reports, per state, how close the
requested increments sit to their own bound. It answers a question the
directional replay could only meet as a failure: is the prescribed history
admissible at all, and if not, by how much does it overshoot?

The replay is not re-run here and no equilibrium is solved. The trial state
used for the bound is the one the material carries after committing the
previous increment along the archived displacement history, which is what the
replay itself feeds the material.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fem_inhouse.core.driven_j2 import DrivenJ2PlaneStressBatch
from fem_inhouse.core.plane_stress_material import ConstitutiveIntegrationError
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIELDS = ROOT / "validation/_generated/performance/experimental_oracle_p43_m20/fields.npz"
PIXEL_SIZE_MM = 0.00184


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, default=DEFAULT_FIELDS)
    parser.add_argument("--increments", default="oracle", choices=("oracle", "ludwik"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    fields = np.load(arguments.fields, allow_pickle=False)
    displacement_history = np.asarray(fields["oracle_displacement_history"])
    increment_history = np.asarray(fields[f"{arguments.increments}_increment_history"])

    nx, ny = displacement_history.shape[1] - 1, displacement_history.shape[2] - 1
    grid = StructuredGrid2D(nx, ny, PIXEL_SIZE_MM * nx, PIXEL_SIZE_MM * ny)
    kinematics = TwoSubcellDiagnostic2D(grid)
    material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )

    records: list[dict[str, float | int]] = []
    for step in range(increment_history.shape[0]):
        strain = kinematics.strain(displacement_history[step + 1]).reshape(-1, 3)
        increment = increment_history[step].reshape(-1)
        bound = material.maximum_admissible_equivalent_plastic_increment(strain)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(bound > 0.0, increment / bound, np.inf)
        plastic = increment > 0.0
        records.append(
            {
                "state": step + 1,
                "plastic_points": int(np.count_nonzero(plastic)),
                "points_at_or_beyond_the_wall": int(np.count_nonzero(ratio >= 1.0)),
                "maximum_ratio": float(np.max(ratio[plastic])) if np.any(plastic) else 0.0,
                "median_ratio": float(np.median(ratio[plastic])) if np.any(plastic) else 0.0,
            }
        )
        try:
            material.evaluate(strain, np.minimum(increment, 0.999 * bound), time_increment=1.0)
            material.commit()
        # The walk is a diagnostic, not a solve: a point that cannot be
        # integrated stops the walk and is reported, rather than aborting it.
        except ConstitutiveIntegrationError:
            material.revert()
            records[-1]["walk_stopped_here"] = 1
            break

    report = {
        "schema_version": 1,
        "fields": str(arguments.fields),
        "increments": arguments.increments,
        "note": (
            "ratio is the requested Delta p divided by its closed-form admissible "
            "bound at the same trial state; >= 1 has no solution with q > 0"
        ),
        "states": records,
        "states_with_a_point_at_or_beyond_the_wall": [
            int(record["state"]) for record in records if record["points_at_or_beyond_the_wall"]
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
