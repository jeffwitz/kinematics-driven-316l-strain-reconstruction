#!/usr/bin/env python3
"""Replay the J2/Ludwik baseline and archive the displacement history it produces.

The oracle archive stores `ludwik_increment_history` but no matching
displacement history, only `oracle_displacement_history`. Any check that needs
the baseline trajectory -- for instance measuring its prescribed increments
against the admissible wall -- would otherwise have to pair Ludwik increments
with the oracle's displacements, which are the trajectory of a different
solution. That mismatch is silent and produces plausible-looking numbers.

This replays the baseline properly: at each state the prescribed Ludwik
increment is imposed on the driven-J2 material and equilibrium is solved under
the measured boundary displacement, exactly as the directional diagnostic does,
and the resulting field is archived alongside the wall ratio of that state.
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
from fem_inhouse.workflows.experimental_mechanical_oracle import (
    solve_fixed_plastic_increment_equilibrium,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIELDS = ROOT / "validation/_generated/performance/experimental_oracle_p43_m20/fields.npz"
PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, default=DEFAULT_FIELDS)
    parser.add_argument("--equilibrium-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--admissible-fraction", type=float, default=0.999)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    fields = np.load(arguments.fields, allow_pickle=False)
    measured = np.asarray(fields["measured_displacement_history"])
    ludwik = np.asarray(fields["ludwik_increment_history"])
    pixels = measured.shape[1] - 1
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    kinematics = TwoSubcellDiagnostic2D(grid)
    material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
    )

    displacement = measured[0].copy()
    history = [displacement.copy()]
    records: list[dict[str, float | int]] = []
    failure: dict[str, object] | None = None

    for step, increment in enumerate(ludwik):
        target = measured[step + 1]
        strain = np.asarray(kinematics.strain(displacement)).reshape(-1, 3)
        bound = material.maximum_admissible_equivalent_plastic_increment(strain)
        flat = np.asarray(increment, dtype=np.float64).reshape(-1)
        plastic = flat > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(bound > 0.0, flat / bound, np.inf)
        requested = np.asarray(increment, dtype=np.float64)
        clipped = np.minimum(
            requested, (arguments.admissible_fraction * bound).reshape(requested.shape)
        )
        try:
            result = solve_fixed_plastic_increment_equilibrium(
                material=material,
                kinematics=kinematics,
                boundary_displacement=target,
                equivalent_plastic_increment=clipped,
                initial_displacement=displacement,
                time_increment=1.0,
                equilibrium_rms_tolerance=arguments.equilibrium_tolerance,
            )
        except ConstitutiveIntegrationError as error:
            failure = {
                "state": step + 1,
                "message": str(error),
                "diagnostics": getattr(error, "diagnostics", None),
            }
            material.revert()
            break
        displacement = result.displacement.copy()
        material.commit()
        history.append(displacement.copy())
        records.append(
            {
                "state": step + 1,
                "equilibrium_rms": float(result.equilibrium_rms),
                "plastic_points": int(np.count_nonzero(plastic)),
                "points_at_or_beyond_the_wall": int(np.count_nonzero(ratio >= 1.0)),
                "maximum_ratio": float(np.max(ratio[plastic])) if np.any(plastic) else 0.0,
                "clipped_points": int(np.count_nonzero(clipped < requested - 0.0)),
            }
        )

    stacked = np.asarray(history)
    arguments.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output / "ludwik_baseline_history.npz",
        ludwik_displacement_history=stacked,
        ludwik_increment_history=ludwik[: len(records)],
    )
    beyond = [int(entry["state"]) for entry in records if entry["points_at_or_beyond_the_wall"]]
    report = {
        "schema_version": 1,
        "fields": str(arguments.fields),
        "admissible_fraction": arguments.admissible_fraction,
        "states_completed": len(records),
        "states_requested": int(ludwik.shape[0]),
        "states_with_a_point_at_or_beyond_the_wall": beyond,
        "maximum_ratio_over_states": max((r["maximum_ratio"] for r in records), default=0.0),
        "states": records,
        "failure": failure,
    }
    (arguments.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"states completed        : {len(records)} / {ludwik.shape[0]}")
    print(f"worst Delta p / bound   : {report['maximum_ratio_over_states']:.4f}")
    print(f"states touching the wall: {beyond}")
    if failure is not None:
        print(f"failure                 : state {failure['state']}: {failure['message']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
