"""Does the plane-stress root still exist where the joint Newton dies?

The 21-unknown UMAT Newton diverges at increment 8 of the frozen history, and
`validation/srix_umat_gps_closure_results.md` records that the failure is
independent of the start, of the Jacobian, of `@IterMax` and of the closure
normalisation. Every one of those tests assumes a root is there to be found.

This script tests that assumption directly. At each committed state along the
history it drives the RAW 3D law over a range of imposed transverse normal
strain and records `sigma_zz(eps_zz)` in the structural frame. The closure
equation of the UMAT is exactly `sigma_zz = 0`, so the curve says how many
roots it has:

- one crossing with a healthy slope -> the wall is a Newton basin problem;
- two crossings with a flat region -> `Cbb` is nearly singular, a conditioning
  problem, and the two branches are about to merge;
- **no crossing** -> the plane-stress root does not exist at that state, the
  UMAT is right to fail, and no solver variant can help.

The in-plane strain and the transverse shears are frozen at their committed
values; only `eps_zz` is swept, which is exactly the scalar closure the nested
reference solves.

Usage:
    MFRONT_BEHAVIOUR_LIBRARY=$PWD/build/mfront/src/libBehaviour.so \\
    .venv/bin/python scripts/diagnose_srix_closure_root_sweep.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

INCREMENTS = 12
MAX_STRAIN = 0.02
ORIENTATION_BUNGE_DEG = (35.0, 20.0, 15.0)


def _history() -> np.ndarray:
    return np.array(
        [
            (i / INCREMENTS) * MAX_STRAIN * np.array([1.0, -0.4, 0.0])
            for i in range(1, INCREMENTS + 1)
        ]
    )


def _bridge(library: str, rotation: np.ndarray) -> Any:
    from fem_inhouse.core.mfront import MFront3DMaterialPointBatch
    from fem_inhouse.core.mfront_behaviours import MFRONT_BEHAVIOURS

    return MFront3DMaterialPointBatch(
        library,
        behaviour_spec=MFRONT_BEHAVIOURS.get("fcc_forest_rubin_srix"),
        point_count=1,
        rotation_global_to_material=rotation[None, :, :],
        thread_count=1,
        behaviour_name="Fcc316LForestRubinSrix",
        behaviour_parameters=None,
    )


def _crossings(values: np.ndarray, responses: np.ndarray) -> list[float]:
    """Sign changes of `responses`, linearly interpolated in `values`."""

    roots: list[float] = []
    finite = np.isfinite(responses)
    for index in range(len(values) - 1):
        if not (finite[index] and finite[index + 1]):
            continue
        left, right = responses[index], responses[index + 1]
        if left == 0.0:
            roots.append(float(values[index]))
        elif left * right < 0.0:
            span = right - left
            roots.append(
                float(values[index] - left * (values[index + 1] - values[index]) / span)
            )
    return roots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=161)
    parser.add_argument("--span", type=float, default=0.012)
    parser.add_argument("--orientation", default="tilted", choices=("identity", "tilted"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/_generated/performance/srix_closure_root_sweep.json"),
    )
    arguments = parser.parse_args()

    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        raise SystemExit("MFRONT_BEHAVIOUR_LIBRARY must be set")

    from fem_inhouse.core.crystal_orientation import rotation_from_euler_bunge_deg
    from fem_inhouse.core.mfront import (
        _ENGINEERING_TO_KELVIN_STRAIN_SCALE,
        _PLANE_STRESS_COMPONENTS,
        _SQRT_TWO,
        _TRANSVERSE_COMPONENTS_3D,
    )

    rotation = (
        np.eye(3)
        if arguments.orientation == "identity"
        else rotation_from_euler_bunge_deg(*ORIENTATION_BUNGE_DEG)
    )
    history = _history()
    offsets = np.linspace(-arguments.span, arguments.span, arguments.samples)

    # One bridge walks the committed path; a second, rebuilt from a snapshot of
    # the first, does the sweep -- so a probe never pollutes the path.
    walker = _bridge(library, rotation)
    committed_transverse = np.zeros(3)
    report: dict[str, Any] = {
        "orientation": arguments.orientation,
        "orientation_bunge_deg": list(ORIENTATION_BUNGE_DEG),
        "samples": arguments.samples,
        "span": arguments.span,
        "increments": {},
    }

    for index, in_plane in enumerate(history, start=1):
        # Sweep from the CURRENT committed state, before advancing it.
        state = np.asarray(walker._manager.s0.internal_state_variables).copy()
        gradients = np.asarray(walker._manager.s0.gradients).copy()
        forces = np.asarray(walker._manager.s0.thermodynamic_forces).copy()

        target = np.zeros(6)
        target[_PLANE_STRESS_COMPONENTS] = in_plane * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
        shears = committed_transverse[1:] / _SQRT_TWO

        responses: list[float] = []
        for offset in offsets:
            probe = _bridge(library, rotation)
            for snapshot in (probe._manager.s0, probe._manager.s1):
                snapshot.internal_state_variables[:] = state
                snapshot.gradients[:] = gradients
                snapshot.thermodynamic_forces[:] = forces
            candidate = target.copy()
            candidate[_TRANSVERSE_COMPONENTS_3D] = np.array(
                [committed_transverse[0] + offset, shears[0], shears[1]]
            )
            try:
                trial = probe.evaluate(candidate[None, :], time_increment=1.0)
                stress = float(trial.stress_kelvin_mpa[0, 2])
            except Exception:
                stress = float("nan")
            responses.append(stress)

        response_array = np.array(responses)
        roots = _crossings(offsets + committed_transverse[0], response_array)
        converged = int(np.count_nonzero(np.isfinite(response_array)))
        report["increments"][str(index)] = {
            "in_plane": in_plane.tolist(),
            "committed_eps_zz": float(committed_transverse[0]),
            "converged_samples": converged,
            "roots": roots,
            "root_count": len(roots),
            "sigma_zz_min": float(np.nanmin(response_array)),
            "sigma_zz_max": float(np.nanmax(response_array)),
            "eps_zz": (offsets + committed_transverse[0]).tolist(),
            "sigma_zz": response_array.tolist(),
        }
        print(
            f"  inc {index:>2}  roots={len(roots)}  "
            f"sigma_zz in [{np.nanmin(response_array):9.2f}, "
            f"{np.nanmax(response_array):9.2f}]  "
            f"converged {converged}/{arguments.samples}  "
            f"roots at {[f'{root:.5f}' for root in roots]}"
        )

        # Advance the committed path on the reference (nested) branch: pick the
        # root closest to the committed value, which is what a continuation
        # method would follow.
        if not roots:
            print("    no plane-stress root: the path cannot be continued")
            break
        chosen = min(roots, key=lambda r: abs(r - committed_transverse[0]))
        advance = target.copy()
        advance[_TRANSVERSE_COMPONENTS_3D] = np.array([chosen, shears[0], shears[1]])
        walker.evaluate(advance[None, :], time_increment=1.0)
        walker.commit()
        committed_transverse = np.array([chosen, committed_transverse[1], committed_transverse[2]])

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
