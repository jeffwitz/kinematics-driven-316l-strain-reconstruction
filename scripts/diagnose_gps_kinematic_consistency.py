"""Is the UMAT GPS state a solution of its own elastic residual?

There is one identity every state of this law must satisfy, whatever branch it
sits on, and it needs no reference to compare against.

The elastic residual is `feel = deel - deto_m + sum(dg m)`. The Schmid tensors
`m` are deviatoric, so `tr(m) = 0`, and a rotation preserves the trace. Taking
the trace of a converged `feel` therefore gives

    tr(eel) = tr(eps_total)

with `eps_total` the in-plane strain the bridge imposes plus the transverse
strain the closure state variables hold. This is an identity of the residual,
not a physical claim: a converged state that violates it is not a solution of
the system the law says it solved.

The reference cannot fail this way. There the transverse strain is IMPOSED, so
the law receives a complete gradient and its kinematics are consistent by
construction. The identity only has something to say once the transverse strain
becomes an unknown of the same system -- which is exactly what the UMAT does.

Usage:
    MFRONT_BEHAVIOUR_LIBRARY=$PWD/build/mfront/src/libBehaviour.so \\
    .venv/bin/python scripts/diagnose_gps_kinematic_consistency.py
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
ORIENTATIONS: dict[str, tuple[float, float, float] | None] = {
    "identity": None,
    "bunge_35_20_15": (35.0, 20.0, 15.0),
}


def _batch(library: str, euler: tuple[float, float, float] | None, parameter_set: str) -> Any:
    from fem_inhouse.core.crystal_orientation import rotation_from_euler_bunge_deg
    from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch

    options: dict[str, Any] = {"parameter_set": parameter_set}
    if euler is not None:
        options["crystal_orientation"] = {
            "mode": "homogeneous",
            "matrix": rotation_from_euler_bunge_deg(*euler).tolist(),
        }
    return create_plane_stress_material_batch(
        "mfront-native-generalised-plane-stress",
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
        mfront_behaviour_id="fcc_forest_rubin_srix_gps",
        constitutive_options=options,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter-set", default="316l_srix_exploratory_r1")
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/_generated/performance/gps_kinematic_consistency.json"),
    )
    arguments = parser.parse_args()

    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        raise SystemExit("MFRONT_BEHAVIOUR_LIBRARY must be set")

    from fem_inhouse.core.mfront import (
        _ENGINEERING_TO_KELVIN_STRAIN_SCALE,
        _PLANE_STRESS_COMPONENTS,
    )

    report: dict[str, Any] = {"parameter_set": arguments.parameter_set, "cases": {}}
    for name, euler in ORIENTATIONS.items():
        batch = _batch(library, euler, arguments.parameter_set)
        inner = getattr(batch, "_material", batch)
        records: list[dict[str, float]] = []
        for index in range(1, arguments.increments + 1):
            in_plane = (index / INCREMENTS) * MAX_STRAIN * np.array([1.0, -0.4, 0.0])
            batch.evaluate(np.atleast_2d(in_plane), time_increment=1.0 / INCREMENTS)
            state = np.asarray(inner._manager.s1.internal_state_variables)[0]
            elastic = state[inner._elastic_offset : inner._elastic_offset + 6]
            total = np.zeros(6)
            total[_PLANE_STRESS_COMPONENTS] = (
                in_plane * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
            )
            total[2] = state[inner._ezz_offset]
            total[4] = state[inner._exz_offset]
            total[5] = state[inner._eyz_offset]
            elastic_trace = float(elastic[:3].sum())
            total_trace = float(total[:3].sum())
            defect = abs(elastic_trace - total_trace) / max(abs(total_trace), 1e-30)
            records.append(
                {
                    "increment": index,
                    "trace_elastic": elastic_trace,
                    "trace_total": total_trace,
                    "relative_defect": defect,
                    "eps_zz": float(state[inner._ezz_offset]),
                }
            )
            print(
                f"  {name:<16} inc {index:>2}  tr(eel)={elastic_trace:+.4e}  "
                f"tr(eps)={total_trace:+.4e}  defect={defect:.2e}  "
                f"eps_zz={state[inner._ezz_offset]:+.5f}"
            )
            batch.commit()
        report["cases"][name] = {
            "records": records,
            "maximum_relative_defect": max(item["relative_defect"] for item in records),
        }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
