"""Finite-difference qualification of the SRIX compact-slip tangent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

HISTORY = np.array(
    [(i / 12.0) * 0.02 * np.array([1.0, -0.4, 0.0]) for i in range(1, 13)]
)


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-30))


def make_batch(
    library: str,
    delta: float,
    zero_derivative: float,
    angles: tuple[float, float, float] | None,
):
    from fem_inhouse.core.crystal_orientation import rotation_from_euler_bunge_deg
    from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch

    options: dict[str, object] = {
        "parameter_set": "316l_srix_transposed_from_nasri2018_rate_1e-3",
        "srix_slip_smoothing_delta": delta,
        "srix_slip_zero_derivative": zero_derivative,
    }
    if angles is not None:
        options["crystal_orientation"] = {
            "mode": "homogeneous",
            "matrix": np.asarray(rotation_from_euler_bunge_deg(*angles)).tolist(),
        }
    return create_plane_stress_material_batch(
        "mfront-native-generalised-plane-stress",
        np.full((1, 1), 250.0), np.full((1, 1), 500.0), 0.245,
        young_modulus_mpa=205000.0, poisson_ratio=0.3,
        hardening_mode="ludwik", plastic_strain_max=0.2,
        plastic_table_points=1000, first_positive_plastic_strain=1.0e-6,
        mfront_library=library, mfront_threads=1,
        mfront_behaviour_id="fcc_forest_rubin_srix",
        constitutive_options=options,
    )


def check_history(batch, step: float) -> dict[str, object]:
    errors: list[float] = []
    transverse: list[float] = []
    for in_plane in HISTORY:
        strain = in_plane[None, :]
        base = batch.evaluate(strain, time_increment=1.0 / 12.0)
        returned = np.asarray(base.tangent_in_plane_mpa)[0]
        fd = np.zeros((3, 3))
        for column in range(3):
            plus = strain.copy()
            minus = strain.copy()
            plus[0, column] += step
            minus[0, column] -= step
            stress_plus = np.asarray(
                batch.evaluate(plus, time_increment=1.0 / 12.0).stress_in_plane_mpa
            )[0]
            stress_minus = np.asarray(
                batch.evaluate(minus, time_increment=1.0 / 12.0).stress_in_plane_mpa
            )[0]
            fd[:, column] = (stress_plus - stress_minus) / (2.0 * step)
        errors.append(relative_error(returned, fd))
        transverse.append(float(np.max(np.abs(base.plane_stress_residual_mpa))))
        batch.revert()
        batch.commit()
    return {
        "max_relative_tangent_error": max(errors),
        "tangent_errors_by_increment": errors,
        "max_transverse_residual_mpa": max(transverse),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--library", default=os.environ.get("MFRONT_BEHAVIOUR_LIBRARY"), required=False
    )
    parser.add_argument("--delta", type=float, default=1.0e-5)
    parser.add_argument("--zero-derivative", type=float, default=-1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.library or not Path(args.library).is_file():
        parser.error("--library is required or MFRONT_BEHAVIOUR_LIBRARY must point to a library")
    report: dict[str, object] = {
        "delta": args.delta,
        "zero_derivative": args.zero_derivative,
        "steps": [1.0e-5, 1.0e-6, 1.0e-7],
        "cases": {},
    }
    for name, angles in {"identity": None, "bunge_35_20_15": (35.0, 20.0, 15.0)}.items():
        report["cases"][name] = {}
        for step in report["steps"]:
            batch = make_batch(
                str(Path(args.library).resolve()),
                args.delta,
                args.zero_derivative,
                angles,
            )
            report["cases"][name][str(step)] = check_history(batch, step)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
