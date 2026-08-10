#!/usr/bin/env python3
"""Measure the constitutive coupling blocks needed by a monolithic solve.

This is a diagnostic only.  It does not change the production staggered
micromorphic algorithm.  The probe uses the existing native micromorphic J2
adapter and finite differences from one committed material state to measure
the three missing coupling derivatives:

* ``d sigma / d chi`` at fixed imposed strain;
* ``d p / d epsilon`` at fixed external chi;
* ``d p / d chi`` at fixed imposed strain.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from fem_inhouse.core.mfront_native import MFrontNativePlaneStressBatch


def _batch(library: Path) -> MFrontNativePlaneStressBatch:
    return MFrontNativePlaneStressBatch(
        library,
        [250.0],
        [380.0],
        [0.245],
        behaviour_name="PixelMicromorphicLudwikJ2Plasticity",
        micromorphic_coupling_modulus_mpa=2_000.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        default=os.environ.get("MFRONT_BEHAVIOUR_LIBRARY"),
        help="compiled MFront behaviour library (or MFRONT_BEHAVIOUR_LIBRARY)",
    )
    parser.add_argument("--strain-step", type=float, default=1.0e-7)
    parser.add_argument("--chi-step", type=float, default=1.0e-7)
    parser.add_argument("--time-increment", type=float, default=1.0)
    args = parser.parse_args()
    if args.library is None:
        parser.error("--library or MFRONT_BEHAVIOUR_LIBRARY is required")
    if args.strain_step <= 0 or args.chi_step <= 0:
        parser.error("finite-difference steps must be positive")

    target = np.array([[0.008, 0.0003, 0.002]], dtype=float)
    chi0 = 0.002
    batch = _batch(Path(args.library))

    def stress_at(chi: float, strain: np.ndarray) -> np.ndarray:
        batch.set_nonlocal_equivalent_plastic_strain([chi])
        result = batch.evaluate_in_plane(
            strain,
            time_increment=args.time_increment,
            consistent_tangent=True,
        )
        stress = np.array(result.stress_in_plane_mpa, copy=True)
        batch.revert()
        return stress[0]

    def source_at(chi: float, strain: np.ndarray) -> float:
        batch.set_nonlocal_equivalent_plastic_strain([chi])
        source, _ = batch.evaluate_nonlocal_state(
            strain,
            time_increment=args.time_increment,
        )
        value = float(source[0])
        batch.revert()
        return value

    stress_plus = stress_at(chi0 + args.chi_step, target)
    stress_minus = stress_at(chi0 - args.chi_step, target)
    dsigma_dchi = (stress_plus - stress_minus) / (2.0 * args.chi_step)

    dp_depsilon = np.empty(3, dtype=float)
    for component in range(3):
        plus = target.copy()
        minus = target.copy()
        plus[0, component] += args.strain_step
        minus[0, component] -= args.strain_step
        dp_depsilon[component] = (
            source_at(chi0, plus) - source_at(chi0, minus)
        ) / (2.0 * args.strain_step)

    dp_dchi = (
        source_at(chi0 + args.chi_step, target)
        - source_at(chi0 - args.chi_step, target)
    ) / (2.0 * args.chi_step)

    payload = {
        "behaviour": "PixelMicromorphicLudwikJ2Plasticity",
        "target_engineering_strain": target[0].tolist(),
        "chi": chi0,
        "coupling_modulus_mpa": 2000.0,
        "time_increment": args.time_increment,
        "strain_step": args.strain_step,
        "chi_step": args.chi_step,
        "d_sigma_in_plane_d_chi_mpa": dsigma_dchi.tolist(),
        "d_peeq_d_engineering_strain": dp_depsilon.tolist(),
        "d_peeq_d_chi": dp_dchi,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
