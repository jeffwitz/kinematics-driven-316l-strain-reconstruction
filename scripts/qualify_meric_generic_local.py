#!/usr/bin/env python3
"""Compare the 3-D Generic Méric law with the legacy MGIS behaviour."""

from __future__ import annotations

import argparse
from pathlib import Path

import mgis.behaviour as mgis
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-library", type=Path, default=Path("build/mfront/src/libBehaviour.so")
    )
    parser.add_argument(
        "--generic-library", type=Path, default=Path("build/meric-generic/src/libBehaviour.so")
    )
    args = parser.parse_args()

    legacy_behaviour = mgis.load(
        str(args.legacy_library), "Fcc316LMericCailletaud", mgis.Hypothesis.Tridimensional
    )
    generic_behaviour = mgis.load(
        str(args.generic_library), "Fcc316LMericCailletaudGeneric3D", mgis.Hypothesis.Tridimensional
    )
    legacy = mgis.MaterialDataManager(legacy_behaviour, 1)
    generic = mgis.MaterialDataManager(generic_behaviour, 1)
    for data in (legacy, generic):
        for state in (data.s0, data.s1):
            mgis.setExternalStateVariable(state, "Temperature", 293.15)
    for state in (generic.s0, generic.s1):
        mgis.setMaterialProperty(state, "MicromorphicCouplingModulus", 0.0)

    max_stress_error = 0.0
    max_gamma_error = 0.0
    for step in range(1, 7):
        strain = np.zeros(6)
        strain[0] = 0.004 * step / 6
        strain[1] = -0.001 * step / 6
        legacy.s1.gradients[:, :] = strain
        generic.s1.gradients[:, :6] = strain
        generic.s1.gradients[:, 6] = 0.0
        for data in (legacy, generic):
            if not mgis.integrate(
                data, mgis.IntegrationType.IntegrationWithConsistentTangentOperator, 1e-3, 0, 1
            ):
                raise RuntimeError(f"Méric integration failed at step {step}")

        legacy_stress = np.asarray(legacy.s1.thermodynamic_forces)[0]
        generic_forces = np.asarray(generic.s1.thermodynamic_forces)[0]
        legacy_slip = np.asarray(legacy.s1.internal_state_variables)[0][18:30]
        max_stress_error = max(
            max_stress_error, float(np.max(np.abs(legacy_stress - generic_forces[:6])))
        )
        max_gamma_error = max(max_gamma_error, float(abs(generic_forces[6] - legacy_slip.sum())))
        mgis.update(legacy)
        mgis.update(generic)

    print(f"meric_generic_local_max_stress_error={max_stress_error:.16e}")
    print(f"meric_generic_local_max_gamma_error={max_gamma_error:.16e}")
    if max_stress_error > 1e-7 or max_gamma_error > 1e-12:
        raise SystemExit("Generic Méric local equivalence check failed")


if __name__ == "__main__":
    main()
