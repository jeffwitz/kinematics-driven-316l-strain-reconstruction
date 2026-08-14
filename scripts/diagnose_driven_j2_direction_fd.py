#!/usr/bin/env python3
"""Finite-difference check of the local driven-J2 direction derivative."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import root

from fem_inhouse.core.constitutive import PLANE_STRESS_VON_MISES_METRIC, von_mises
from fem_inhouse.core.driven_j2 import DrivenJ2PlaneStressBatch
from fem_inhouse.core.element import plane_stress_elasticity


def main() -> None:
    elasticity = np.asarray(plane_stress_elasticity(205_000.0, 0.30))
    inverse_metric = np.linalg.inv(PLANE_STRESS_VON_MISES_METRIC)
    strain = np.array([[0.004, -0.0015, 0.002]], dtype=np.float64)
    increment = np.array([2.0e-4])
    material = DrivenJ2PlaneStressBatch(
        1, young_modulus_mpa=205_000.0, poisson_ratio=0.30
    )
    trial = material.evaluate(strain, increment, time_increment=1.0)
    stress = trial.stress_in_plane_mpa[0]
    direction = trial.observables["flow_direction"][0]
    trial_stress = elasticity @ strain[0]
    raw = np.array([0.7, -0.2, 0.5])
    q = von_mises(stress[None])[0]
    projected = raw - direction * (stress @ raw / q)
    analytic = -increment[0] * trial.tangent_in_plane_mpa[0] @ projected

    def solve(parameter: float) -> np.ndarray:
        def residual(candidate: np.ndarray) -> np.ndarray:
            candidate_direction = PLANE_STRESS_VON_MISES_METRIC @ candidate / von_mises(
                candidate[None]
            )[0]
            perturbed = candidate_direction + parameter * projected
            norm = np.sqrt(perturbed @ inverse_metric @ perturbed)
            flow = perturbed / norm
            return candidate - trial_stress + increment[0] * elasticity @ flow

        result = root(residual, stress, method="lm", options={"ftol": 1.0e-14, "xtol": 1.0e-14})
        if not result.success:
            raise RuntimeError(result.message)
        return result.x

    rows = []
    for step in 10.0 ** (-np.arange(2, 7, dtype=float)):
        finite_difference = (solve(step) - solve(-step)) / (2.0 * step)
        error = np.linalg.norm(finite_difference - analytic) / np.linalg.norm(analytic)
        rows.append({"step": float(step), "relative_error": float(error)})
    print(json.dumps({"analytic": analytic.tolist(), "checks": rows}, indent=2))


if __name__ == "__main__":
    main()
