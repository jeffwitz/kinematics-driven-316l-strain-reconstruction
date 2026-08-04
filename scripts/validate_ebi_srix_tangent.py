"""Directional derivative check of the EBI-SRIX matrix-free Jacobian."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from fem_inhouse.core.plane_stress_material import (
    CachedHookeanPlaneStressMaterialBatch,
    create_plane_stress_material_batch,
)
from fem_inhouse.spectral2d import (
    EBIPlaneStressElementBatch,
    EBITwoTriangleKinematics2D,
    StructuredGrid2D,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    grid = StructuredGrid2D(2, 2, 1.0, 1.0)
    raw = create_plane_stress_material_batch(
        "mfront-3d-condensed-plane-stress",
        np.full(4, 250.0),
        np.full(4, 500.0),
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1_000,
        first_positive_plastic_strain=1.0e-6,
        mfront_library=os.environ.get(
            "MFRONT_BEHAVIOUR_LIBRARY", "build/mfront/src/libBehaviour.so"
        ),
        mfront_threads=1,
        mfront_behaviour_id="fcc_forest_rubin_srix",
        constitutive_options={
            "crystal_orientation": {
                "mode": "homogeneous",
                "euler_bunge_deg": [35.0, 20.0, 15.0],
            }
        },
    )
    material = CachedHookeanPlaneStressMaterialBatch(raw)
    elements = EBIPlaneStressElementBatch(material, grid.pixel_shape)
    kinematics = EBITwoTriangleKinematics2D(grid)
    prestrain = np.broadcast_to(np.array([5.0e-3, -2.0e-3, 1.0e-3]), (2, 2, 2, 3))
    pretrial = elements.evaluate_samples(prestrain, time_increment=0.5, consistent_tangent=True)
    prestate = elements.complete_trial(pretrial)
    accumulated_slip = float(np.max(prestate.observables["accumulated_slip"]))
    elements.commit()

    x, y = grid.coordinates
    displacement = np.zeros((*grid.node_shape, 2))
    displacement[..., 0] = 6.0e-3 * x[:, None] + 2.0e-4 * y[None, :]
    displacement[..., 1] = -2.0e-3 * y[None, :] + 1.0e-4 * x[:, None]
    displacement[1, 1] += np.array([2.0e-4, -1.0e-4])
    direction = np.zeros_like(displacement)
    direction[1, 1] = np.array([0.7, -0.4])
    base_strain = kinematics.strain_samples(displacement)
    trial = elements.evaluate_samples(base_strain, time_increment=0.5, consistent_tangent=True)
    analytical = elements.tangent_action(direction, kinematics=kinematics, trial=trial)
    base_residual = kinematics.divergence_from_sample_stress(trial.sample_stress_mpa)
    errors = {}
    for step in (1.0e-6, 3.0e-7, 1.0e-7, 3.0e-8):
        perturbed = elements.evaluate_samples(
            kinematics.strain_samples(displacement + step * direction),
            time_increment=0.5,
            consistent_tangent=True,
        )
        perturbed_residual = kinematics.divergence_from_sample_stress(perturbed.sample_stress_mpa)
        numerical = (perturbed_residual - base_residual) / step
        errors[f"{step:.0e}"] = float(
            np.linalg.norm(numerical - analytical) / max(float(np.linalg.norm(analytical)), 1.0)
        )
    minimum_error = min(errors.values())
    report = {
        "passed": accumulated_slip > 0.0 and minimum_error < 1.0e-5,
        "accumulated_slip": accumulated_slip,
        "relative_errors": errors,
        "minimum_relative_error": minimum_error,
        "material_states_per_pixel": 1,
        "kinematic_samples_per_pixel": 2,
    }
    elements.revert()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
