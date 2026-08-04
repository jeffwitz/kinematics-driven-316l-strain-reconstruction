"""Validate the Hookean stress reconstruction required by experimental EBI-SRIX."""

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
from fem_inhouse.core.tensor_reconstruction import tensor_to_kelvin_3d
from fem_inhouse.spectral2d.ebi import hookean_plane_stress_relative_error

ORIENTATIONS = ((0.0, 0.0, 0.0), (35.0, 20.0, 15.0), (71.0, 37.0, 11.0))
STRAINS = (
    np.array([0.0, 0.0, 0.0]),
    np.array([1.0e-6, -2.0e-7, 3.0e-7]),
    np.array([1.0e-2, -4.0e-3, 2.0e-3]),
)
YOUNG_MODULUS_MPA = 205_000.0
POISSON_RATIO = 0.30


def full_3d_hookean_error(
    stress: np.ndarray, elastic_strain: np.ndarray, tangent: np.ndarray
) -> float:
    stress_kelvin = tensor_to_kelvin_3d(stress, quantity="stress")
    strain_kelvin = tensor_to_kelvin_3d(elastic_strain, quantity="strain")
    predicted = np.einsum("pij,pj->pi", tangent, strain_kelvin)
    return float(
        np.linalg.norm(stress_kelvin - predicted) / max(float(np.linalg.norm(stress_kelvin)), 1.0)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", "build/mfront/src/libBehaviour.so")
    records: list[dict[str, object]] = []
    for orientation in ORIENTATIONS:
        raw = create_plane_stress_material_batch(
            "mfront-3d-condensed-plane-stress",
            np.array([250.0]),
            np.array([500.0]),
            0.245,
            young_modulus_mpa=YOUNG_MODULUS_MPA,
            poisson_ratio=POISSON_RATIO,
            hardening_mode="ludwik",
            plastic_strain_max=0.2,
            plastic_table_points=1_000,
            first_positive_plastic_strain=1.0e-6,
            mfront_library=library,
            mfront_threads=1,
            mfront_behaviour_id="fcc_forest_rubin_srix",
            constitutive_options={
                "crystal_orientation": {
                    "mode": "homogeneous",
                    "euler_bunge_deg": list(orientation),
                }
            },
        )
        material = CachedHookeanPlaneStressMaterialBatch(raw)
        full_elastic_tangent = material.elastic_tangent_3d_kelvin_mpa
        if full_elastic_tangent is None:
            raise RuntimeError("SRIX backend exposes no full elastic tangent")
        initial_tangent = material.elastic_tangent_in_plane_mpa.copy()
        for state, strain in enumerate(STRAINS):
            trial = material.evaluate_in_plane(
                strain.reshape(1, 3), time_increment=1.0 / len(STRAINS)
            )
            completed = material.complete_trial(trial)
            error = hookean_plane_stress_relative_error(
                completed, material.elastic_tangent_in_plane_mpa
            )
            full_3d_error = full_3d_hookean_error(
                completed.full_stress_tensor_mpa,
                completed.elastic_strain_tensor,
                full_elastic_tangent,
            )
            accumulated_slip = float(completed.observables["accumulated_slip"][0])
            plastic_state_reached = state != 2 or accumulated_slip > 0.0
            records.append(
                {
                    "orientation_bunge_deg": orientation,
                    "state": state,
                    "relative_error": error,
                    "full_3d_relative_error": full_3d_error,
                    "accumulated_slip": accumulated_slip,
                    "plastic_state_reached": plastic_state_reached,
                    "passed": (error < 1.0e-9 and full_3d_error < 1.0e-9 and plastic_state_reached),
                }
            )
            material.commit()
        if not np.array_equal(initial_tangent, material.elastic_tangent_in_plane_mpa):
            raise RuntimeError("cached elastic tangent changed with the plastic state")
    report = {"passed": all(bool(record["passed"]) for record in records), "states": records}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
