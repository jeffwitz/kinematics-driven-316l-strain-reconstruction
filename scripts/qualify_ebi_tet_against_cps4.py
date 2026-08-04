"""Qualify experimental one-state EBI-TET SRIX mechanics against CPS4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from fem_inhouse.config import CaseStudyConfig, MaterialConfig, SolverConfig
from fem_inhouse.core.plane_stress_material import (
    CachedHookeanPlaneStressMaterialBatch,
    create_plane_stress_material_batch,
)
from fem_inhouse.solver import run_case_study
from fem_inhouse.spectral2d import (
    EBISpectralSolverConfig,
    StructuredGrid2D,
    solve_ebi_dirichlet_plane_stress,
)

try:
    from scripts.qualify_spectral2d_against_newton import (
        ORIENTATION_BUNGE_DEG,
        build_case,
        relative_l2,
    )
except ModuleNotFoundError:  # Direct script execution.
    from qualify_spectral2d_against_newton import (  # type: ignore[no-redef]
        ORIENTATION_BUNGE_DEG,
        build_case,
        relative_l2,
    )


def solve_cps4(case, library: str, increments: int, tolerance: float):
    solver = SolverConfig(
        increments=increments,
        residual_tolerance=tolerance,
        constitutive_backend="mfront-3d-condensed-plane-stress",
        mfront_library=library,
        mfront_behaviour_id="fcc_forest_rubin_srix",
        element_formulation="cps4",
        constitutive_options={
            "crystal_orientation": {
                "mode": "homogeneous",
                "euler_bunge_deg": list(ORIENTATION_BUNGE_DEG),
            }
        },
    )
    return run_case_study(
        CaseStudyConfig(mesh=case["mesh"], material=MaterialConfig(), solver=solver),
        displacement_x_mm=case["displacement_x_mm"],
        displacement_y_mm=case["displacement_y_mm"],
        yield_stress_mpa=case["yield_stress_mpa"],
        hardening_coefficient_mpa=case["hardening_coefficient_mpa"],
    )


def solve_ebi(case, library: str, increments: int, tolerance: float, scale: float):
    mesh = case["mesh"]
    point_count = mesh.nx * mesh.ny
    raw = create_plane_stress_material_batch(
        "mfront-3d-condensed-plane-stress",
        np.full(point_count, 250.0),
        np.full(point_count, 500.0),
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
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
                "euler_bunge_deg": list(ORIENTATION_BUNGE_DEG),
            }
        },
    )
    material = CachedHookeanPlaneStressMaterialBatch(raw)
    final = np.stack((case["displacement_x_mm"], case["displacement_y_mm"]), axis=-1)
    history = np.stack([fraction * final for fraction in np.linspace(0.0, 1.0, increments + 1)])
    started = time.perf_counter()
    result = solve_ebi_dirichlet_plane_stress(
        grid=StructuredGrid2D(mesh.nx, mesh.ny, *mesh.physical_size_mm),
        material=material,
        boundary_displacement_history=history,
        config=EBISpectralSolverConfig(
            relative_equilibrium_tolerance=tolerance,
            reference_parameter_scale=scale,
        ),
    )
    return result, time.perf_counter() - started


def digest_fields(*fields: np.ndarray) -> str:
    digest = hashlib.sha256()
    for field in fields:
        digest.update(np.ascontiguousarray(field).view(np.uint8))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=int, default=12)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--tolerances", type=float, nargs="+", default=(1.0e-6, 1.0e-8))
    parser.add_argument("--reference-scales", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument("--output", type=Path, default=Path("validation/_generated/ebi_tet"))
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", "build/mfront/src/libBehaviour.so")
    case = build_case(arguments.mesh)
    report = {
        "status": "experimental",
        "mesh": arguments.mesh,
        "increments": arguments.increments,
        "material_states_per_pixel": 1,
        "kinematic_samples_per_pixel": 2,
        "orientation_bunge_deg": ORIENTATION_BUNGE_DEG,
        "runs": [],
    }
    for tolerance in arguments.tolerances:
        cps4_started = time.perf_counter()
        cps4 = solve_cps4(case, library, arguments.increments, tolerance)
        cps4_elapsed = time.perf_counter() - cps4_started
        for scale in arguments.reference_scales:
            record = {"tolerance": tolerance, "reference_scale": scale}
            try:
                ebi, elapsed = solve_ebi(case, library, arguments.increments, tolerance, scale)
                slip = ebi.observables["accumulated_slip"]
                record.update(
                    {
                        "converged": True,
                        "elapsed_seconds": elapsed,
                        "cps4_elapsed_seconds": cps4_elapsed,
                        "newton_iterations": sum(ebi.diagnostics.iterations_per_increment),
                        "gmres_iterations": int(ebi.diagnostics.timings["gmres_iterations"]),
                        "material_evaluations": int(
                            ebi.diagnostics.timings["material_evaluations"]
                        ),
                        "final_residual": (ebi.diagnostics.dimensionless_equilibrium_history[-1]),
                        "verification_residual": (ebi.diagnostics.verification_residual),
                        "E_u": relative_l2(ebi.displacement, cps4.displacement_mm),
                        "E_sigma": relative_l2(
                            ebi.stress_in_plane_mpa.mean(axis=2), cps4.stress_mpa
                        ),
                        "E_Gamma": relative_l2(slip, cps4.cumulated_slip),
                        "E_R": relative_l2(ebi.reaction_forces, cps4.reaction_force),
                        "high_frequency_fraction_max": max(
                            ebi.diagnostics.high_frequency_energy_fraction_history,
                            default=0.0,
                        ),
                        "field_sha256": digest_fields(
                            ebi.displacement,
                            ebi.stress_in_plane_mpa,
                            slip,
                            ebi.reaction_forces,
                        ),
                    }
                )
                np.savez_compressed(
                    arguments.output / f"ebi_m{arguments.mesh}_tol{tolerance:.0e}_s{scale:g}.npz",
                    displacement=ebi.displacement,
                    stress_in_plane_mpa=ebi.stress_in_plane_mpa,
                    accumulated_slip=slip,
                    reaction_forces=ebi.reaction_forces,
                )
            except Exception as error:
                record.update(
                    {
                        "converged": False,
                        "failure": type(error).__name__,
                        "message": str(error)[:500],
                    }
                )
            report["runs"].append(record)
    destination = arguments.output / "ebi_tet_against_cps4.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
