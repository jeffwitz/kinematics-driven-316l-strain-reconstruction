"""Separate TRI2 stencil error from EBI state-sharing error."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch
from fem_inhouse.spectral2d import EBISpectralSolverConfig, StructuredGrid2D
from fem_inhouse.spectral2d.newton_two_state import solve_two_state_dirichlet_plane_stress

try:
    from scripts.qualify_ebi_tet_against_cps4 import solve_cps4, solve_ebi
    from scripts.qualify_spectral2d_against_newton import build_case, relative_l2
except ModuleNotFoundError:  # Direct script execution.
    from qualify_ebi_tet_against_cps4 import solve_cps4, solve_ebi  # type: ignore[no-redef]
    from qualify_spectral2d_against_newton import build_case, relative_l2  # type: ignore[no-redef]


def solve_two_state(case, library: str, increments: int, tolerance: float, scale: float):
    mesh = case["mesh"]
    points = 2 * mesh.nx * mesh.ny
    material = create_plane_stress_material_batch(
        "mfront-3d-condensed-plane-stress",
        np.full(points, 250.0),
        np.full(points, 500.0),
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
            "crystal_orientation": {"mode": "homogeneous", "euler_bunge_deg": [35.0, 20.0, 15.0]}
        },
    )
    final = np.stack((case["displacement_x_mm"], case["displacement_y_mm"]), axis=-1)
    history = np.stack([fraction * final for fraction in np.linspace(0.0, 1.0, increments + 1)])
    started = time.perf_counter()
    result = solve_two_state_dirichlet_plane_stress(
        grid=StructuredGrid2D(mesh.nx, mesh.ny, *mesh.physical_size_mm),
        material=material,
        boundary_displacement_history=history,
        config=EBISpectralSolverConfig(
            relative_equilibrium_tolerance=tolerance,
            reference_parameter_scale=scale,
        ),
    )
    return result, time.perf_counter() - started


def side_resultants(reaction: np.ndarray) -> np.ndarray:
    values = np.asarray(reaction)
    return np.array(
        (
            values[0].sum(axis=0),
            values[-1].sum(axis=0),
            values[:, 0].sum(axis=0),
            values[:, -1].sum(axis=0),
        )
    )


def moment(reaction: np.ndarray, grid: StructuredGrid2D) -> float:
    x, y = grid.coordinates
    return float(np.sum(x * reaction[..., 1] - y * reaction[..., 0]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=int, default=12)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", "build/mfront/src/libBehaviour.so")
    case = build_case(arguments.mesh)
    cps4 = solve_cps4(case, library, arguments.increments, arguments.tolerance)
    ebi, ebi_time = solve_ebi(
        case,
        library,
        arguments.increments,
        arguments.tolerance,
        1.0,
    )
    tet, tet_time = solve_two_state(case, library, arguments.increments, arguments.tolerance, 1.0)
    grid = StructuredGrid2D(arguments.mesh, arguments.mesh, *case["mesh"].physical_size_mm)
    ebi_slip = ebi.observables["accumulated_slip"]
    tet_slip = tet.observables["accumulated_slip"].mean(axis=2)
    ebi_stress = ebi.stress_in_plane_mpa.mean(axis=2)
    tet_stress = tet.stress_in_plane_mpa.mean(axis=2)
    ebi_sides = side_resultants(ebi.reaction_forces)
    tet_sides = side_resultants(tet.reaction_forces)
    cps_sides = side_resultants(cps4.reaction_force)
    report = {
        "mesh": arguments.mesh,
        "tolerance": arguments.tolerance,
        "material_states_per_pixel": {"ebi": 1, "tet_two_state": 2},
        "errors": {
            "tet_cps4": {
                "E_u": relative_l2(tet.displacement, cps4.displacement_mm),
                "E_sigma": relative_l2(tet_stress, cps4.stress_mpa),
                "E_Gamma": relative_l2(tet_slip, cps4.cumulated_slip),
                "E_R_nodal": relative_l2(tet.reaction_forces, cps4.reaction_force),
                "E_R_sides": relative_l2(tet_sides, cps_sides),
                "E_Mz": abs(moment(tet.reaction_forces, grid) - moment(cps4.reaction_force, grid))
                / max(abs(moment(cps4.reaction_force, grid)), 1.0e-30),
            },
            "ebi_tet": {
                "E_u": relative_l2(ebi.displacement, tet.displacement),
                "E_sigma": relative_l2(ebi_stress, tet_stress),
                "E_Gamma": relative_l2(ebi_slip, tet_slip),
                "E_R_nodal": relative_l2(ebi.reaction_forces, tet.reaction_forces),
                "E_R_sides": relative_l2(ebi_sides, tet_sides),
                "E_Mz": abs(moment(ebi.reaction_forces, grid) - moment(tet.reaction_forces, grid))
                / max(abs(moment(tet.reaction_forces, grid)), 1.0e-30),
            },
            "ebi_cps4": {
                "E_u": relative_l2(ebi.displacement, cps4.displacement_mm),
                "E_sigma": relative_l2(ebi_stress, cps4.stress_mpa),
                "E_Gamma": relative_l2(ebi_slip, cps4.cumulated_slip),
                "E_R_nodal": relative_l2(ebi.reaction_forces, cps4.reaction_force),
                "E_R_sides": relative_l2(ebi_sides, cps_sides),
                "E_Mz": abs(moment(ebi.reaction_forces, grid) - moment(cps4.reaction_force, grid))
                / max(abs(moment(cps4.reaction_force, grid)), 1.0e-30),
            },
        },
        "side_resultants": {
            "cps4": cps_sides.tolist(),
            "tet_two_state": tet_sides.tolist(),
            "ebi": ebi_sides.tolist(),
        },
        "timings": {"ebi_seconds": ebi_time, "tet_two_state_seconds": tet_time},
        "iterations": {
            "ebi_newton": sum(ebi.diagnostics.iterations_per_increment),
            "tet_newton": sum(tet.diagnostics.iterations_per_increment),
            "ebi_gmres": int(ebi.diagnostics.timings["gmres_iterations"]),
            "tet_gmres": int(tet.diagnostics.timings["gmres_iterations"]),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
