"""Compare the first spectral solver on the registered Newton/Broyden SRIX case.

This deliberately reuses the case construction from ``qualify_broyden_correction``:
12x12 pixels, eight increments, non-affine DIC-like boundary data and the
homogeneous SRIX orientation.  CPS4 remains the scientific reference; the
unaccelerated CPS4R-AS and global-Broyden runs are retained as algorithmic
baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from fem_inhouse.config import CaseStudyConfig, MaterialConfig, MeshConfig, SolverConfig
from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch
from fem_inhouse.results import FEMResult
from fem_inhouse.solver import run_case_study
from fem_inhouse.spectral2d import (
    Spectral2DConfig,
    StructuredGrid2D,
    solve_dirichlet_plane_stress_spectral,
)

SPACING_MM = 0.00184
ORIENTATION_BUNGE_DEG = (35.0, 20.0, 15.0)


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.linalg.norm(reference))
    difference = float(np.linalg.norm(candidate - reference))
    return difference / scale if scale > 0.0 else difference


def pixel_average_if_needed(values: np.ndarray, scheme: str) -> np.ndarray:
    values = np.asarray(values)
    return values.mean(axis=2) if scheme == "tri2" else values


def build_case(mesh_size: int) -> dict[str, Any]:
    mesh = MeshConfig(nx=mesh_size, ny=mesh_size, base_pixel_size_mm=SPACING_MM, scale_factor=1.0)
    span = mesh.physical_size_mm[0]
    nodes = np.linspace(0.0, span, mesh_size + 1)
    grid_x, grid_y = np.meshgrid(nodes, nodes, indexing="ij")
    perturbation = 0.05 * 0.010 * span
    return {
        "mesh": mesh,
        "displacement_x_mm": -0.004 * grid_x
        + perturbation * np.sin(2.0 * np.pi * grid_y / span) * (grid_x / span),
        "displacement_y_mm": 0.010 * grid_y
        + perturbation * np.sin(2.0 * np.pi * grid_x / span) * (grid_y / span),
        "yield_stress_mpa": np.full((mesh_size, mesh_size), 250.0),
        "hardening_coefficient_mpa": np.full((mesh_size, mesh_size), 500.0),
    }


def solve_fem(case: dict[str, Any], formulation: str, library: str, increments: int) -> FEMResult:
    base_formulation = cast(
        Literal["cps4", "cps4r", "cps4r_as"],
        "cps4r_as" if formulation == "cps4r_as_broyden" else formulation,
    )
    solver = SolverConfig(
        increments=increments,
        residual_tolerance=1.0e-6,
        constitutive_backend="mfront-3d-condensed-plane-stress",
        mfront_library=library,
        mfront_behaviour_id="fcc_forest_rubin_srix",
        element_formulation=base_formulation,
        jacobian_correction="global_broyden" if formulation == "cps4r_as_broyden" else "none",
        jacobian_correction_memory=1,
        newton_line_search=formulation == "cps4r_as_broyden",
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


def solve_spectral(
    case: dict[str, Any], scheme: str, library: str, increments: int, tolerance: float,
    trace_callback=None,
) -> tuple[Any, float]:
    mesh = case["mesh"]
    grid = StructuredGrid2D(mesh.nx, mesh.ny, *mesh.physical_size_mm)
    final = np.stack((case["displacement_x_mm"], case["displacement_y_mm"]), axis=-1)
    history = np.stack([fraction * final for fraction in np.linspace(0.0, 1.0, increments + 1)])
    points = mesh.nx * mesh.ny * (2 if scheme == "tri2" else 1)
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
            "crystal_orientation": {
                "mode": "homogeneous",
                "euler_bunge_deg": list(ORIENTATION_BUNGE_DEG),
            }
        },
    )
    started = time.perf_counter()
    result = solve_dirichlet_plane_stress_spectral(
        grid=grid,
        material=material,
        boundary_displacement_history=history,
        config=Spectral2DConfig(
            spatial_scheme=cast(Literal["quad1", "tri2"], scheme),
            relative_equilibrium_tolerance=tolerance,
            reference_lambda_0=69_230.76923076923,
            reference_mu_0=78_846.15384615384,
        ),
        trace_callback=trace_callback,
    )
    return result, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=int, default=12)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--output", type=Path, default=Path("validation/_generated/spectral2d"))
    arguments = parser.parse_args()
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", "build/mfront/src/libBehaviour.so")
    arguments.output.mkdir(parents=True, exist_ok=True)
    case = build_case(arguments.mesh)

    references: dict[str, Any] = {}
    for name, formulation in (
        ("cps4", "cps4"),
        ("cps4r_as", "cps4r_as"),
        ("global_broyden_m1", "cps4r_as_broyden"),
    ):
        fem_result = solve_fem(case, formulation, library, arguments.increments)
        references[name] = fem_result

    report: dict[str, Any] = {
        "mesh": arguments.mesh,
        "increments": arguments.increments,
        "spectral_tolerance": arguments.tolerance,
        "orientation_bunge_deg": list(ORIENTATION_BUNGE_DEG),
        "variants": {},
    }
    reference = references["cps4"]
    for scheme in ("quad1", "tri2"):
        timings: list[float] = []
        spectral_result = None
        failure: Exception | None = None
        for _ in range(arguments.repeats):
            trace_file = None
            try:
                trace_path = arguments.output / (
                    f"spectral2d_{scheme}_tol_{arguments.tolerance:.0e}_trace.jsonl"
                )
                trace_file = trace_path.open("w", encoding="utf-8")

                def write_trace(event: dict[str, object], stream=trace_file) -> None:
                    stream.write(json.dumps(event) + "\n")
                    stream.flush()

                spectral_result, elapsed = solve_spectral(
                    case, scheme, library, arguments.increments, arguments.tolerance,
                    write_trace,
                )
                trace_file.close()
                timings.append(elapsed)
            except Exception as exception:
                if trace_file is not None:
                    trace_file.close()
                failure = exception
                break
        if spectral_result is None:
            report["variants"][scheme] = {
                "converged": False,
                "failure": type(failure).__name__ if failure is not None else "unknown",
                "message": str(failure)[:300] if failure is not None else "unknown",
            }
            continue
        spectral_diagnostics = spectral_result.diagnostics
        spectral_slip = spectral_result.observables.get("accumulated_slip")
        fem_slip = reference.cumulated_slip
        if spectral_slip is None or fem_slip is None:
            raise RuntimeError("both solvers must expose accumulated slip")
        report["variants"][scheme] = {
            "converged": True,
            "elapsed_median": statistics.median(timings),
            "iterations": sum(spectral_diagnostics.iterations_per_increment),
            "cutbacks": spectral_diagnostics.cutbacks,
            "E_u_against_cps4": relative_l2(
                spectral_result.displacement, reference.displacement_mm
            ),
            "E_sigma_against_cps4": relative_l2(
                pixel_average_if_needed(spectral_result.stress_in_plane_mpa, scheme),
                reference.stress_mpa,
            ),
            "E_Gamma_against_cps4": relative_l2(
                pixel_average_if_needed(spectral_slip, scheme), fem_slip
            ),
            "E_R_against_cps4": relative_l2(
                spectral_result.reaction_forces, reference.reaction_force
            ),
            "plane_stress_residual_mpa": spectral_diagnostics.maximum_plane_stress_residual_mpa,
            "fem_plane_stress_residual_mpa": (
                float(np.max(np.abs(reference.plane_stress_residual_mpa)))
                if reference.plane_stress_residual_mpa is not None
                else None
            ),
            "equilibrium_residual_dimensionless": (
                spectral_diagnostics.dimensionless_equilibrium_history[-1]
            ),
            "equilibrium_residual_dimensional": spectral_diagnostics.absolute_residual_history[-1],
            "verification_residual": spectral_diagnostics.verification_residual,
            "material_points": spectral_diagnostics.material_points,
        }
    report["fem_baselines"] = {
        name: {
            "iterations": result.diagnostics.total_newton_iterations,
            "cutbacks": result.diagnostics.cutbacks,
            "final_relative_residual": result.diagnostics.final_relative_residual,
            "final_residual_norm": result.diagnostics.final_residual_norm,
            "elapsed_seconds": None,
        }
        for name, result in references.items()
    }
    destination = arguments.output / "spectral2d_against_newton_broyden.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
