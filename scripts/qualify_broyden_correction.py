"""Does the Broyden correction of the hourglass Jacobian earn its place?

Section 24 of the 2026-08-04 specification, against the thresholds frozen in
`validation/cps4r_as_broyden_preregistration.md`.

The comparison that decides the question is **CPS4R-AS against itself**: same
element, same stabilisation, same residual, same converged solution, and only
the matrix handed to Newton differs. Anything but the iteration count moving is
a defect, not a result -- so `E_u`, `E_sigma` and `E_Gamma` are measured against
the un-accelerated run with a `1e-6` bound, and the errors against CPS4 are
measured again to show they have not drifted.

The memory is swept over `1, 3, 5` and nothing else. A memory that only wins on
one case does not become the default. Use `--correction global_broyden` to
qualify the global direction accelerator.

Usage:
    MFRONT_BEHAVIOUR_LIBRARY=$PWD/build/mfront/src/libBehaviour.so \\
    python scripts/qualify_broyden_correction.py --repeats 5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from fem_inhouse.config import CaseStudyConfig, MaterialConfig, MeshConfig, SolverConfig
from fem_inhouse.results import FEMResult
from fem_inhouse.solver import run_case_study

# Deliberately the same registered case as
# `compare_reduced_integration_formulations.py`, so the 47-against-37 iteration
# count this work exists to close is the very number being re-measured.
SPACING_MM = 0.00184
ORIENTATION_BUNGE_DEG = (35.0, 20.0, 15.0)
MEMORIES = (1, 3, 5)

COMPARED_FIELDS: dict[str, str] = {
    "displacement_mm": "E_u",
    "stress_mpa": "E_sigma",
    "reaction_force": "E_R",
    "cumulated_slip": "E_Gamma",
}


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.linalg.norm(reference))
    difference = float(np.linalg.norm(candidate - reference))
    return difference / scale if scale > 0.0 else difference


def _errors(candidate: FEMResult, reference: FEMResult) -> dict[str, float]:
    errors: dict[str, float] = {}
    for field, criterion in COMPARED_FIELDS.items():
        left = getattr(candidate, field, None)
        right = getattr(reference, field, None)
        if left is None or right is None:
            continue
        errors[criterion] = _relative_l2(np.asarray(left), np.asarray(right))
    return errors


def build_case(mesh_size: int) -> dict[str, Any]:
    mesh = MeshConfig(
        nx=mesh_size, ny=mesh_size, base_pixel_size_mm=SPACING_MM, scale_factor=1.0
    )
    span = mesh.physical_size_mm[0]
    nodes = np.linspace(0.0, span, mesh_size + 1)
    grid_x, grid_y = np.meshgrid(nodes, nodes, indexing="ij")
    # Non-affine: an affine field on a homogeneous material is an exact solution
    # for every formulation and measures nothing.
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


def solve(
    case: dict[str, Any],
    formulation: str,
    *,
    increments: int,
    repeats: int,
    library: str,
    **options: Any,
) -> tuple[FEMResult, dict[str, Any]]:
    residual_tolerance = options.pop("residual_tolerance", 1.0e-6)
    solver = SolverConfig(
        increments=increments,
        residual_tolerance=residual_tolerance,
        constitutive_backend="mfront-3d-condensed-plane-stress",
        mfront_library=library,
        mfront_behaviour_id="fcc_forest_rubin_srix",
        element_formulation=formulation,
        constitutive_options={
            "crystal_orientation": {
                "mode": "homogeneous",
                "euler_bunge_deg": list(ORIENTATION_BUNGE_DEG),
            }
        },
        **options,
    )
    configuration = CaseStudyConfig(
        mesh=case["mesh"], material=MaterialConfig(), solver=solver
    )
    elapsed: list[float] = []
    constitutive: list[float] = []
    result: FEMResult | None = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = run_case_study(
            configuration,
            displacement_x_mm=case["displacement_x_mm"],
            displacement_y_mm=case["displacement_y_mm"],
            yield_stress_mpa=case["yield_stress_mpa"],
            hardening_coefficient_mpa=case["hardening_coefficient_mpa"],
        )
        elapsed.append(time.perf_counter() - started)
        assert result.diagnostics is not None
        constitutive.append(result.diagnostics.constitutive_seconds)
    assert result is not None and result.diagnostics is not None
    diagnostics = result.diagnostics
    return result, {
        "elapsed_seconds": elapsed,
        "elapsed_median": statistics.median(elapsed),
        "constitutive_median": statistics.median(constitutive),
        "material_points": diagnostics.constitutive_material_point_count,
        "newton_iterations": diagnostics.total_newton_iterations,
        "cutbacks": diagnostics.cutbacks,
        "jacobian_correction": diagnostics.jacobian_correction,
        "broyden": diagnostics.jacobian_correction_diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=12)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    # Both runs stop on the same relative residual, so they land at two points
    # of one convergence ball. Tightening this is how one tells a path
    # difference -- which shrinks with the tolerance -- from a real drift of the
    # solution, which does not.
    parser.add_argument("--residual-tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--correction", choices=("broyden", "global_broyden"), default="broyden"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("validation/_generated/cps4r_as")
    )
    arguments = parser.parse_args()

    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        raise SystemExit("MFRONT_BEHAVIOUR_LIBRARY must be set")
    arguments.output.mkdir(parents=True, exist_ok=True)
    case = build_case(arguments.mesh)
    common = {
        "increments": arguments.increments,
        "repeats": arguments.repeats,
        "library": library,
        "residual_tolerance": arguments.residual_tolerance,
    }

    cps4, cps4_timing = solve(case, "cps4", **common)
    print(f"  CPS4                     iterations={cps4_timing['newton_iterations']}")
    baseline, baseline_timing = solve(case, "cps4r_as", **common)
    baseline_errors = _errors(baseline, cps4)
    print(
        f"  CPS4R-AS no correction   iterations={baseline_timing['newton_iterations']}"
        f"  E_u={baseline_errors['E_u'] * 100:.3f}%"
    )

    report: dict[str, Any] = {
        "mesh": arguments.mesh,
        "increments": arguments.increments,
        "repeats": arguments.repeats,
        "orientation_bunge_deg": list(ORIENTATION_BUNGE_DEG),
        "residual_tolerance": arguments.residual_tolerance,
        "cps4": cps4_timing,
        "baseline": {**baseline_timing, "errors_against_cps4": baseline_errors},
        "memories": {},
    }

    for memory in MEMORIES:
        try:
            candidate, timing = solve(
                case,
                "cps4r_as",
                jacobian_correction=arguments.correction,
                jacobian_correction_memory=memory,
                newton_line_search=arguments.correction == "global_broyden",
                **common,
            )
        except Exception as exc:  # a failure to converge IS a result here
            report["memories"][str(memory)] = {
                "converged": False,
                "failure": type(exc).__name__,
                "message": str(exc)[:300],
            }
            print(f"  m={memory:<2}  did not converge ({type(exc).__name__})")
            continue
        against_baseline = _errors(candidate, baseline)
        against_cps4 = _errors(candidate, cps4)
        drift = {
            name: abs(against_cps4[name] - baseline_errors[name]) * 100.0
            for name in against_cps4
            if name in baseline_errors
        }
        report["memories"][str(memory)] = {
            "converged": True,
            **timing,
            "errors_against_baseline": against_baseline,
            "errors_against_cps4": against_cps4,
            "cps4_error_drift_points": drift,
            "iteration_reduction": (
                1.0
                - timing["newton_iterations"] / baseline_timing["newton_iterations"]
            ),
            "constitutive_speedup": (
                cps4_timing["constitutive_median"] / timing["constitutive_median"]
            ),
            "total_speedup": cps4_timing["elapsed_median"] / timing["elapsed_median"],
            "additional_material_points": (
                timing["material_points"] - baseline_timing["material_points"]
            ),
        }
        entry = report["memories"][str(memory)]
        print(
            f"  m={memory:<2}  iterations={timing['newton_iterations']}"
            f"  (reduction {entry['iteration_reduction'] * 100:+.1f}%)"
            f"  E_u/baseline={against_baseline['E_u']:.2e}"
            f"  drift={max(drift.values()):.4f} pt"
            f"  cutbacks={timing['cutbacks']}"
        )

    suffix = (
        ""
        if arguments.residual_tolerance == 1.0e-6
        else f"_tol{arguments.residual_tolerance:g}"
    )
    destination = arguments.output / (
        f"{arguments.correction}_qualification{suffix}.json"
    )
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
