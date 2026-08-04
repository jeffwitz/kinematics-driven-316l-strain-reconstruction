"""Compare CPS4, CPS4R-elastic and CPS4R-AS on one case, reproducibly.

Written after a review pointed out that the first SRIX numbers were reported
without a generator: with 0.17 points of margin against a preregistered bound,
the exact command, the individual repetition times, the field digests and the
Gauss-point reduction all have to be on the record.

The reduction is the arithmetic mean over Gauss points, which is the identity
for the one-point formulations and averages four states for CPS4. It is
therefore part of the measured difference, not a neutral projection.

Usage:
    python scripts/compare_reduced_integration_formulations.py \\
        --case srix --mesh 12 --increments 8 --repeats 5 \\
        --output validation/_generated/cps4r_as
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from fem_inhouse.config import CaseStudyConfig, MaterialConfig, MeshConfig, SolverConfig
from fem_inhouse.results import FEMResult
from fem_inhouse.solver import run_case_study

SPACING_MM = 0.00184
ORIENTATION_BUNGE_DEG = (35.0, 20.0, 15.0)

#: Fields compared, and the criterion each one feeds.
COMPARED_FIELDS: dict[str, str] = {
    "displacement_mm": "E_u",
    "stress_mpa": "E_sigma",
    "reaction_force": "E_R",
    "equivalent_plastic_strain": "E_p",
    "cumulated_slip": "E_Gamma",
}


def _digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=float))
    digest = sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.linalg.norm(reference))
    difference = float(np.linalg.norm(candidate - reference))
    return difference / scale if scale > 0.0 else difference


def build_case(case: str, mesh_size: int) -> dict[str, Any]:
    """Geometry, boundary data and material maps of one registered case."""

    mesh = MeshConfig(
        nx=mesh_size, ny=mesh_size, base_pixel_size_mm=SPACING_MM, scale_factor=1.0
    )
    span = mesh.physical_size_mm[0]
    nodes = np.linspace(0.0, span, mesh_size + 1)
    grid_x, grid_y = np.meshgrid(nodes, nodes, indexing="ij")

    if case == "srix":
        # Non-affine on purpose: an affine field on a homogeneous material is an
        # exact equilibrium solution for every formulation and measures nothing.
        # A first run of this comparison returned errors of exactly zero for
        # that reason and was discarded.
        perturbation = 0.05 * 0.010 * span
        return {
            "mesh": mesh,
            "displacement_x_mm": -0.004 * grid_x
            + perturbation * np.sin(2.0 * np.pi * grid_y / span) * (grid_x / span),
            "displacement_y_mm": 0.010 * grid_y
            + perturbation * np.sin(2.0 * np.pi * grid_x / span) * (grid_y / span),
            "yield_stress_mpa": np.full((mesh_size, mesh_size), 250.0),
            "hardening_coefficient_mpa": np.full((mesh_size, mesh_size), 500.0),
            "backend": "mfront-3d-condensed-plane-stress",
            "behaviour": "fcc_forest_rubin_srix",
            "orientation": ORIENTATION_BUNGE_DEG,
        }
    if case == "heterogeneous_j2":
        generator = np.random.default_rng(20260804)
        yield_stress = 250.0 * (
            1.0 + 0.15 * generator.standard_normal((mesh_size, mesh_size))
        )
        band = slice(mesh_size // 2 - 1, mesh_size // 2 + 1)
        yield_stress[:, band] *= 0.75
        return {
            "mesh": mesh,
            "displacement_x_mm": -0.010 * grid_x,
            "displacement_y_mm": 0.020 * grid_y,
            "yield_stress_mpa": np.clip(yield_stress, 120.0, None),
            "hardening_coefficient_mpa": np.full((mesh_size, mesh_size), 500.0),
            "backend": "python",
            "behaviour": None,
            "orientation": None,
        }
    raise SystemExit(f"unknown case {case!r}; available: srix, heterogeneous_j2")


def solve(
    case: dict[str, Any],
    formulation: str,
    *,
    increments: int,
    repeats: int,
    library: str | None,
    **stabilisation: Any,
) -> tuple[FEMResult, dict[str, Any]]:
    options: dict[str, Any] = {}
    if case["orientation"] is not None:
        options["crystal_orientation"] = {
            "mode": "homogeneous",
            "euler_bunge_deg": list(case["orientation"]),
        }
    solver = SolverConfig(
        increments=increments,
        constitutive_backend=case["backend"],
        mfront_library=library or SolverConfig().mfront_library,
        mfront_behaviour_id=case["behaviour"],
        element_formulation=formulation,
        constitutive_options=options,
        **stabilisation,
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
    return result, {
        "elapsed_seconds": elapsed,
        "elapsed_median": statistics.median(elapsed),
        "constitutive_seconds": constitutive,
        "constitutive_median": statistics.median(constitutive),
        "material_points": result.diagnostics.constitutive_material_point_count,
        "gauss_points_per_element": result.diagnostics.gauss_points_per_element,
        "newton_iterations": result.diagnostics.total_newton_iterations,
        "cutbacks": result.diagnostics.cutbacks,
    }


def compare(candidate: FEMResult, reference: FEMResult) -> dict[str, Any]:
    errors: dict[str, Any] = {}
    for field, criterion in COMPARED_FIELDS.items():
        left = getattr(candidate, field, None)
        right = getattr(reference, field, None)
        if left is None or right is None:
            continue
        errors[criterion] = {
            "field": field,
            "relative_l2": _relative_l2(np.asarray(left), np.asarray(right)),
            "candidate_sha256": _digest(np.asarray(left)),
            "reference_sha256": _digest(np.asarray(right)),
        }
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="srix")
    parser.add_argument("--mesh", type=int, default=12)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("validation/_generated/cps4r_as")
    )
    arguments = parser.parse_args()

    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    case = build_case(arguments.case, arguments.mesh)
    if case["backend"] != "python" and library is None:
        raise SystemExit("MFRONT_BEHAVIOUR_LIBRARY must be set for this case")
    arguments.output.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, str, dict[str, Any]]] = [
        ("CPS4R-elastic", "cps4r", {}),
        ("CPS4R-AS-energy-asmd", "cps4r_as", {}),
        (
            "CPS4R-AS-current-asmd",
            "cps4r_as",
            {"stabilisation_strategy": "assumed_strain_current"},
        ),
    ]

    reference, reference_timing = solve(
        case,
        "cps4",
        increments=arguments.increments,
        repeats=arguments.repeats,
        library=library,
    )
    report: dict[str, Any] = {
        "case": arguments.case,
        "mesh": arguments.mesh,
        "increments": arguments.increments,
        "repeats": arguments.repeats,
        "orientation_bunge_deg": case["orientation"],
        "gauss_reduction": (
            "arithmetic mean over Gauss points; identity for the one-point "
            "formulations, an average of four states for CPS4, and therefore part "
            "of the measured difference rather than a neutral projection"
        ),
        "reference": {"formulation": "cps4", **reference_timing},
        "variants": {},
    }

    for label, formulation, stabilisation in variants:
        try:
            candidate, timing = solve(
                case,
                formulation,
                increments=arguments.increments,
                repeats=arguments.repeats,
                library=library,
                **stabilisation,
            )
        except Exception as exc:
            report["variants"][label] = {
                "converged": False,
                "failure": type(exc).__name__,
                "message": str(exc)[:200],
            }
            print(f"  {label:<24} did not converge ({type(exc).__name__})")
            continue
        errors = compare(candidate, reference)
        report["variants"][label] = {
            "converged": True,
            "formulation": formulation,
            "stabilisation": stabilisation,
            **timing,
            "total_speedup": reference_timing["elapsed_median"] / timing["elapsed_median"],
            "constitutive_speedup": (
                reference_timing["constitutive_median"] / timing["constitutive_median"]
            ),
            "errors": errors,
        }
        summary = "  ".join(
            f"{name}={item['relative_l2'] * 100:.3f}%" for name, item in errors.items()
        )
        print(
            f"  {label:<24} {summary}  total="
            f"{report['variants'][label]['total_speedup']:.2f}x"
        )

    destination = arguments.output / f"formulation_comparison_{arguments.case}.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
