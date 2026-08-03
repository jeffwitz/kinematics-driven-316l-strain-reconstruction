"""CPS4R against CPS4, on the cases registered in the preregistration.

Runs `validation/cps4r_qualification_preregistration.md`. Everything here is
small and synthetic; no experimental ROI is touched and no campaign is replayed.

Usage:
    python scripts/qualify_reduced_integration.py --output validation/_generated
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from fem_inhouse.config import CaseStudyConfig, MaterialConfig, MeshConfig, SolverConfig
from fem_inhouse.core.crystal_orientation import rotation_from_euler_bunge_deg
from fem_inhouse.results import FEMResult
from fem_inhouse.solver import run_case_study

#: Robust per-state DIC boundary noise, from
#: `validation/dic_boundary_loading_subspace_p0043_results.md`.
DIC_NOISE_MM = 9.40e-5

#: Registered acceptance bounds. Derived in the preregistration; not tunable
#: here, and deliberately module-level constants so a reader can see that the
#: script did not choose them.
PEEQ_RELATIVE_BOUND = 5.0e-3
DISPLACEMENT_RELATIVE_BOUND = 1.0e-3
HOURGLASS_RATIO_BOUND = 1.0e-2
#: Below this the case does not excite the stabilised modes (falsifier F1).
CASE_ADMISSIBILITY_RATIO = 1.0e-8

BETAS = (0.1, 0.25, 0.5, 1.0)
SRIX = "fcc_forest_rubin_srix"


@dataclass(frozen=True, slots=True)
class Case:
    """A solvable case, identical apart from the element formulation."""

    name: str
    config: CaseStudyConfig
    displacement_x_mm: np.ndarray
    displacement_y_mm: np.ndarray
    yield_stress_mpa: np.ndarray
    hardening_coefficient_mpa: np.ndarray

    def solve(
        self,
        formulation: str,
        *,
        hourglass_scale: float = 1.0,
        repeats: int = 1,
    ) -> FEMResult:
        """Solve, and with `repeats > 1` replace the timings by their median.

        The solve is deterministic, so repeating it changes nothing but the
        clock. A single elapsed time on a shared machine is not a measurement:
        the first sweep showed a 57 percent spread between two configurations
        that performed an identical number of Newton iterations.
        """

        solver = replace(
            self.config.solver,
            element_formulation=formulation,
            hourglass_scale=hourglass_scale,
        )
        config = replace(self.config, solver=solver)
        results = [
            run_case_study(
                config,
                displacement_x_mm=self.displacement_x_mm,
                displacement_y_mm=self.displacement_y_mm,
                yield_stress_mpa=self.yield_stress_mpa,
                hardening_coefficient_mpa=self.hardening_coefficient_mpa,
            )
            for _ in range(max(repeats, 1))
        ]
        result = results[0]
        if len(results) > 1 and result.diagnostics is not None:
            diagnostics = [r.diagnostics for r in results]
            assert all(d is not None for d in diagnostics)
            result = replace(
                result,
                diagnostics=replace(
                    result.diagnostics,
                    elapsed_seconds=float(
                        np.median([d.elapsed_seconds for d in diagnostics])  # type: ignore[union-attr]
                    ),
                    constitutive_seconds=float(
                        np.median([d.constitutive_seconds for d in diagnostics])  # type: ignore[union-attr]
                    ),
                ),
            )
        return result


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Relative L2 difference, or the absolute one when the reference vanishes."""

    difference = float(np.linalg.norm(candidate - reference))
    scale = float(np.linalg.norm(reference))
    return difference / scale if scale > 0.0 else difference


def heterogeneous_j2_case(*, library: str, size: int = 32) -> Case:
    """C1: pixel-wise heterogeneous J2, affine tensile boundary.

    The boundary is affine on purpose. In this project the boundary is measured
    data and is smooth; what makes the interior solution non-affine, and what
    therefore excites the hourglass modes, is the pixel-wise material map. That
    is the configuration a real campaign runs, so it is the one worth
    qualifying -- an artificial short-wavelength boundary would excite the modes
    more strongly while representing nothing.
    """

    mesh = MeshConfig(nx=size, ny=size, base_pixel_size_mm=0.00184, scale_factor=1.0)
    material = MaterialConfig()
    solver = SolverConfig(
        increments=20,
        max_newton_iterations=15,
        residual_tolerance=1e-8,
        constitutive_backend="mfront",
        mfront_library=library,
        mfront_threads=1,
    )
    span_x, span_y = mesh.physical_size_mm
    nodes_x = np.linspace(0.0, span_x, size + 1)
    nodes_y = np.linspace(0.0, span_y, size + 1)
    grid_x, grid_y = np.meshgrid(nodes_x, nodes_y, indexing="ij")

    # A soft horizontal band plus deterministic pixel scatter: the band sets a
    # localisation site, the scatter gives the element-scale contrast that the
    # DIC-driven maps really have.
    generator = np.random.default_rng(20260803)
    yield_stress = 250.0 * (1.0 + 0.15 * generator.standard_normal((size, size)))
    band = slice(size // 2 - 1, size // 2 + 1)
    yield_stress[:, band] *= 0.75
    yield_stress = np.clip(yield_stress, 120.0, None)

    return Case(
        name="heterogeneous_j2",
        config=CaseStudyConfig(mesh=mesh, material=material, solver=solver),
        # 2 percent axial extension, with the transverse contraction a plastic
        # incompressible response would give, so the case yields well past first
        # yield without needing a large number of increments.
        displacement_x_mm=-0.010 * grid_x,
        displacement_y_mm=0.020 * grid_y,
        yield_stress_mpa=yield_stress,
        hardening_coefficient_mpa=np.full((size, size), 500.0),
    )


def tilted_crystal_case(*, library: str, size: int = 8) -> Case:
    """C2: SRIX, one homogeneous orientation away from the axes.

    Tilted so the condensed cubic operator is genuinely anisotropic, with
    extension--shear coupling an isotropic reference could not produce.
    """

    mesh = MeshConfig(nx=size, ny=size, base_pixel_size_mm=0.00184, scale_factor=1.0)
    material = MaterialConfig()
    orientation = rotation_from_euler_bunge_deg(35.0, 20.0, 15.0)
    solver = SolverConfig(
        increments=10,
        max_newton_iterations=15,
        residual_tolerance=1e-8,
        constitutive_backend="mfront-3d-condensed-plane-stress",
        mfront_library=library,
        mfront_behaviour_id=SRIX,
        mfront_threads=1,
        constitutive_options={
            "crystal_orientation": {
                "mode": "homogeneous",
                "matrix": orientation.tolist(),
            }
        },
    )
    span_x, span_y = mesh.physical_size_mm
    nodes_x = np.linspace(0.0, span_x, size + 1)
    nodes_y = np.linspace(0.0, span_y, size + 1)
    grid_x, grid_y = np.meshgrid(nodes_x, nodes_y, indexing="ij")

    # Affine here would be the trap the preregistration names: an affine field
    # on every edge is an exact equilibrium solution for any homogeneous
    # material, so the two formulations could not differ. A perturbation is
    # required precisely because the material is homogeneous in this case, and
    # it is kept to five percent of the axial displacement: large enough to
    # excite the modes, small enough that both formulations follow the same
    # equilibrium path rather than diverging through cutbacks.
    perturbation = 0.05 * 0.008 * span_y
    return Case(
        name="tilted_crystal_srix",
        config=CaseStudyConfig(mesh=mesh, material=material, solver=solver),
        displacement_x_mm=(
            -0.004 * grid_x
            + perturbation * np.sin(2.0 * np.pi * grid_y / span_y) * (grid_x / span_x)
        ),
        displacement_y_mm=(
            0.008 * grid_y
            + perturbation * np.sin(2.0 * np.pi * grid_x / span_x) * (grid_y / span_y)
        ),
        yield_stress_mpa=np.full((size, size), 250.0),
        hardening_coefficient_mpa=np.full((size, size), 500.0),
    )


def compare(reference: FEMResult, candidate: FEMResult) -> dict[str, Any]:
    """Registered metrics of a CPS4R result against its CPS4 reference."""

    assert reference.diagnostics is not None
    assert candidate.diagnostics is not None
    displacement_difference = candidate.displacement_mm - reference.displacement_mm
    return {
        "displacement_relative_l2": _relative_l2(
            candidate.displacement_mm, reference.displacement_mm
        ),
        "displacement_rms_mm": float(np.sqrt(np.mean(displacement_difference**2))),
        "displacement_rms_over_dic_noise": float(
            np.sqrt(np.mean(displacement_difference**2)) / DIC_NOISE_MM
        ),
        "peeq_relative_l2": _relative_l2(
            candidate.equivalent_plastic_strain, reference.equivalent_plastic_strain
        ),
        "peeq_max_absolute_difference": float(
            np.abs(
                candidate.equivalent_plastic_strain
                - reference.equivalent_plastic_strain
            ).max()
        ),
        "reference_peeq_max": float(np.abs(reference.equivalent_plastic_strain).max()),
        # Crystal laws leave PEEQ at zero, so a PEEQ comparison there would
        # report a perfect score against an empty field. The stress comparison
        # is the one that carries the accuracy check in that case.
        "stress_relative_l2": _relative_l2(candidate.stress_mpa, reference.stress_mpa),
        "reference_stress_norm_mpa": float(np.linalg.norm(reference.stress_mpa)),
        "reaction_relative_l2": _relative_l2(
            candidate.reaction_force, reference.reaction_force
        ),
        "hourglass_energy_ratio": candidate.diagnostics.hourglass_energy_ratio,
        "hourglass_energy": candidate.diagnostics.hourglass_energy,
        "internal_work": candidate.diagnostics.internal_work,
        "cutbacks": candidate.diagnostics.cutbacks,
        "reference_cutbacks": reference.diagnostics.cutbacks,
        "newton_iterations": candidate.diagnostics.total_newton_iterations,
        "reference_newton_iterations": reference.diagnostics.total_newton_iterations,
        "elapsed_seconds": candidate.diagnostics.elapsed_seconds,
        "reference_elapsed_seconds": reference.diagnostics.elapsed_seconds,
        "constitutive_seconds": candidate.diagnostics.constitutive_seconds,
        "reference_constitutive_seconds": reference.diagnostics.constitutive_seconds,
        "material_points": candidate.diagnostics.constitutive_material_point_count,
        "reference_material_points": (
            reference.diagnostics.constitutive_material_point_count
        ),
    }


def verdict(metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the registered criteria. No judgement is exercised here."""

    # A field that is identically zero in the reference cannot certify anything.
    # Crystal laws do not populate PEEQ, so A1 falls back to the stress field
    # rather than reporting a perfect score against nothing.
    if metrics["reference_peeq_max"] > 0.0:
        a1_field = "peeq"
        a1_value = metrics["peeq_relative_l2"]
    elif metrics["reference_stress_norm_mpa"] > 0.0:
        a1_field = "stress"
        a1_value = metrics["stress_relative_l2"]
    else:
        raise ValueError("reference has neither plastic strain nor stress to compare")
    a1 = a1_value <= PEEQ_RELATIVE_BOUND
    a2 = metrics["displacement_relative_l2"] <= DISPLACEMENT_RELATIVE_BOUND
    a4 = metrics["hourglass_energy_ratio"] < HOURGLASS_RATIO_BOUND
    a5 = metrics["cutbacks"] <= metrics["reference_cutbacks"]
    return {
        "A1_constitutive": a1,
        "A1_field": a1_field,
        "A1_value": a1_value,
        "A2_displacement": a2,
        "A4_hourglass_ratio": a4,
        "A5_no_new_cutback": a5,
        "recommendable": bool(a1 and a2 and a4 and a5),
        # F3: the ratio is a conservative gate only if passing it implies
        # passing the accuracy bound. A pass on A4 with a failure on A1 refutes
        # the diagnostic, and that is the point of the campaign.
        "F3_diagnostic_conservative": bool((not a4) or a1),
    }


def run_beta_sweep(
    case: Case, *, repeats: int = 1
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    reference = case.solve("cps4", repeats=repeats)
    assert reference.diagnostics is not None
    sweep: dict[str, Any] = {
        "case": case.name,
        "reference": {
            "formulation": "cps4",
            "material_points": reference.diagnostics.constitutive_material_point_count,
            "elapsed_seconds": reference.diagnostics.elapsed_seconds,
            "constitutive_seconds": reference.diagnostics.constitutive_seconds,
            "cutbacks": reference.diagnostics.cutbacks,
            "peeq_max": float(np.abs(reference.equivalent_plastic_strain).max()),
        },
        "beta": {},
    }
    fields: dict[str, np.ndarray] = {
        "reference_peeq": reference.equivalent_plastic_strain,
    }
    for beta in BETAS:
        candidate = case.solve("cps4r", hourglass_scale=beta, repeats=repeats)
        metrics = compare(reference, candidate)
        sweep["beta"][f"{beta}"] = {"metrics": metrics, "verdict": verdict(metrics)}
        if candidate.hourglass_energy_by_element is not None:
            fields[f"hourglass_energy_beta_{beta}"] = (
                candidate.hourglass_energy_by_element
            )
        if beta == 1.0:
            fields["peeq_beta_1"] = candidate.equivalent_plastic_strain
    admissible = (
        sweep["beta"]["1.0"]["metrics"]["hourglass_energy_ratio"]
        >= CASE_ADMISSIBILITY_RATIO
    )
    sweep["F1_case_excites_the_modes"] = bool(admissible)
    return sweep, fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("validation/_generated"))
    parser.add_argument("--mesh", type=int, default=32)
    parser.add_argument("--crystal-mesh", type=int, default=8)
    parser.add_argument("--skip-crystal", action="store_true")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="solve each configuration this many times and take the median timing",
    )
    args = parser.parse_args()

    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        raise SystemExit("MFRONT_BEHAVIOUR_LIBRARY must be set")

    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "preregistration": "validation/cps4r_qualification_preregistration.md",
        "dic_noise_mm": DIC_NOISE_MM,
        "bounds": {
            "peeq_relative": PEEQ_RELATIVE_BOUND,
            "displacement_relative": DISPLACEMENT_RELATIVE_BOUND,
            "hourglass_ratio": HOURGLASS_RATIO_BOUND,
        },
        "timing_repeats": args.repeats,
        "cases": {},
    }

    j2_case = heterogeneous_j2_case(library=library, size=args.mesh)
    sweep, fields = run_beta_sweep(j2_case, repeats=args.repeats)
    report["cases"]["C1_heterogeneous_j2"] = sweep
    for name, field in fields.items():
        np.save(args.output / f"c1_{name}.npy", field)

    if not args.skip_crystal:
        crystal = tilted_crystal_case(library=library, size=args.crystal_mesh)
        reference = crystal.solve("cps4", repeats=args.repeats)
        candidate = crystal.solve("cps4r", repeats=args.repeats)
        metrics = compare(reference, candidate)
        report["cases"]["C2_tilted_crystal_srix"] = {
            "case": crystal.name,
            "metrics": metrics,
            "verdict": verdict(metrics),
        }
        if candidate.hourglass_energy_by_element is not None:
            np.save(
                args.output / "c2_hourglass_energy.npy",
                candidate.hourglass_energy_by_element,
            )
        np.save(args.output / "c2_reference_peeq.npy", reference.equivalent_plastic_strain)

    destination = args.output / "cps4r_qualification.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
