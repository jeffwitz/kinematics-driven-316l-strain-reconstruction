"""Small reproducible examples and analytical validation cases."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.config import CaseStudyConfig, MaterialConfig, MeshConfig, SolverConfig
from fem_inhouse.postprocessing import von_mises_stress
from fem_inhouse.results import FEMResult
from fem_inhouse.solver import run_case_study


@dataclass(frozen=True, slots=True)
class ReducedBiaxialCase:
    """Inputs and analytical targets for homogeneous equal-biaxial tension."""

    config: CaseStudyConfig
    displacement_x_mm: NDArray
    displacement_y_mm: NDArray
    yield_stress_mpa: NDArray
    hardening_coefficient_mpa: NDArray
    target_stress_mpa: float
    target_equivalent_plastic_strain: float


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Measured errors for the reduced analytical case."""

    stress_mpa: float
    equivalent_plastic_strain: float
    relative_stress_error: float
    relative_plastic_strain_error: float
    relative_displacement_error: float
    relative_reaction_imbalance: float
    passed: bool


def reduced_biaxial_case(
    *,
    nx: int = 10,
    ny: int = 10,
    target_stress_mpa: float = 400.0,
    yield_stress_mpa: float = 250.0,
    hardening_coefficient_mpa: float = 500.0,
    constitutive_backend: Literal["python", "mfront"] = "mfront",
    mfront_library: str = "build/mfront/src/libBehaviour.so",
    mfront_threads: int = 1,
) -> ReducedBiaxialCase:
    """Construct the homogeneous closed-form verification used by the project."""

    if target_stress_mpa <= yield_stress_mpa:
        raise ValueError("target_stress_mpa must exceed yield_stress_mpa")
    material = MaterialConfig()
    mesh = MeshConfig(nx=nx, ny=ny, base_pixel_size_mm=0.001, scale_factor=1.0)
    solver = SolverConfig(
        increments=20,
        max_newton_iterations=15,
        residual_tolerance=1e-8,
        hardening_mode="ludwik",
        constitutive_backend=constitutive_backend,
        mfront_library=mfront_library,
        mfront_threads=mfront_threads,
    )
    config = CaseStudyConfig(mesh=mesh, material=material, solver=solver)
    equivalent_plastic_strain = (
        (target_stress_mpa - yield_stress_mpa) / hardening_coefficient_mpa
    ) ** (1.0 / material.hardening_exponent)
    elastic_strain = target_stress_mpa * (1.0 - material.poisson_ratio) / material.young_modulus_mpa
    total_normal_strain = elastic_strain + equivalent_plastic_strain / 2.0
    x = np.linspace(0.0, mesh.physical_size_mm[0], nx + 1)
    y = np.linspace(0.0, mesh.physical_size_mm[1], ny + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return ReducedBiaxialCase(
        config=config,
        displacement_x_mm=total_normal_strain * xx,
        displacement_y_mm=total_normal_strain * yy,
        yield_stress_mpa=np.full((nx, ny), yield_stress_mpa),
        hardening_coefficient_mpa=np.full((nx, ny), hardening_coefficient_mpa),
        target_stress_mpa=target_stress_mpa,
        target_equivalent_plastic_strain=equivalent_plastic_strain,
    )


def validate_reduced_case(case: ReducedBiaxialCase) -> tuple[FEMResult, ValidationReport]:
    """Run the reduced case and evaluate pre-declared numerical thresholds."""

    result = run_case_study(
        case.config,
        displacement_x_mm=case.displacement_x_mm,
        displacement_y_mm=case.displacement_y_mm,
        yield_stress_mpa=case.yield_stress_mpa,
        hardening_coefficient_mpa=case.hardening_coefficient_mpa,
    )
    equivalent_stress = von_mises_stress(
        result.stress_mpa[..., 0],
        result.stress_mpa[..., 1],
        result.stress_mpa[..., 2],
    )
    stress = float(equivalent_stress.mean())
    plastic_strain = float(result.equivalent_plastic_strain.mean())
    expected_displacement = np.stack(
        (case.displacement_x_mm, case.displacement_y_mm),
        axis=-1,
    )
    displacement_denominator = max(float(np.linalg.norm(expected_displacement)), 1e-30)
    displacement_error = float(
        np.linalg.norm(result.displacement_mm - expected_displacement) / displacement_denominator
    )
    stress_error = abs(stress - case.target_stress_mpa) / case.target_stress_mpa
    plastic_error = (
        abs(plastic_strain - case.target_equivalent_plastic_strain)
        / case.target_equivalent_plastic_strain
    )
    net_reaction = np.linalg.norm(result.reaction_force.sum(axis=(0, 1)))
    total_reaction = np.linalg.norm(result.reaction_force, axis=-1).sum()
    reaction_imbalance = float(net_reaction / max(float(total_reaction), 1e-30))
    report = ValidationReport(
        stress_mpa=stress,
        equivalent_plastic_strain=plastic_strain,
        relative_stress_error=stress_error,
        relative_plastic_strain_error=plastic_error,
        relative_displacement_error=displacement_error,
        relative_reaction_imbalance=reaction_imbalance,
        passed=stress_error < 0.005
        and plastic_error < 0.005
        and displacement_error < 1e-8
        and reaction_imbalance < 1e-10,
    )
    return result, report


def save_reduced_example(
    directory: str | Path,
    *,
    nx: int = 10,
    ny: int = 10,
    constitutive_backend: Literal["python", "mfront"] = "mfront",
    mfront_library: str = "build/mfront/src/libBehaviour.so",
    mfront_threads: int = 1,
) -> ValidationReport:
    """Execute and save a self-contained reduced example."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    case = reduced_biaxial_case(
        nx=nx,
        ny=ny,
        constitutive_backend=constitutive_backend,
        mfront_library=mfront_library,
        mfront_threads=mfront_threads,
    )
    result, report = validate_reduced_case(case)
    np.save(destination / "displacement_mm.npy", result.displacement_mm)
    np.save(destination / "stress_mpa.npy", result.stress_mpa)
    np.save(
        destination / "equivalent_plastic_strain.npy",
        result.equivalent_plastic_strain,
    )
    metadata = {
        "config": asdict(case.config),
        "diagnostics": asdict(result.diagnostics) if result.diagnostics else None,
        "validation": asdict(report),
    }
    (destination / "report.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
