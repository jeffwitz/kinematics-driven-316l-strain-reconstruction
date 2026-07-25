"""Validated public solver API for the supported 316L case study."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.config import CaseStudyConfig
from fem_inhouse.core import nonlinear
from fem_inhouse.core.tensor_reconstruction import reconstruct_python_plane_stress_state
from fem_inhouse.results import FEMResult, FrameResult, SolverDiagnostics

LOGGER = logging.getLogger(__name__)


def require_pypardiso() -> None:
    """Fail explicitly when the production sparse solver is unavailable."""

    try:
        import_module("pypardiso")
    except Exception as error:
        raise RuntimeError(
            "PyPardiso/MKL is required for case-study solves; install the project "
            "dependencies in the active environment."
        ) from error


def linear_solver_backend() -> str:
    """Return the backend selected by the numerical kernel."""

    return nonlinear._SOLVER_NAME


def _validated_field(
    values: ArrayLike,
    *,
    name: str,
    shape: tuple[int, ...],
    positive: bool = False,
    nonnegative: bool = False,
) -> NDArray:
    field = np.asarray(values, dtype=float)
    if field.shape != shape:
        raise ValueError(f"{name} has shape {field.shape}, expected {shape}")
    if not np.isfinite(field).all():
        raise ValueError(f"{name} contains non-finite values")
    if positive and np.any(field <= 0):
        raise ValueError(f"{name} must be strictly positive")
    if nonnegative and np.any(field < 0):
        raise ValueError(f"{name} must be nonnegative")
    return field


def _validated_snapshots(snapshots: tuple[float, ...]) -> tuple[float, ...]:
    values = tuple(float(value) for value in snapshots)
    if any(not np.isfinite(value) or not 0 < value <= 1 for value in values):
        raise ValueError("snapshot fractions must be finite and lie in (0, 1]")
    if len(set(values)) != len(values):
        raise ValueError("snapshot fractions must be unique")
    return tuple(sorted(values))


def _convert_frame(frame: dict[str, Any]) -> FrameResult:
    return FrameResult(
        stress_mpa=np.asarray(frame["S"]),
        total_strain=np.asarray(frame["E"]),
        equivalent_plastic_strain=np.asarray(frame["PEEQ"]),
        displacement_mm=np.asarray(frame["U"]),
    )


def _convert_result(raw: dict[str, Any], *, poisson_ratio: float) -> FEMResult:
    diagnostics = raw["diagnostics"]
    displacement = np.asarray(raw["U"])
    stress = np.asarray(raw["S"])
    total_strain = np.asarray(raw["E"])
    plastic_strain = np.asarray(raw["PE"])
    equivalent_plastic_strain = np.asarray(raw["PEEQ"])
    reaction_force = np.asarray(raw["RF"])
    historical_fields = (
        displacement,
        stress,
        total_strain,
        plastic_strain,
        equivalent_plastic_strain,
        reaction_force,
    )
    if not all(np.isfinite(field).all() for field in historical_fields):
        raise RuntimeError("solver returned non-finite final fields")
    reconstructed_keys = (
        "S_3D",
        "E_3D",
        "EE_3D",
        "PE_3D",
        "S33_RESIDUAL_MPA",
    )
    present = tuple(key in raw for key in reconstructed_keys)
    if any(present) and not all(present):
        missing = [
            key for key, available in zip(reconstructed_keys, present, strict=True) if not available
        ]
        raise RuntimeError(f"solver returned an incomplete reconstructed tensor state: {missing}")
    if all(present):
        stress_tensor = np.asarray(raw["S_3D"])
        total_strain_tensor = np.asarray(raw["E_3D"])
        elastic_strain_tensor = np.asarray(raw["EE_3D"])
        plastic_strain_tensor = np.asarray(raw["PE_3D"])
        plane_stress_residual = np.asarray(raw["S33_RESIDUAL_MPA"])
        if "PLANE_STRESS_RESIDUAL_MPA" in raw:
            plane_stress_residual_vector = np.asarray(raw["PLANE_STRESS_RESIDUAL_MPA"])
        else:
            plane_stress_residual_vector = np.zeros(
                (*plane_stress_residual.shape, 3),
                dtype=float,
            )
            plane_stress_residual_vector[..., 0] = plane_stress_residual
    else:
        legacy_state = reconstruct_python_plane_stress_state(
            total_strain,
            plastic_strain,
            stress,
            poisson_ratio,
        )
        stress_tensor = legacy_state.stress_tensor_mpa
        total_strain_tensor = legacy_state.total_strain_tensor
        elastic_strain_tensor = legacy_state.elastic_strain_tensor
        plastic_strain_tensor = legacy_state.plastic_strain_tensor
        plane_stress_residual = legacy_state.plane_stress_residual_mpa
        plane_stress_residual_vector = legacy_state.plane_stress_residual_vector_mpa
    nonlocal_keys = (
        "PEEQ_NONLOCAL",
        "PEEQ_MISMATCH",
        "NONLOCAL_HARDENING_MPA",
        "YIELD_SURFACE_RADIUS_MPA",
        "NONLOCAL_RESIDUAL",
    )
    nonlocal_present = tuple(key in raw for key in nonlocal_keys)
    if any(nonlocal_present) and not all(nonlocal_present):
        missing = [
            key
            for key, available in zip(nonlocal_keys, nonlocal_present, strict=True)
            if not available
        ]
        raise RuntimeError(f"solver returned an incomplete nonlocal state: {missing}")
    nonlocal_arrays = (
        tuple(np.asarray(raw[key]) for key in nonlocal_keys)
        if all(nonlocal_present)
        else (None, None, None, None, None)
    )
    result = FEMResult(
        displacement_mm=displacement,
        stress_mpa=stress,
        total_strain=total_strain,
        plastic_strain=plastic_strain,
        equivalent_plastic_strain=equivalent_plastic_strain,
        reaction_force=reaction_force,
        stress_tensor_mpa=stress_tensor,
        total_strain_tensor=total_strain_tensor,
        elastic_strain_tensor=elastic_strain_tensor,
        plastic_strain_tensor=plastic_strain_tensor,
        plane_stress_residual_mpa=plane_stress_residual,
        plane_stress_residual_vector_mpa=plane_stress_residual_vector,
        nonlocal_equivalent_plastic_strain=nonlocal_arrays[0],
        equivalent_plastic_strain_mismatch=nonlocal_arrays[1],
        nonlocal_hardening_mpa=nonlocal_arrays[2],
        yield_surface_radius_mpa=nonlocal_arrays[3],
        nonlocal_residual=nonlocal_arrays[4],
        frames={
            float(fraction): _convert_frame(frame)
            for fraction, frame in raw.get("frames", {}).items()
        },
        diagnostics=SolverDiagnostics(**diagnostics),
    )
    if not all(np.isfinite(field).all() for field in result.arrays()):
        raise RuntimeError("solver returned non-finite final fields")
    return result


def run_case_study(
    config: CaseStudyConfig,
    *,
    displacement_x_mm: ArrayLike,
    displacement_y_mm: ArrayLike,
    yield_stress_mpa: ArrayLike,
    hardening_coefficient_mpa: ArrayLike,
    snapshots: tuple[float, ...] = (),
    verbose: bool = False,
) -> FEMResult:
    """Solve one structured partition using prescribed boundary displacements.

    Array axes follow ``docs/scientific_contract.md``. Displacement fields are
    nodal with shape ``(nx + 1, ny + 1)``; both material maps are element fields
    with shape ``(nx, ny)``.
    """

    mesh = config.mesh
    nodal_shape = (mesh.nx + 1, mesh.ny + 1)
    element_shape = (mesh.nx, mesh.ny)
    displacement_x = _validated_field(
        displacement_x_mm,
        name="displacement_x_mm",
        shape=nodal_shape,
    )
    displacement_y = _validated_field(
        displacement_y_mm,
        name="displacement_y_mm",
        shape=nodal_shape,
    )
    yield_map = _validated_field(
        yield_stress_mpa,
        name="yield_stress_mpa",
        shape=element_shape,
        positive=True,
    )
    hardening_map = _validated_field(
        hardening_coefficient_mpa,
        name="hardening_coefficient_mpa",
        shape=element_shape,
        nonnegative=True,
    )
    snapshot_fractions = _validated_snapshots(snapshots)

    if config.solver.require_pypardiso:
        require_pypardiso()
    LOGGER.info(
        "Starting %sx%s element solve with %s",
        mesh.nx,
        mesh.ny,
        linear_solver_backend(),
    )

    raw = nonlinear.run_fem(
        displacement_x,
        displacement_y,
        yield_map,
        hardening_map,
        config.material.hardening_exponent,
        mesh.nx * mesh.base_pixel_size_mm,
        mesh.ny * mesh.base_pixel_size_mm,
        mesh.base_pixel_size_mm,
        mesh.scale_factor,
        E_mod=config.material.young_modulus_mpa,
        nu=config.material.poisson_ratio,
        N_inc=config.solver.increments,
        max_nr=config.solver.max_newton_iterations,
        nr_tol=config.solver.residual_tolerance,
        hardening=config.solver.hardening_mode,
        ep_table_max=config.material.plastic_strain_max,
        n_table=config.material.plastic_table_points,
        first_positive_plastic_strain=config.material.first_positive_plastic_strain,
        minimum_step_divisor=config.solver.minimum_step_divisor,
        constitutive_backend=config.solver.constitutive_backend,
        mfront_library=config.solver.mfront_library,
        mfront_threads=config.solver.mfront_threads,
        local_plane_stress_tolerance_mpa=config.solver.local_plane_stress_tolerance_mpa,
        local_plane_stress_relative_tolerance=(config.solver.local_plane_stress_relative_tolerance),
        maximum_local_plane_stress_iterations=(config.solver.maximum_local_plane_stress_iterations),
        maximum_cbb_condition_number=config.solver.maximum_cbb_condition_number,
        nonlocal_plasticity_enabled=config.nonlocal_plasticity.enabled,
        nonlocal_length_scale_mm=config.nonlocal_plasticity.length_scale_mm,
        nonlocal_coupling_modulus_mpa=(
            config.nonlocal_plasticity.coupling_modulus_mpa
        ),
        nonlocal_relaxation=config.nonlocal_plasticity.relaxation,
        nonlocal_relative_tolerance=(
            config.nonlocal_plasticity.relative_tolerance
        ),
        nonlocal_maximum_iterations=config.nonlocal_plasticity.maximum_iterations,
        nonlocal_maximum_helmholtz_residual=(
            config.nonlocal_plasticity.maximum_helmholtz_residual
        ),
        snapshot_fractions=snapshot_fractions,
        verbose=verbose,
    )
    return _convert_result(raw, poisson_ratio=config.material.poisson_ratio)
