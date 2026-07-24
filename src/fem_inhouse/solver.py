"""Validated public solver API for the supported 316L case study."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.config import CaseStudyConfig
from fem_inhouse.core import solver_legacy
from fem_inhouse.results import FEMResult, FrameResult

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

    return solver_legacy._SOLVER_NAME


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


def _convert_result(raw: dict[str, Any]) -> FEMResult:
    result = FEMResult(
        displacement_mm=np.asarray(raw["U"]),
        stress_mpa=np.asarray(raw["S"]),
        total_strain=np.asarray(raw["E"]),
        plastic_strain=np.asarray(raw["PE"]),
        equivalent_plastic_strain=np.asarray(raw["PEEQ"]),
        reaction_force=np.asarray(raw["RF"]),
        frames={
            float(fraction): _convert_frame(frame)
            for fraction, frame in raw.get("frames", {}).items()
        },
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

    raw = solver_legacy.run_fem(
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
        snapshot_fractions=snapshot_fractions,
        verbose=verbose,
    )
    return _convert_result(raw)
