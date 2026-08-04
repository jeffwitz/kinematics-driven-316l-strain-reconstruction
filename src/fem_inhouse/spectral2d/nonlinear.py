"""Fixed-point equilibrium solver for full-Dirichlet plane-stress fields."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike

from fem_inhouse.core.plane_stress_material import (
    ConstitutiveIntegrationError,
    ConstitutiveTrial,
    PlaneStressMaterialBatch,
)
from fem_inhouse.spectral2d.anderson import DisplacementAndersonAccelerator
from fem_inhouse.spectral2d.boundary import HarmonicDirichletExtension2D
from fem_inhouse.spectral2d.config import Spectral2DConfig
from fem_inhouse.spectral2d.diagnostics import Spectral2DDiagnostics
from fem_inhouse.spectral2d.green import (
    B0Green2D,
    TwoMuGreen2D,
    project_isotropic_plane_stress_tangent,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import QUAD1_2D, TRI2_2D, DiscreteKinematics2D
from fem_inhouse.spectral2d.result import Spectral2DResult
from fem_inhouse.spectral2d.transforms import FullDirichletDSTIPlan2D


class SpectralIncrementConvergenceError(RuntimeError):
    """The fixed-point iteration did not converge for an increment."""


def _evaluate_material(material, strain, time_increment):
    """Evaluate a batch with its canonical ``(point_count, 3)`` input shape."""
    flat_strain = np.asarray(strain, dtype=np.float64).reshape(-1, 3)
    trial = material.evaluate_in_plane(
        flat_strain, time_increment=time_increment, consistent_tangent=True
    )
    stress = np.asarray(trial.stress_in_plane_mpa, dtype=np.float64).reshape(strain.shape)
    return trial, stress


def _reshape_constitutive_trial(
    trial: ConstitutiveTrial, leading_shape: tuple[int, ...]
) -> ConstitutiveTrial:
    point_count = int(np.prod(leading_shape))

    def reshape_field(values, trailing_shape):
        array = np.asarray(values)
        if array.size == 0:
            return array
        return array.reshape((point_count, *trailing_shape)).reshape(
            (*leading_shape, *trailing_shape)
        )

    observables = {
        name: reshape_field(values, np.asarray(values).shape[1:])
        for name, values in trial.observables.items()
    }
    return replace(
        trial,
        stress_in_plane_mpa=reshape_field(trial.stress_in_plane_mpa, (3,)),
        tangent_in_plane_mpa=(
            None
            if trial.tangent_in_plane_mpa is None
            else reshape_field(trial.tangent_in_plane_mpa, (3, 3))
        ),
        full_stress_tensor_mpa=reshape_field(trial.full_stress_tensor_mpa, (3, 3)),
        full_strain_tensor=reshape_field(trial.full_strain_tensor, (3, 3)),
        elastic_strain_tensor=reshape_field(trial.elastic_strain_tensor, (3, 3)),
        plastic_strain_tensor=reshape_field(trial.plastic_strain_tensor, (3, 3)),
        plane_stress_residual_mpa=reshape_field(trial.plane_stress_residual_mpa, (3,)),
        observables=observables,
    )


def _equilibrium_metrics(stress, nodal_force, grid, points_per_pixel):
    """Return dimensionless equilibrium and dimensional force norms."""
    cell_area = grid.spacing_x * grid.spacing_y
    point_area = cell_area / points_per_pixel
    stress_norm = np.sqrt(point_area * float(np.sum(np.asarray(stress) ** 2)))
    interior_force_norm = float(np.linalg.norm(nodal_force[1:-1, 1:-1]))
    divergence_norm = interior_force_norm / np.sqrt(cell_area)
    dimensionless = np.sqrt(cell_area) * divergence_norm / max(stress_norm, 1.0e-30)
    return dimensionless, divergence_norm, interior_force_norm


def _boundary_reactions(nodal_force):
    """Return reactions ``B^T sigma`` and suppress interior entries."""
    reactions = -np.asarray(nodal_force, dtype=np.float64).copy()
    reactions[1:-1, 1:-1] = 0.0
    return reactions


def solve_dirichlet_plane_stress_spectral(
    *,
    grid: StructuredGrid2D,
    material: PlaneStressMaterialBatch,
    boundary_displacement_history: ArrayLike,
    config: Spectral2DConfig,
) -> Spectral2DResult:
    """Solve a sequence of full-Dirichlet displacement increments."""
    started = perf_counter()
    history = np.asarray(boundary_displacement_history, dtype=np.float64)
    expected = (history.shape[0], *grid.node_shape, 2)
    if history.ndim != 4 or history.shape != expected:
        raise ValueError(f"expected boundary history shape {expected}, got {history.shape}")
    if not np.allclose(history[0], 0.0):
        raise ValueError("the first boundary displacement must be zero")

    operator: DiscreteKinematics2D = (
        QUAD1_2D(grid) if config.spatial_scheme == "quad1" else TRI2_2D(grid)
    )
    if material.point_count != operator.material_point_count:
        raise ValueError("material point count does not match the selected spatial scheme")

    plan = FullDirichletDSTIPlan2D(grid)
    symbols = operator.reference_operator_symbols(plan)
    reference_trial = material.evaluate_in_plane(
        np.zeros((material.point_count, 3), dtype=np.float64),
        time_increment=1.0,
        consistent_tangent=True,
    )
    material.revert()
    if reference_trial.tangent_in_plane_mpa is None:
        raise ValueError("the material must provide an elastic tangent for the Green operator")
    tangent = np.asarray(reference_trial.tangent_in_plane_mpa, dtype=np.float64)
    tangent = 0.5 * (tangent + np.swapaxes(tangent, -1, -2))
    lambda_0, mu_0, _ = project_isotropic_plane_stress_tangent(
        tangent.mean(axis=0), tolerance=config.reference_projection_tolerance
    )
    green: B0Green2D | TwoMuGreen2D
    if config.green_operator == "two_mu":
        green = TwoMuGreen2D(
            symbols,
            mu_0=mu_0,
            symbol_null_tolerance=config.symbol_null_tolerance,
        )
    elif config.green_operator == "b0":
        green = B0Green2D(
            symbols,
            lambda_0=lambda_0,
            mu_0=mu_0,
            symbol_null_tolerance=config.symbol_null_tolerance,
        )
    else:
        raise NotImplementedError("the compatible C0 Green operator is not implemented")
    extension = HarmonicDirichletExtension2D()
    fluctuation = np.zeros((*grid.node_shape, 2), dtype=np.float64)
    final_trial: ConstitutiveTrial | None = None
    final_applied = np.zeros_like(history[0])
    final_strain = np.empty((0,))
    final_stress = np.empty((0,))
    relative_history: list[float] = []
    dimensionless_history: list[float] = []
    absolute_history: list[float] = []
    iterations_per_increment: list[int] = []
    anderson = DisplacementAndersonAccelerator(
        config.anderson_memory, config.anderson_regularization
    )
    minimum_relaxation = 1.0
    initial_increment = 1.0 / (history.shape[0] - 1)
    load_points = [
        (history[index].copy(), initial_increment)
        for index in range(1, history.shape[0])
    ]
    cutbacks = 0
    increment_cutbacks = 0
    load_index = 0
    previous_boundary = history[0].copy()

    while load_index < len(load_points):
        boundary_target, time_increment = load_points[load_index]
        increment = load_index + 1
        applied = extension.extend(boundary_target, grid)
        fluctuation_start = fluctuation.copy()
        anderson.reset()
        accepted = False
        failed = False
        for iteration in range(config.maximum_fixed_point_iterations):
            total_u = applied + fluctuation
            strain = operator.strain(total_u)
            trial, stress = _evaluate_material(material, strain, time_increment)
            residual = operator.divergence(stress)
            relative, divergence_norm, _residual_norm = _equilibrium_metrics(
                stress, residual, grid, operator.points_per_pixel
            )
            relative_history.append(relative)
            dimensionless_history.append(relative)
            absolute_history.append(divergence_norm)
            if relative <= config.relative_equilibrium_tolerance:
                final_trial = _reshape_constitutive_trial(
                    material.complete_trial(trial), strain.shape[:-1]
                )
                material.commit()
                final_applied = applied.copy()
                final_strain = np.asarray(strain).copy()
                final_stress = stress.copy()
                accepted = True
                iterations_per_increment.append(iteration + 1)
                break

            transformed_fluctuation = plan.forward_displacement(fluctuation[1:-1, 1:-1])
            reference_force = green.reference_force(transformed_fluctuation)
            transformed_residual = plan.forward_displacement(residual[1:-1, 1:-1])
            polarization = reference_force - transformed_residual
            image_interior = plan.inverse_displacement(green.apply(polarization))
            image = plan.embed_interior(image_interior)
            fixed_residual = image - fluctuation
            candidate = image
            if (
                config.anderson_enabled
                and iteration + 1 >= config.anderson_start_iteration
                and config.anderson_target == "polarization"
            ):
                probe_strain = operator.strain(applied + image)
                try:
                    _, probe_stress = _evaluate_material(material, probe_strain, time_increment)
                    probe_residual = operator.divergence(probe_stress)
                    probe_polarization = green.reference_force(
                        plan.forward_displacement(image[1:-1, 1:-1])
                    ) - plan.forward_displacement(probe_residual[1:-1, 1:-1])
                    accelerated_polarization = anderson.propose(
                        polarization,
                        probe_polarization,
                        probe_polarization - polarization,
                    )
                    image = plan.embed_interior(
                        plan.inverse_displacement(green.apply(accelerated_polarization))
                    )
                    fixed_residual = image - fluctuation
                    candidate = image
                except ConstitutiveIntegrationError:
                    material.revert()
            elif (
                config.anderson_enabled
                and config.anderson_target == "displacement"
                and iteration + 1 >= config.anderson_start_iteration
            ):
                candidate = anderson.propose(fluctuation, image, fixed_residual)
            relaxation = 1.0
            target = candidate
            while True:
                candidate_fluctuation = fluctuation + relaxation * (target - fluctuation)
                candidate_strain = operator.strain(applied + candidate_fluctuation)
                try:
                    _candidate_trial, candidate_stress = _evaluate_material(
                        material, candidate_strain, time_increment
                    )
                    candidate_residual = operator.divergence(candidate_stress)
                    candidate_norm = _equilibrium_metrics(
                        candidate_stress,
                        candidate_residual,
                        grid,
                        operator.points_per_pixel,
                    )[0]
                except ConstitutiveIntegrationError:
                    material.revert()
                    candidate_norm = np.inf
                if candidate_norm <= (1.0 - config.armijo_coefficient * relaxation) * relative:
                    break
                relaxation *= config.relaxation_reduction
                if relaxation >= config.minimum_relaxation:
                    continue
                if target is not image:
                    anderson.reset()
                    target = image
                    relaxation = 1.0
                    continue
                material.revert()
                failed = True
                break
            if failed:
                break
            minimum_relaxation = min(minimum_relaxation, relaxation)
            fluctuation = candidate_fluctuation
        if not accepted:
            material.revert()
            fluctuation = fluctuation_start
            if (
                increment_cutbacks >= config.maximum_cutbacks_per_increment
                or time_increment * 0.5
                < initial_increment * config.minimum_increment_fraction
            ):
                raise SpectralIncrementConvergenceError(
                    f"increment {increment} did not converge after {cutbacks} cutbacks"
                )
            midpoint = 0.5 * (previous_boundary + boundary_target)
            load_points[load_index] = (midpoint, 0.5 * time_increment)
            load_points.insert(load_index + 1, (boundary_target, 0.5 * time_increment))
            cutbacks += 1
            increment_cutbacks += 1
            continue
        load_index += 1
        previous_boundary = boundary_target.copy()
        increment_cutbacks = 0

    if final_trial is None:
        raise SpectralIncrementConvergenceError("no increment was solved")
    anderson_diagnostics = anderson.diagnostics
    diagnostics = Spectral2DDiagnostics(
        spatial_scheme=config.spatial_scheme,
        green_operator=config.green_operator,
        pixels=grid.pixel_shape,
        material_points=operator.material_point_count,
        points_per_pixel=operator.points_per_pixel,
        spacing_x=grid.spacing_x,
        spacing_y=grid.spacing_y,
        relative_residual_history=tuple(relative_history),
        dimensionless_equilibrium_history=tuple(dimensionless_history),
        absolute_residual_history=tuple(absolute_history),
        iterations_per_increment=tuple(iterations_per_increment),
        cutbacks=cutbacks,
        anderson_proposals=anderson_diagnostics.proposals,
        anderson_accelerated_proposals=anderson_diagnostics.accelerated_proposals,
        anderson_resets=anderson_diagnostics.resets,
        minimum_relaxation=minimum_relaxation,
        maximum_plane_stress_residual_mpa=float(
            np.max(np.abs(final_trial.plane_stress_residual_mpa))
        ),
        total_seconds=perf_counter() - started,
    )
    return Spectral2DResult(
        displacement=final_applied + fluctuation,
        applied_displacement=final_applied,
        fluctuation_displacement=fluctuation.copy(),
        strain_in_plane=final_strain,
        stress_in_plane_mpa=final_stress,
        full_stress_tensor_mpa=final_trial.full_stress_tensor_mpa,
        full_strain_tensor=final_trial.full_strain_tensor,
        elastic_strain_tensor=final_trial.elastic_strain_tensor,
        plastic_strain_tensor=final_trial.plastic_strain_tensor,
        observables=final_trial.observables,
        reaction_forces=_boundary_reactions(operator.divergence(final_stress)),
        diagnostics=diagnostics,
    )
