"""Newton-GMRES oracle using two independent SRIX states per TRI2 pixel."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator, gmres

from fem_inhouse.core.plane_stress_material import (
    InPlaneConstitutiveTrial,
    PlaneStressMaterialBatch,
)
from fem_inhouse.spectral2d.boundary import HarmonicDirichletExtension2D
from fem_inhouse.spectral2d.diagnostics import (
    JacobianActionDiagnostics,
    LinearSolveDiagnostics,
    PreconditionerActionDiagnostics,
    Spectral2DDiagnostics,
    collect_runtime_provenance,
)
from fem_inhouse.spectral2d.green import B0Green2D, project_isotropic_plane_stress_tangent
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.spectral2d.newton_ebi import (
    EBISpectralSolverConfig,
    pack_interior,
    unpack_interior,
)
from fem_inhouse.spectral2d.nonlinear import _boundary_reactions, _equilibrium_metrics
from fem_inhouse.spectral2d.result import Spectral2DResult
from fem_inhouse.spectral2d.transform_factory import create_full_dirichlet_dsti_plan
from fem_inhouse.spectral2d.transforms import BufferedTransformPlan2D, TransformPlan2D

FloatArray = NDArray[np.float64]


def _accumulate_jacobian(
    target: JacobianActionDiagnostics,
    source: JacobianActionDiagnostics,
) -> None:
    target.calls += source.calls
    target.total_seconds += source.total_seconds
    target.unpack_seconds += source.unpack_seconds
    target.gradient_seconds += source.gradient_seconds
    target.tangent_seconds += source.tangent_seconds
    target.divergence_seconds += source.divergence_seconds
    target.pack_seconds += source.pack_seconds


def _accumulate_preconditioner(
    target: PreconditionerActionDiagnostics,
    source: PreconditionerActionDiagnostics,
) -> None:
    target.calls += source.calls
    target.total_seconds += source.total_seconds
    target.reshape_seconds += source.reshape_seconds
    target.forward_transform_seconds += source.forward_transform_seconds
    target.green_seconds += source.green_seconds
    target.inverse_transform_seconds += source.inverse_transform_seconds
    target.output_copy_seconds += source.output_copy_seconds


@dataclass(frozen=True, slots=True)
class TraditionalTwoStateTrial:
    sample_strain: FloatArray
    sample_stress_mpa: FloatArray
    algorithmic_tangent_in_plane_mpa: FloatArray
    mean_stress_mpa: FloatArray
    material_trial: InPlaneConstitutiveTrial


class TraditionalTwoStateTriangleBatch:
    """Direct TRI2 material integration with two independent histories."""

    def __init__(self, material: PlaneStressMaterialBatch, pixel_shape: tuple[int, int]):
        self.material = material
        self.pixel_shape = pixel_shape
        if material.point_count != 2 * pixel_shape[0] * pixel_shape[1]:
            raise ValueError("two-state TRI2 requires two material states per pixel")

    def evaluate_samples(
        self, sample_strain: ArrayLike, *, time_increment: float
    ) -> TraditionalTwoStateTrial:
        values = np.asarray(sample_strain, dtype=np.float64)
        expected = (*self.pixel_shape, 2, 3)
        if values.shape != expected:
            raise ValueError(f"expected sample strain shape {expected}, got {values.shape}")
        material_trial = self.material.evaluate_in_plane(
            values.reshape(-1, 3), time_increment=time_increment, consistent_tangent=True
        )
        stress = np.asarray(material_trial.stress_in_plane_mpa).reshape(*self.pixel_shape, 2, 3)
        tangent = np.asarray(material_trial.tangent_in_plane_mpa).reshape(
            *self.pixel_shape, 2, 3, 3
        )
        return TraditionalTwoStateTrial(
            sample_strain=values.copy(),
            sample_stress_mpa=stress,
            algorithmic_tangent_in_plane_mpa=tangent,
            mean_stress_mpa=stress.mean(axis=2),
            material_trial=material_trial,
        )

    def tangent_action(
        self,
        displacement_increment: ArrayLike,
        *,
        kinematics: TwoSubcellDiagnostic2D,
        trial: TraditionalTwoStateTrial,
        action_diagnostics: JacobianActionDiagnostics | None = None,
    ) -> FloatArray:
        gradient_started = time.perf_counter()
        delta_sample = kinematics.strain_samples(displacement_increment)
        gradient_seconds = time.perf_counter() - gradient_started
        tangent_started = time.perf_counter()
        delta_stress = np.einsum(
            "xyqij,xyqj->xyqi", trial.algorithmic_tangent_in_plane_mpa, delta_sample
        )
        tangent_seconds = time.perf_counter() - tangent_started
        divergence_started = time.perf_counter()
        result = kinematics.divergence_from_sample_stress(delta_stress)
        divergence_seconds = time.perf_counter() - divergence_started
        if action_diagnostics is not None:
            action_diagnostics.gradient_seconds += gradient_seconds
            action_diagnostics.tangent_seconds += tangent_seconds
            action_diagnostics.divergence_seconds += divergence_seconds
        return result

    def complete_trial(self, trial: TraditionalTwoStateTrial):
        return self.material.complete_trial(trial.material_trial)

    def commit(self) -> None:
        self.material.commit()

    def revert(self) -> None:
        self.material.revert()


def _reshape_two_state(values: ArrayLike, grid: StructuredGrid2D) -> FloatArray:
    array = np.asarray(values)
    return array.reshape(*grid.pixel_shape, 2, *array.shape[1:])


def solve_two_state_dirichlet_plane_stress(
    *,
    grid: StructuredGrid2D,
    material: PlaneStressMaterialBatch,
    boundary_displacement_history: ArrayLike,
    config: EBISpectralSolverConfig,
    transform_plan: TransformPlan2D | None = None,
) -> Spectral2DResult:
    """Solve the direct two-state TRI2 oracle with the EBI Newton machinery."""

    history = np.asarray(boundary_displacement_history, dtype=np.float64)
    expected = (history.shape[0], *grid.node_shape, 2)
    if history.ndim != 4 or history.shape != expected or not np.allclose(history[0], 0.0):
        raise ValueError(f"invalid boundary history shape {history.shape}")
    kinematics = TwoSubcellDiagnostic2D(grid)
    elements = TraditionalTwoStateTriangleBatch(material, grid.pixel_shape)
    extension = HarmonicDirichletExtension2D()
    material_seconds = 0.0
    gradient_seconds = 0.0
    divergence_seconds = 0.0
    gmres_seconds = 0.0
    material_evaluations = 1

    def evaluate_samples_timed(strain: FloatArray, dt: float) -> TraditionalTwoStateTrial:
        nonlocal material_seconds, material_evaluations
        started = time.perf_counter()
        result = elements.evaluate_samples(strain, time_increment=dt)
        material_seconds += time.perf_counter() - started
        material_evaluations += 1
        return result

    def strain_timed(displacement: FloatArray) -> FloatArray:
        nonlocal gradient_seconds
        started = time.perf_counter()
        result = kinematics.strain_samples(displacement)
        gradient_seconds += time.perf_counter() - started
        return result

    def divergence_timed(stress: FloatArray) -> FloatArray:
        nonlocal divergence_seconds
        started = time.perf_counter()
        result = kinematics.divergence_from_sample_stress(stress)
        divergence_seconds += time.perf_counter() - started
        return result

    plan = transform_plan or create_full_dirichlet_dsti_plan(grid, config.transform)
    initial_material_started = time.perf_counter()
    zero_trial = material.evaluate_in_plane(
        np.zeros((material.point_count, 3)), time_increment=1.0, consistent_tangent=True
    )
    material_seconds += time.perf_counter() - initial_material_started
    material.revert()
    tangent = np.asarray(zero_trial.tangent_in_plane_mpa).reshape(*grid.pixel_shape, 2, 3, 3)
    lambda_0, mu_0, projection_error = project_isotropic_plane_stress_tangent(
        tangent.mean(axis=(0, 1, 2))
    )
    lambda_0 *= config.reference_parameter_scale * config.reference_lambda_mu_ratio
    mu_0 *= config.reference_parameter_scale
    green = B0Green2D(
        kinematics.reference_operator_symbols(plan),
        lambda_0=lambda_0,
        mu_0=mu_0,
        symbol_null_tolerance=config.symbol_null_tolerance,
    )
    interior_shape = (*grid.interior_shape, 2)
    spectral_buffer = np.empty(interior_shape, dtype=np.float64)
    green_buffer = np.empty_like(spectral_buffer)
    physical_buffer = np.empty_like(spectral_buffer)
    fluctuation = np.zeros((*grid.node_shape, 2))
    residual_history: list[float] = []
    absolute_history: list[float] = []
    verification_history: list[float] = []
    verification_mismatch_history: list[float] = []
    iterations_per_increment: list[int] = []
    gmres_iterations = 0
    linear_solves: list[LinearSolveDiagnostics] = []
    jacobian_totals = JacobianActionDiagnostics()
    preconditioner_totals = PreconditionerActionDiagnostics()
    final_trial = None
    final_sample_strain = None
    verification_residual = 0.0
    final_applied = history[0].copy()
    time_increment = 1.0 / (history.shape[0] - 1)

    for increment in range(1, history.shape[0]):
        applied = extension.extend(history[increment], grid)
        converged = False
        for iteration in range(config.maximum_newton_iterations):
            sample_strain = strain_timed(applied + fluctuation)
            trial = evaluate_samples_timed(sample_strain, time_increment)
            residual = divergence_timed(trial.sample_stress_mpa)
            relative, absolute, _ = _equilibrium_metrics(trial.sample_stress_mpa, residual, grid, 2)
            residual_history.append(relative)
            absolute_history.append(absolute)
            if relative <= config.relative_equilibrium_tolerance:
                solver_residual = relative
                elements.revert()
                verification_trial = evaluate_samples_timed(sample_strain, time_increment)
                verification_force = divergence_timed(verification_trial.sample_stress_mpa)
                verification_residual = _equilibrium_metrics(
                    verification_trial.sample_stress_mpa,
                    verification_force,
                    grid,
                    2,
                )[0]
                verification_mismatch = abs(verification_residual - solver_residual) / max(
                    solver_residual, 1.0e-30
                )
                verification_history.append(verification_residual)
                verification_mismatch_history.append(verification_mismatch)
                if (
                    verification_residual <= config.relative_equilibrium_tolerance
                    and verification_mismatch <= 1.0e-3
                ):
                    final_trial = elements.complete_trial(verification_trial)
                    elements.commit()
                    final_sample_strain = sample_strain.copy()
                    final_applied = applied.copy()
                    iterations_per_increment.append(iteration + 1)
                    converged = True
                    break
                trial = verification_trial
                residual = verification_force
                relative = verification_residual

            size = 2 * (grid.nx - 1) * (grid.ny - 1)
            jacobian_local = JacobianActionDiagnostics()
            preconditioner_local = PreconditionerActionDiagnostics()
            gmres_iterations_before = gmres_iterations

            def jacobian_action(
                vector: FloatArray,
                active_trial=trial,
                action_counters=jacobian_local,
            ) -> FloatArray:
                started = time.perf_counter()
                action_counters.calls += 1
                unpack_started = time.perf_counter()
                field = unpack_interior(vector, grid)
                action_counters.unpack_seconds += time.perf_counter() - unpack_started
                pack_started = time.perf_counter()
                result = pack_interior(
                    elements.tangent_action(
                        field,
                        kinematics=kinematics,
                        trial=active_trial,
                        action_diagnostics=action_counters,
                    )
                )
                action_counters.pack_seconds += time.perf_counter() - pack_started
                action_counters.total_seconds += time.perf_counter() - started
                return result

            def preconditioner_action(
                vector: FloatArray,
                action_counters=preconditioner_local,
            ) -> FloatArray:
                started = time.perf_counter()
                action_counters.calls += 1
                reshape_started = time.perf_counter()
                interior = np.asarray(vector, dtype=np.float64).reshape(interior_shape)
                action_counters.reshape_seconds += time.perf_counter() - reshape_started
                forward_started = time.perf_counter()
                if hasattr(plan, "forward_into"):
                    buffered_plan = cast(BufferedTransformPlan2D, plan)
                    buffered_plan.forward_into(interior, spectral_buffer)
                else:
                    spectral_buffer[...] = plan.forward_displacement(interior)
                action_counters.forward_transform_seconds += (
                    time.perf_counter() - forward_started
                )
                green_started = time.perf_counter()
                green.apply_into(spectral_buffer, green_buffer)
                action_counters.green_seconds += time.perf_counter() - green_started
                inverse_started = time.perf_counter()
                if hasattr(plan, "forward_into"):
                    buffered_plan = cast(BufferedTransformPlan2D, plan)
                    buffered_plan.inverse_into(green_buffer, physical_buffer)
                else:
                    physical_buffer[...] = plan.inverse_displacement(green_buffer)
                action_counters.inverse_transform_seconds += (
                    time.perf_counter() - inverse_started
                )
                copy_started = time.perf_counter()
                result = physical_buffer.reshape(-1).copy()
                action_counters.output_copy_seconds += time.perf_counter() - copy_started
                action_counters.total_seconds += time.perf_counter() - started
                return result

            gmres_matrix = LinearOperator((size, size), matvec=jacobian_action, dtype=float)
            preconditioner = LinearOperator((size, size), matvec=preconditioner_action, dtype=float)

            def count_gmres(_residual: object) -> None:
                nonlocal gmres_iterations
                gmres_iterations += 1

            gmres_started = time.perf_counter()
            correction, info = gmres(
                gmres_matrix,
                -pack_interior(residual),
                M=preconditioner,
                rtol=config.gmres_relative_tolerance,
                atol=0.0,
                restart=config.gmres_restart,
                maxiter=config.gmres_maximum_iterations,
                callback=count_gmres,
                callback_type="pr_norm",
            )
            gmres_elapsed = time.perf_counter() - gmres_started
            gmres_seconds += gmres_elapsed
            if info != 0 or not np.isfinite(correction).all():
                _accumulate_jacobian(jacobian_totals, jacobian_local)
                _accumulate_preconditioner(preconditioner_totals, preconditioner_local)
                elements.revert()
                raise RuntimeError(f"two-state GMRES failed with info={info}")
            direction = unpack_interior(correction, grid)
            accepted = False
            factor = 1.0
            for _ in range(config.maximum_line_search_reductions + 1):
                candidate = fluctuation + factor * direction
                candidate_strain = strain_timed(applied + candidate)
                candidate_trial = evaluate_samples_timed(candidate_strain, time_increment)
                candidate_residual = divergence_timed(candidate_trial.sample_stress_mpa)
                candidate_relative = _equilibrium_metrics(
                    candidate_trial.sample_stress_mpa, candidate_residual, grid, 2
                )[0]
                if candidate_relative < relative:
                    fluctuation = candidate
                    accepted = True
                    break
                factor *= 0.5
            if not accepted:
                _accumulate_jacobian(jacobian_totals, jacobian_local)
                _accumulate_preconditioner(preconditioner_totals, preconditioner_local)
                elements.revert()
                raise RuntimeError("two-state Newton line search failed")
            _accumulate_jacobian(jacobian_totals, jacobian_local)
            _accumulate_preconditioner(preconditioner_totals, preconditioner_local)
            linear_solves.append(
                LinearSolveDiagnostics(
                    increment=increment,
                    newton_iteration=iteration + 1,
                    nonlinear_residual_before=relative,
                    requested_relative_tolerance=config.gmres_relative_tolerance,
                    gmres_info=int(info),
                    gmres_iterations=gmres_iterations - gmres_iterations_before,
                    jacobian_calls=jacobian_local.calls,
                    preconditioner_calls=preconditioner_local.calls,
                    gmres_seconds=gmres_elapsed,
                    jacobian_seconds=jacobian_local.total_seconds,
                    preconditioner_seconds=preconditioner_local.total_seconds,
                    krylov_overhead_seconds=max(
                        0.0,
                        gmres_elapsed
                        - jacobian_local.total_seconds
                        - preconditioner_local.total_seconds,
                    ),
                    restart=config.gmres_restart,
                    line_search_factor=factor,
                )
            )
        if not converged:
            elements.revert()
            raise RuntimeError(f"two-state increment {increment} did not converge")

    if final_trial is None or final_sample_strain is None:
        raise RuntimeError("no two-state increment converged")
    transform_diagnostics = plan.diagnostics
    observables = {
        name: _reshape_two_state(values, grid) for name, values in final_trial.observables.items()
    }
    diagnostics = Spectral2DDiagnostics(
        spatial_scheme="traditional_two_state_triangle",
        green_operator="b0",
        pixels=grid.pixel_shape,
        material_points=material.point_count,
        points_per_pixel=2,
        spacing_x=grid.spacing_x,
        spacing_y=grid.spacing_y,
        relative_residual_history=tuple(residual_history),
        dimensionless_equilibrium_history=tuple(residual_history),
        absolute_residual_history=tuple(absolute_history),
        iterations_per_increment=tuple(iterations_per_increment),
        reference_lambda_0=lambda_0,
        reference_mu_0=mu_0,
        reference_projection_error=projection_error,
        verification_residual=verification_residual,
        verification_residual_history=tuple(verification_history),
        verification_relative_mismatch_history=tuple(verification_mismatch_history),
        transform_backend=transform_diagnostics.backend,
        transform_implementation=transform_diagnostics.implementation,
        transform_interior_shape=transform_diagnostics.interior_shape,
        transform_batch_components=transform_diagnostics.batch_components,
        transform_dtype=transform_diagnostics.dtype,
        transform_workers=transform_diagnostics.workers,
        transform_planner_effort=transform_diagnostics.planner_effort,
        transform_wisdom_loaded=transform_diagnostics.wisdom_loaded,
        transform_planning_seconds=transform_diagnostics.planning_seconds,
        linear_solves=tuple(linear_solves),
        provenance=collect_runtime_provenance(
            transform_diagnostics,
            gmres_restart=config.gmres_restart,
            gmres_maximum_iterations=config.gmres_maximum_iterations,
            gmres_relative_tolerance=config.gmres_relative_tolerance,
        ),
        timings={
            "material_evaluations": float(material_evaluations),
            "material_seconds": material_seconds,
            "gradient_seconds": gradient_seconds,
            "divergence_seconds": divergence_seconds,
            "jacobian_seconds": jacobian_totals.total_seconds,
            "preconditioner_seconds": preconditioner_totals.total_seconds,
            "gmres_iterations": float(gmres_iterations),
            "gmres_seconds": gmres_seconds,
            "jacobian_calls": float(jacobian_totals.calls),
            "jacobian_unpack_seconds": jacobian_totals.unpack_seconds,
            "jacobian_gradient_seconds": jacobian_totals.gradient_seconds,
            "jacobian_tangent_seconds": jacobian_totals.tangent_seconds,
            "jacobian_divergence_seconds": jacobian_totals.divergence_seconds,
            "jacobian_pack_seconds": jacobian_totals.pack_seconds,
            "preconditioner_calls": float(preconditioner_totals.calls),
            "preconditioner_reshape_seconds": preconditioner_totals.reshape_seconds,
            "preconditioner_forward_transform_seconds": (
                preconditioner_totals.forward_transform_seconds
            ),
            "preconditioner_green_seconds": preconditioner_totals.green_seconds,
            "preconditioner_inverse_transform_seconds": (
                preconditioner_totals.inverse_transform_seconds
            ),
            "preconditioner_output_copy_seconds": preconditioner_totals.output_copy_seconds,
            "krylov_overhead_seconds": max(
                0.0,
                gmres_seconds
                - jacobian_totals.total_seconds
                - preconditioner_totals.total_seconds,
            ),
        },
    )
    sample_stress = _reshape_two_state(final_trial.stress_in_plane_mpa, grid)
    return Spectral2DResult(
        displacement=final_applied + fluctuation,
        applied_displacement=final_applied,
        fluctuation_displacement=fluctuation.copy(),
        strain_in_plane=final_sample_strain,
        stress_in_plane_mpa=sample_stress,
        full_stress_tensor_mpa=_reshape_two_state(final_trial.full_stress_tensor_mpa, grid),
        full_strain_tensor=_reshape_two_state(final_trial.full_strain_tensor, grid),
        elastic_strain_tensor=_reshape_two_state(final_trial.elastic_strain_tensor, grid),
        plastic_strain_tensor=_reshape_two_state(final_trial.plastic_strain_tensor, grid),
        observables=observables,
        reaction_forces=_boundary_reactions(
            kinematics.divergence_from_sample_stress(sample_stress)
        ),
        diagnostics=diagnostics,
    )
