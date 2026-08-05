"""Newton-GMRES oracle using two independent SRIX states per TRI2 pixel."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator

from fem_inhouse.core.plane_stress_material import (
    InPlaneConstitutiveTrial,
    PlaneStressMaterialBatch,
    ResponseLevel,
    evaluate_in_plane_response,
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
from fem_inhouse.spectral2d.krylov import KrylovRecycleState, solve_nonsymmetric_krylov
from fem_inhouse.spectral2d.newton_ebi import (
    EBISpectralSolverConfig,
    pack_interior,
    pack_interior_into,
    unpack_interior,
    unpack_interior_into,
)
from fem_inhouse.spectral2d.nonlinear import _boundary_reactions, _equilibrium_metrics
from fem_inhouse.spectral2d.result import Spectral2DResult
from fem_inhouse.spectral2d.transform_factory import create_full_dirichlet_dsti_plan
from fem_inhouse.spectral2d.transforms import BufferedTransformPlan2D, TransformPlan2D

FloatArray = NDArray[np.float64]
TangentKernel = Literal["einsum", "explicit"]


@dataclass(slots=True)
class TwoStateJacobianWorkspace:
    """Persistent buffers for one two-state TRI2 Newton resolution."""

    nodal_increment: FloatArray
    sample_strain_increment: FloatArray
    sample_stress_increment: FloatArray
    nodal_force: FloatArray
    interior_force: FloatArray

    @classmethod
    def create(cls, grid: StructuredGrid2D) -> TwoStateJacobianWorkspace:
        return cls(
            nodal_increment=np.zeros((*grid.node_shape, 2), dtype=np.float64),
            sample_strain_increment=np.empty((*grid.pixel_shape, 2, 3), dtype=np.float64),
            sample_stress_increment=np.empty((*grid.pixel_shape, 2, 3), dtype=np.float64),
            nodal_force=np.empty((*grid.node_shape, 2), dtype=np.float64),
            interior_force=np.empty(2 * (grid.nx - 1) * (grid.ny - 1), dtype=np.float64),
        )


def apply_tangent_into(
    tangent: ArrayLike,
    strain: ArrayLike,
    destination: FloatArray,
    *,
    kernel: TangentKernel = "einsum",
) -> None:
    """Apply all 3x3 sample tangents into a reusable stress buffer."""

    values = np.asarray(tangent, dtype=np.float64)
    delta = np.asarray(strain, dtype=np.float64)
    if destination.shape != delta.shape or values.shape != (*delta.shape[:-1], 3, 3):
        raise ValueError("incompatible tangent action shapes")
    if kernel == "einsum":
        np.einsum("xyqij,xyqj->xyqi", values, delta, out=destination)
    elif kernel == "explicit":
        e0, e1, e2 = delta[..., 0], delta[..., 1], delta[..., 2]
        destination[..., 0] = (
            values[..., 0, 0] * e0 + values[..., 0, 1] * e1 + values[..., 0, 2] * e2
        )
        destination[..., 1] = (
            values[..., 1, 0] * e0 + values[..., 1, 1] * e1 + values[..., 1, 2] * e2
        )
        destination[..., 2] = (
            values[..., 2, 0] * e0 + values[..., 2, 1] * e1 + values[..., 2, 2] * e2
        )
    else:
        raise ValueError(f"unsupported tangent kernel: {kernel}")


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


def _linear_tolerance(
    config: EBISpectralSolverConfig,
    residual: float,
    previous_residual: float | None,
    *,
    force_fixed: bool,
) -> float:
    if config.linear_tolerance_mode == "fixed" or force_fixed:
        return config.gmres_relative_tolerance
    if previous_residual is None:
        eta = config.forcing_initial
    else:
        ratio = residual / max(previous_residual, 1.0e-30)
        eta = config.forcing_gamma * ratio**config.forcing_alpha
        eta = min(config.forcing_maximum, max(config.forcing_minimum, eta))
    if residual <= 10.0 * config.relative_equilibrium_tolerance:
        eta = min(
            eta,
            max(config.forcing_minimum, 0.1 * residual),
        )
    return max(config.forcing_minimum, min(config.forcing_maximum, eta))


@dataclass(frozen=True, slots=True)
class TraditionalTwoStateTrial:
    sample_strain: FloatArray
    sample_stress_mpa: FloatArray
    algorithmic_tangent_in_plane_mpa: FloatArray | None
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
        self,
        sample_strain: ArrayLike,
        *,
        time_increment: float,
        response_level: ResponseLevel = "tangent",
    ) -> TraditionalTwoStateTrial:
        values = np.asarray(sample_strain, dtype=np.float64)
        expected = (*self.pixel_shape, 2, 3)
        if values.shape != expected:
            raise ValueError(f"expected sample strain shape {expected}, got {values.shape}")
        material_trial = evaluate_in_plane_response(
            self.material,
            values.reshape(-1, 3),
            time_increment=time_increment,
            response_level=response_level,
            consistent_tangent=response_level != "residual",
        )
        stress = np.asarray(material_trial.stress_in_plane_mpa).reshape(*self.pixel_shape, 2, 3)
        tangent = (
            None
            if material_trial.tangent_in_plane_mpa is None
            else np.asarray(material_trial.tangent_in_plane_mpa).reshape(
                *self.pixel_shape, 2, 3, 3
            )
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
        if trial.algorithmic_tangent_in_plane_mpa is None:
            raise ValueError("TRI2 tangent action requires a tangent trial")
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

    def tangent_action_into(
        self,
        *,
        kinematics: TwoSubcellDiagnostic2D,
        trial: TraditionalTwoStateTrial,
        workspace: TwoStateJacobianWorkspace,
        action_diagnostics: JacobianActionDiagnostics | None = None,
        kernel: TangentKernel = "einsum",
    ) -> FloatArray:
        """Apply the tangent using persistent TRI2 work arrays."""

        gradient_started = time.perf_counter()
        if trial.algorithmic_tangent_in_plane_mpa is None:
            raise ValueError("TRI2 tangent action requires a tangent trial")
        kinematics.strain_samples_into(
            workspace.nodal_increment,
            workspace.sample_strain_increment,
        )
        gradient_seconds = time.perf_counter() - gradient_started
        tangent_started = time.perf_counter()
        apply_tangent_into(
            trial.algorithmic_tangent_in_plane_mpa,
            workspace.sample_strain_increment,
            workspace.sample_stress_increment,
            kernel=kernel,
        )
        tangent_seconds = time.perf_counter() - tangent_started
        divergence_started = time.perf_counter()
        kinematics.divergence_from_sample_stress_into(
            workspace.sample_stress_increment,
            workspace.nodal_force,
        )
        divergence_seconds = time.perf_counter() - divergence_started
        pack_started = time.perf_counter()
        pack_interior_into(workspace.nodal_force, workspace.interior_force)
        pack_seconds = time.perf_counter() - pack_started
        if action_diagnostics is not None:
            action_diagnostics.gradient_seconds += gradient_seconds
            action_diagnostics.tangent_seconds += tangent_seconds
            action_diagnostics.divergence_seconds += divergence_seconds
            action_diagnostics.pack_seconds += pack_seconds
        return workspace.interior_force.copy()

    def complete_trial(self, trial: TraditionalTwoStateTrial):
        return self.material.complete_trial(trial.material_trial)

    def commit(self) -> None:
        self.material.commit()

    def revert(self) -> None:
        self.material.revert()

    def accept_global_trial(self) -> None:
        accept = getattr(self.material, "accept_global_trial", None)
        if callable(accept):
            accept()


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

    def evaluate_samples_timed(
        strain: FloatArray,
        dt: float,
        response_level: ResponseLevel = "tangent",
    ) -> TraditionalTwoStateTrial:
        nonlocal material_seconds, material_evaluations
        started = time.perf_counter()
        result = elements.evaluate_samples(
            strain,
            time_increment=dt,
            response_level=response_level,
        )
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
    zero_trial = evaluate_in_plane_response(
        material,
        np.zeros((material.point_count, 3)),
        time_increment=1.0,
        response_level="tangent",
        consistent_tangent=True,
    )
    material_seconds += time.perf_counter() - initial_material_started
    material.revert()
    tangent = np.asarray(zero_trial.tangent_in_plane_mpa).reshape(*grid.pixel_shape, 2, 3, 3)
    projected_lambda, projected_mu, projection_error = project_isotropic_plane_stress_tangent(
        tangent.mean(axis=(0, 1, 2))
    )
    if config.reference_parameter_mode == "explicit":
        assert config.reference_lambda_0 is not None
        assert config.reference_mu_0 is not None
        lambda_0 = config.reference_lambda_0
        mu_0 = config.reference_mu_0
    else:
        lambda_0 = (
            projected_lambda
            * config.reference_parameter_scale
            * config.reference_lambda_mu_ratio
        )
        mu_0 = projected_mu * config.reference_parameter_scale
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
    jacobian_workspace = TwoStateJacobianWorkspace.create(grid)
    fluctuation = np.zeros((*grid.node_shape, 2))
    residual_history: list[float] = []
    absolute_history: list[float] = []
    verification_history: list[float] = []
    verification_mismatch_history: list[float] = []
    iterations_per_increment: list[int] = []
    reference_updates: list[dict[str, str | int | float | bool]] = []
    gmres_iterations = 0
    linear_solves: list[LinearSolveDiagnostics] = []
    jacobian_totals = JacobianActionDiagnostics()
    preconditioner_totals = PreconditionerActionDiagnostics()
    final_trial = None
    final_sample_strain = None
    verification_residual = 0.0
    final_applied = history[0].copy()
    time_increment = 1.0 / (history.shape[0] - 1)
    krylov_recycle = KrylovRecycleState()

    for increment in range(1, history.shape[0]):
        krylov_recycle.reset()
        applied = extension.extend(history[increment], grid)
        converged = False
        previous_nonlinear_residual: float | None = None
        force_fixed_linear_tolerance = False
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
                verification_trial = evaluate_samples_timed(
                    sample_strain,
                    time_increment,
                    response_level="complete",
                )
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

            if config.reference_update_mode != "initial":
                if config.reference_parameter_mode == "explicit":
                    reference_updates.append(
                        {
                            "increment": increment,
                            "newton_iteration": iteration + 1,
                            "accepted": False,
                            "reason": "explicit_reference_parameters",
                        }
                    )
                elif config.reference_update_mode == "per_newton" or iteration == 0:
                    if trial.algorithmic_tangent_in_plane_mpa is None:
                        raise RuntimeError("reference update requires a tangent trial")
                    candidate_lambda, candidate_mu, _ = project_isotropic_plane_stress_tangent(
                        trial.algorithmic_tangent_in_plane_mpa.mean(axis=(0, 1, 2))
                    )
                    candidate_lambda *= config.reference_parameter_scale
                    candidate_mu *= config.reference_parameter_scale
                    relaxation = config.reference_update_relaxation
                    updated_lambda = (1.0 - relaxation) * lambda_0 + relaxation * candidate_lambda
                    updated_mu = (1.0 - relaxation) * mu_0 + relaxation * candidate_mu
                    mu_ratio = updated_mu / mu_0
                    bulk_ratio = (updated_lambda + updated_mu) / (lambda_0 + mu_0)
                    relative_change = max(
                        abs(updated_mu - mu_0) / max(abs(mu_0), 1.0e-30),
                        abs(updated_lambda - lambda_0) / max(abs(lambda_0), 1.0e-30),
                    )
                    accepted_update = (
                        np.isfinite(updated_lambda)
                        and np.isfinite(updated_mu)
                        and 0.25 <= mu_ratio <= 4.0
                        and 0.25 <= bulk_ratio <= 4.0
                        and relative_change >= config.reference_minimum_relative_change
                    )
                    if accepted_update:
                        green.update_parameters(
                            lambda_0=updated_lambda,
                            mu_0=updated_mu,
                        )
                        lambda_0 = updated_lambda
                        mu_0 = updated_mu
                        reason = "updated"
                    elif relative_change < config.reference_minimum_relative_change:
                        reason = "below_minimum_change"
                    else:
                        reason = "safeguard_rejected"
                    reference_updates.append(
                        {
                            "increment": increment,
                            "newton_iteration": iteration + 1,
                            "accepted": accepted_update,
                            "reason": reason,
                            "lambda_0": lambda_0,
                            "mu_0": mu_0,
                            "candidate_lambda": candidate_lambda,
                            "candidate_mu": candidate_mu,
                        }
                    )

            size = 2 * (grid.nx - 1) * (grid.ny - 1)
            jacobian_local = JacobianActionDiagnostics()
            preconditioner_local = PreconditionerActionDiagnostics()
            gmres_iterations_before = gmres_iterations
            requested_linear_tolerance = _linear_tolerance(
                config,
                relative,
                previous_nonlinear_residual,
                force_fixed=force_fixed_linear_tolerance,
            )
            force_fixed_linear_tolerance = False

            def jacobian_action(
                vector: FloatArray,
                active_trial=trial,
                action_counters=jacobian_local,
            ) -> FloatArray:
                started = time.perf_counter()
                action_counters.calls += 1
                unpack_started = time.perf_counter()
                unpack_interior_into(vector, grid, jacobian_workspace.nodal_increment)
                action_counters.unpack_seconds += time.perf_counter() - unpack_started
                result = elements.tangent_action_into(
                    kinematics=kinematics,
                    trial=active_trial,
                    workspace=jacobian_workspace,
                    action_diagnostics=action_counters,
                )
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
            rhs = -pack_interior(residual)
            correction, info, _krylov_calls = solve_nonsymmetric_krylov(
                gmres_matrix,
                rhs,
                preconditioner=preconditioner,
                method=config.krylov_method,
                rtol=requested_linear_tolerance,
                maximum_iterations=config.gmres_maximum_iterations,
                restart=config.gmres_restart,
                recycle=krylov_recycle if config.krylov_recycling else None,
                lgmres_inner_m=config.lgmres_inner_m,
                lgmres_outer_k=config.lgmres_outer_k,
                gcrotmk_m=config.gcrotmk_m,
                gcrotmk_k=config.gcrotmk_k,
                callback=count_gmres,
            )
            linear_residual_ratio: float | None = None
            if config.verify_linear_residual and info == 0 and np.isfinite(correction).all():
                linear_residual = jacobian_action(correction) - rhs
                linear_residual_ratio = float(
                    np.linalg.norm(linear_residual)
                    / max(np.linalg.norm(rhs), 1.0e-30)
                )
                force_fixed_linear_tolerance = linear_residual_ratio > (
                    1.5 * requested_linear_tolerance
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
                candidate_trial = evaluate_samples_timed(
                    candidate_strain,
                    time_increment,
                    response_level="residual",
                )
                candidate_residual = divergence_timed(candidate_trial.sample_stress_mpa)
                candidate_relative = _equilibrium_metrics(
                    candidate_trial.sample_stress_mpa, candidate_residual, grid, 2
                )[0]
                if candidate_relative < relative:
                    fluctuation = candidate
                    accept_global_trial = getattr(elements, "accept_global_trial", None)
                    if callable(accept_global_trial):
                        accept_global_trial()
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
                    requested_relative_tolerance=requested_linear_tolerance,
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
                    linear_residual_ratio=linear_residual_ratio,
                    krylov_method=config.krylov_method,
                    krylov_recycling=config.krylov_recycling,
                )
            )
            previous_nonlinear_residual = relative
            if factor < 1.0:
                force_fixed_linear_tolerance = True
        if not converged:
            elements.revert()
            raise RuntimeError(f"two-state increment {increment} did not converge")

    if final_trial is None or final_sample_strain is None:
        raise RuntimeError("no two-state increment converged")
    transform_diagnostics = plan.diagnostics
    material_timing = getattr(material, "timing_statistics", None)
    material_timing_values = {
        name: float(getattr(material_timing, name, 0.0))
        for name in (
            "rotation_to_material_seconds",
            "integration_seconds",
            "rotation_to_global_seconds",
            "condensation_seconds",
            "condition_check_seconds",
            "local_solve_seconds",
            "reconstruction_seconds",
            "observable_seconds",
        )
    }
    provenance = collect_runtime_provenance(
        transform_diagnostics,
        gmres_restart=config.gmres_restart,
        gmres_maximum_iterations=config.gmres_maximum_iterations,
        gmres_relative_tolerance=config.gmres_relative_tolerance,
        linear_tolerance_mode=config.linear_tolerance_mode,
        forcing_initial=config.forcing_initial,
        forcing_minimum=config.forcing_minimum,
        forcing_maximum=config.forcing_maximum,
        forcing_gamma=config.forcing_gamma,
        forcing_alpha=config.forcing_alpha,
        krylov_method=config.krylov_method,
        krylov_recycling=config.krylov_recycling,
    )
    provenance.update(
        {
            "material_backend": getattr(material, "backend_name", type(material).__name__),
            "material_matrix_type": getattr(
                material, "linear_system_matrix_type", "unspecified"
            ),
            "mfront_threads": getattr(material, "thread_count", None),
            "local_condition_check_mode": getattr(
                material, "local_condition_check_mode", None
            ),
        }
    )
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
        reference_updates=tuple(reference_updates),
        provenance=provenance,
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
            **{
                f"material_{name}": value
                for name, value in material_timing_values.items()
            },
            "material_condition_checks": float(
                getattr(material_timing, "condition_checks", 0)
            ),
            "material_evaluate_calls": float(
                getattr(material_timing, "evaluate_calls", 0)
            ),
            "material_warm_start_uses": float(
                getattr(material, "warm_start_uses", 0)
            ),
            "material_warm_start_resets": float(
                getattr(material, "warm_start_resets", 0)
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
