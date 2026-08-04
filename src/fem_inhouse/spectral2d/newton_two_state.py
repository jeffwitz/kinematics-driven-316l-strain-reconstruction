"""Newton-GMRES oracle using two independent SRIX states per TRI2 pixel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator, gmres

from fem_inhouse.core.plane_stress_material import (
    InPlaneConstitutiveTrial,
    PlaneStressMaterialBatch,
)
from fem_inhouse.spectral2d.boundary import HarmonicDirichletExtension2D
from fem_inhouse.spectral2d.diagnostics import Spectral2DDiagnostics
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
from fem_inhouse.spectral2d.transforms import FullDirichletDSTIPlan2D

FloatArray = NDArray[np.float64]


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
    ) -> FloatArray:
        delta_sample = kinematics.strain_samples(displacement_increment)
        delta_stress = np.einsum(
            "xyqij,xyqj->xyqi", trial.algorithmic_tangent_in_plane_mpa, delta_sample
        )
        return kinematics.divergence_from_sample_stress(delta_stress)

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
) -> Spectral2DResult:
    """Solve the direct two-state TRI2 oracle with the EBI Newton machinery."""

    history = np.asarray(boundary_displacement_history, dtype=np.float64)
    expected = (history.shape[0], *grid.node_shape, 2)
    if history.ndim != 4 or history.shape != expected or not np.allclose(history[0], 0.0):
        raise ValueError(f"invalid boundary history shape {history.shape}")
    kinematics = TwoSubcellDiagnostic2D(grid)
    elements = TraditionalTwoStateTriangleBatch(material, grid.pixel_shape)
    extension = HarmonicDirichletExtension2D()
    plan = FullDirichletDSTIPlan2D(grid)
    zero_trial = material.evaluate_in_plane(
        np.zeros((material.point_count, 3)), time_increment=1.0, consistent_tangent=True
    )
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
    fluctuation = np.zeros((*grid.node_shape, 2))
    residual_history: list[float] = []
    absolute_history: list[float] = []
    verification_history: list[float] = []
    verification_mismatch_history: list[float] = []
    iterations_per_increment: list[int] = []
    material_evaluations = 1
    gmres_iterations = 0
    final_trial = None
    final_sample_strain = None
    verification_residual = 0.0
    final_applied = history[0].copy()
    time_increment = 1.0 / (history.shape[0] - 1)

    for increment in range(1, history.shape[0]):
        applied = extension.extend(history[increment], grid)
        converged = False
        for iteration in range(config.maximum_newton_iterations):
            sample_strain = kinematics.strain_samples(applied + fluctuation)
            trial = elements.evaluate_samples(sample_strain, time_increment=time_increment)
            material_evaluations += 1
            residual = kinematics.divergence_from_sample_stress(trial.sample_stress_mpa)
            relative, absolute, _ = _equilibrium_metrics(trial.sample_stress_mpa, residual, grid, 2)
            residual_history.append(relative)
            absolute_history.append(absolute)
            if relative <= config.relative_equilibrium_tolerance:
                solver_residual = relative
                elements.revert()
                verification_trial = elements.evaluate_samples(
                    sample_strain, time_increment=time_increment
                )
                material_evaluations += 1
                verification_force = kinematics.divergence_from_sample_stress(
                    verification_trial.sample_stress_mpa
                )
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

            def jacobian_action(vector: FloatArray, active_trial=trial) -> FloatArray:
                field = unpack_interior(vector, grid)
                return pack_interior(
                    elements.tangent_action(field, kinematics=kinematics, trial=active_trial)
                )

            def preconditioner_action(vector: FloatArray) -> FloatArray:
                field = unpack_interior(vector, grid)
                transformed = plan.forward_displacement(field[1:-1, 1:-1])
                corrected = plan.embed_interior(plan.inverse_displacement(green.apply(transformed)))
                return pack_interior(corrected)

            gmres_matrix = LinearOperator((size, size), matvec=jacobian_action, dtype=float)
            preconditioner = LinearOperator((size, size), matvec=preconditioner_action, dtype=float)

            def count_gmres(_residual: object) -> None:
                nonlocal gmres_iterations
                gmres_iterations += 1

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
            if info != 0 or not np.isfinite(correction).all():
                elements.revert()
                raise RuntimeError(f"two-state GMRES failed with info={info}")
            direction = unpack_interior(correction, grid)
            accepted = False
            factor = 1.0
            for _ in range(config.maximum_line_search_reductions + 1):
                candidate = fluctuation + factor * direction
                candidate_trial = elements.evaluate_samples(
                    kinematics.strain_samples(applied + candidate), time_increment=time_increment
                )
                material_evaluations += 1
                candidate_residual = kinematics.divergence_from_sample_stress(
                    candidate_trial.sample_stress_mpa
                )
                candidate_relative = _equilibrium_metrics(
                    candidate_trial.sample_stress_mpa, candidate_residual, grid, 2
                )[0]
                if candidate_relative < relative:
                    fluctuation = candidate
                    accepted = True
                    break
                factor *= 0.5
            if not accepted:
                elements.revert()
                raise RuntimeError("two-state Newton line search failed")
        if not converged:
            elements.revert()
            raise RuntimeError(f"two-state increment {increment} did not converge")

    if final_trial is None or final_sample_strain is None:
        raise RuntimeError("no two-state increment converged")
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
        timings={
            "material_evaluations": float(material_evaluations),
            "gmres_iterations": float(gmres_iterations),
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
