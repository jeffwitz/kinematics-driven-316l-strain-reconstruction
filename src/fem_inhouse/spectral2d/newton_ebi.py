"""Matrix-free Newton-GMRES solver for EBI two-triangle plane stress."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator, gmres

from fem_inhouse.core.plane_stress_material import HookeanPlaneStressMaterialBatch
from fem_inhouse.spectral2d.boundary import HarmonicDirichletExtension2D
from fem_inhouse.spectral2d.diagnostics import Spectral2DDiagnostics
from fem_inhouse.spectral2d.ebi import EBIPlaneStressElementBatch, EBIPlaneStressTrial
from fem_inhouse.spectral2d.green import B0Green2D, project_isotropic_plane_stress_tangent
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import EBITwoTriangleKinematics2D
from fem_inhouse.spectral2d.nonlinear import _boundary_reactions, _equilibrium_metrics
from fem_inhouse.spectral2d.result import Spectral2DResult
from fem_inhouse.spectral2d.transforms import FullDirichletDSTIPlan2D

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EBISpectralSolverConfig:
    relative_equilibrium_tolerance: float = 1.0e-8
    maximum_newton_iterations: int = 40
    gmres_relative_tolerance: float = 1.0e-8
    gmres_maximum_iterations: int = 200
    gmres_restart: int = 50
    maximum_line_search_reductions: int = 8
    reference_parameter_mode: Literal["explicit", "projected"] = "projected"
    reference_parameter_scale: float = 1.0
    reference_lambda_mu_ratio: float = 1.0
    reference_lambda_0: float | None = None
    reference_mu_0: float | None = None
    symbol_null_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.relative_equilibrium_tolerance <= 0.0:
            raise ValueError("equilibrium tolerance must be positive")
        if self.maximum_newton_iterations < 1 or self.gmres_maximum_iterations < 1:
            raise ValueError("iteration limits must be positive")
        if self.reference_parameter_scale <= 0.0:
            raise ValueError("reference parameter scale must be positive")
        if self.reference_lambda_mu_ratio <= 0.0:
            raise ValueError("reference lambda/mu ratio must be positive")
        if self.reference_parameter_mode == "explicit":
            if self.reference_lambda_0 is None or self.reference_mu_0 is None:
                raise ValueError("explicit B0 parameters require lambda_0 and mu_0")
            if self.reference_parameter_scale != 1.0:
                raise ValueError("explicit B0 parameters cannot also be scaled")
            if self.reference_mu_0 <= 0.0 or self.reference_lambda_0 + self.reference_mu_0 <= 0.0:
                raise ValueError("B0 parameters must satisfy mu>0 and lambda+mu>0")
        elif self.reference_parameter_mode == "projected":
            if self.reference_lambda_0 is not None or self.reference_mu_0 is not None:
                raise ValueError("projected B0 parameters reject explicit values")
        else:
            raise ValueError("unsupported reference parameter mode")


def pack_interior(field: ArrayLike) -> FloatArray:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("nodal field must have shape (nx+1, ny+1, 2)")
    return values[1:-1, 1:-1, :].reshape(-1).copy()


def unpack_interior(vector: ArrayLike, grid: StructuredGrid2D) -> FloatArray:
    values = np.asarray(vector, dtype=np.float64)
    expected = 2 * (grid.nx - 1) * (grid.ny - 1)
    if values.shape != (expected,):
        raise ValueError(f"expected interior vector length {expected}, got {values.shape}")
    field = np.zeros((*grid.node_shape, 2), dtype=np.float64)
    field[1:-1, 1:-1, :] = values.reshape(grid.nx - 1, grid.ny - 1, 2)
    return field


def _reshape_mean_field(values: ArrayLike, grid: StructuredGrid2D) -> FloatArray:
    array = np.asarray(values)
    return array.reshape(*grid.pixel_shape, *array.shape[1:])


def solve_ebi_dirichlet_plane_stress(
    *,
    grid: StructuredGrid2D,
    material: HookeanPlaneStressMaterialBatch,
    boundary_displacement_history: ArrayLike,
    config: EBISpectralSolverConfig,
) -> Spectral2DResult:
    """Solve full-Dirichlet EBI equilibrium with matrix-free Newton-GMRES."""

    history = np.asarray(boundary_displacement_history, dtype=np.float64)
    expected = (history.shape[0], *grid.node_shape, 2)
    if history.ndim != 4 or history.shape != expected:
        raise ValueError(f"expected boundary history shape {expected}, got {history.shape}")
    if not np.allclose(history[0], 0.0):
        raise ValueError("the first boundary displacement must be zero")
    kinematics = EBITwoTriangleKinematics2D(grid)
    if material.point_count != grid.nx * grid.ny:
        raise ValueError("EBI requires exactly one material state per pixel")
    elements = EBIPlaneStressElementBatch(material, grid.pixel_shape)
    extension = HarmonicDirichletExtension2D()
    plan = FullDirichletDSTIPlan2D(grid)
    symbols = kinematics.reference_operator_symbols(plan)
    projected_lambda, projected_mu, projection_error = project_isotropic_plane_stress_tangent(
        np.asarray(material.elastic_tangent_in_plane_mpa).mean(axis=0)
    )
    if config.reference_parameter_mode == "explicit":
        assert config.reference_lambda_0 is not None
        assert config.reference_mu_0 is not None
        lambda_0 = config.reference_lambda_0
        mu_0 = config.reference_mu_0
    else:
        lambda_0 = (
            projected_lambda * config.reference_parameter_scale * config.reference_lambda_mu_ratio
        )
        mu_0 = projected_mu * config.reference_parameter_scale
    green = B0Green2D(
        symbols,
        lambda_0=lambda_0,
        mu_0=mu_0,
        symbol_null_tolerance=config.symbol_null_tolerance,
    )
    fluctuation = np.zeros((*grid.node_shape, 2), dtype=np.float64)
    residual_history: list[float] = []
    absolute_residual_history: list[float] = []
    verification_history: list[float] = []
    verification_mismatch_history: list[float] = []
    highest_mode_energy_history: list[float] = []
    high_frequency_fraction_history: list[float] = []
    fluctuation_norm_history: list[float] = []
    highest_mode_residual_history: list[float] = []
    line_search_factors: list[float] = []
    iterations_per_increment: list[int] = []
    final_trial = None
    final_ebi_trial = None
    final_applied = np.zeros_like(history[0])
    material_evaluations = 0
    gmres_iterations = 0
    verification_residual = 0.0
    time_increment = 1.0 / (history.shape[0] - 1)
    mode_x, mode_y = np.meshgrid(
        np.arange(symbols.laplacian.shape[0]),
        np.arange(symbols.laplacian.shape[1]),
        indexing="ij",
    )
    normalized_radius = np.sqrt(
        (mode_x / max(symbols.laplacian.shape[0] - 1, 1)) ** 2
        + (mode_y / max(symbols.laplacian.shape[1] - 1, 1)) ** 2
    )
    high_frequency_mask = normalized_radius >= np.quantile(normalized_radius, 0.9)

    for increment in range(1, history.shape[0]):
        applied = extension.extend(history[increment], grid)
        converged = False
        for newton_iteration in range(config.maximum_newton_iterations):
            sample_strain = kinematics.strain_samples(applied + fluctuation)
            trial = elements.evaluate_samples(
                sample_strain,
                time_increment=time_increment,
                consistent_tangent=True,
            )
            material_evaluations += 1
            residual = kinematics.divergence_from_sample_stress(trial.sample_stress_mpa)
            relative, divergence_norm, _ = _equilibrium_metrics(
                trial.sample_stress_mpa, residual, grid, 2
            )
            residual_history.append(relative)
            absolute_residual_history.append(divergence_norm)
            transformed_fluctuation = plan.forward_displacement(fluctuation[1:-1, 1:-1])
            transformed_residual = plan.forward_displacement(residual[1:-1, 1:-1])
            spectral_energy = np.sum(np.abs(transformed_fluctuation) ** 2, axis=-1)
            total_spectral_energy = float(np.sum(spectral_energy))
            highest_mode_energy_history.append(float(spectral_energy[-1, -1]))
            high_frequency_fraction_history.append(
                float(
                    np.sum(spectral_energy[high_frequency_mask])
                    / max(total_spectral_energy, 1.0e-30)
                )
            )
            fluctuation_norm_history.append(float(np.linalg.norm(fluctuation)))
            highest_mode_residual_history.append(
                float(np.linalg.norm(transformed_residual[-1, -1]))
            )
            if relative <= config.relative_equilibrium_tolerance:
                solver_residual = relative
                elements.revert()
                verification_trial = elements.evaluate_samples(
                    sample_strain,
                    time_increment=time_increment,
                    consistent_tangent=True,
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
                    final_ebi_trial = verification_trial
                    final_applied = applied.copy()
                    iterations_per_increment.append(newton_iteration + 1)
                    converged = True
                    break
                trial = verification_trial
                residual = verification_force
                relative = verification_residual

            size = 2 * (grid.nx - 1) * (grid.ny - 1)

            def jacobian_action(
                vector: FloatArray, active_trial: EBIPlaneStressTrial = trial
            ) -> FloatArray:
                increment_field = unpack_interior(vector, grid)
                return pack_interior(
                    elements.tangent_action(
                        increment_field,
                        kinematics=kinematics,
                        trial=active_trial,
                    )
                )

            def preconditioner_action(vector: FloatArray) -> FloatArray:
                nodal = unpack_interior(vector, grid)
                transformed = plan.forward_displacement(nodal[1:-1, 1:-1])
                corrected = plan.embed_interior(plan.inverse_displacement(green.apply(transformed)))
                return pack_interior(corrected)

            jacobian = LinearOperator((size, size), matvec=jacobian_action, dtype=np.float64)
            preconditioner = LinearOperator(
                (size, size), matvec=preconditioner_action, dtype=np.float64
            )

            def count_gmres(_residual: object) -> None:
                nonlocal gmres_iterations
                gmres_iterations += 1

            correction, info = gmres(
                jacobian,
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
                raise RuntimeError(f"EBI GMRES failed with info={info}")
            direction = unpack_interior(correction, grid)
            accepted = False
            factor = 1.0
            for _ in range(config.maximum_line_search_reductions + 1):
                candidate = fluctuation + factor * direction
                candidate_strain = kinematics.strain_samples(applied + candidate)
                candidate_trial = elements.evaluate_samples(
                    candidate_strain,
                    time_increment=time_increment,
                    consistent_tangent=True,
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
                    line_search_factors.append(factor)
                    accepted = True
                    break
                factor *= 0.5
            if not accepted:
                elements.revert()
                raise RuntimeError("EBI Newton line search failed")
        if not converged:
            elements.revert()
            raise RuntimeError(f"EBI increment {increment} did not converge")

    if final_trial is None or final_ebi_trial is None:
        raise RuntimeError("no EBI increment converged")
    observables = {
        name: _reshape_mean_field(values, grid) for name, values in final_trial.observables.items()
    }
    diagnostics = Spectral2DDiagnostics(
        spatial_scheme="ebi_two_triangle",
        green_operator="b0",
        pixels=grid.pixel_shape,
        material_points=material.point_count,
        points_per_pixel=2,
        spacing_x=grid.spacing_x,
        spacing_y=grid.spacing_y,
        relative_residual_history=tuple(residual_history),
        dimensionless_equilibrium_history=tuple(residual_history),
        absolute_residual_history=tuple(absolute_residual_history),
        iterations_per_increment=tuple(iterations_per_increment),
        verification_residual=verification_residual,
        verification_residual_history=tuple(verification_history),
        verification_relative_mismatch_history=tuple(verification_mismatch_history),
        highest_mode_energy_history=tuple(highest_mode_energy_history),
        high_frequency_energy_fraction_history=tuple(high_frequency_fraction_history),
        fluctuation_norm_history=tuple(fluctuation_norm_history),
        highest_mode_residual_history=tuple(highest_mode_residual_history),
        reference_lambda_0=lambda_0,
        reference_mu_0=mu_0,
        reference_projection_error=projection_error,
        timings={
            "material_evaluations": float(material_evaluations),
            "gmres_iterations": float(gmres_iterations),
            "minimum_line_search_factor": min(line_search_factors, default=1.0),
        },
    )
    return Spectral2DResult(
        displacement=final_applied + fluctuation,
        applied_displacement=final_applied,
        fluctuation_displacement=fluctuation.copy(),
        strain_in_plane=final_ebi_trial.sample_strain,
        stress_in_plane_mpa=final_ebi_trial.sample_stress_mpa,
        full_stress_tensor_mpa=_reshape_mean_field(final_trial.full_stress_tensor_mpa, grid),
        full_strain_tensor=_reshape_mean_field(final_trial.full_strain_tensor, grid),
        elastic_strain_tensor=_reshape_mean_field(final_trial.elastic_strain_tensor, grid),
        plastic_strain_tensor=_reshape_mean_field(final_trial.plastic_strain_tensor, grid),
        observables=observables,
        reaction_forces=_boundary_reactions(
            kinematics.divergence_from_sample_stress(final_ebi_trial.sample_stress_mpa)
        ),
        diagnostics=diagnostics,
    )
