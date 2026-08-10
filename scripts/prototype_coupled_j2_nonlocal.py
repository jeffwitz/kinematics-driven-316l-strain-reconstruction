#!/usr/bin/env python3
"""Run a small real J2 micromorphic ``(u, chi)`` Newton pilot.

The pilot uses the existing TET2 kinematics and MFront micromorphic law.  The
mechanical/non-local diagonal blocks use the actual DST-I/B0 and
DCT-II/Helmholtz inverses, while the constitutive coupling blocks are built
from pointwise material derivatives and applied matrix-free.  Production
partitioned mechanics is untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator

from fem_inhouse.core.constitutive_sensitivities import (
    finite_difference_sensitivities,
)
from fem_inhouse.core.mfront_native import MFrontNativePlaneStressBatch
from fem_inhouse.core.nonlocal_plasticity import evaluate_nonlocal_fixed_point
from fem_inhouse.core.plane_stress_material import InPlaneConstitutiveTrial
from fem_inhouse.spectral2d.coupled_blocks import (
    CoupledBlockActions,
    make_dct_helmholtz_inverse,
    make_dct_helmholtz_operator,
    make_dst_b0_inverse,
)
from fem_inhouse.spectral2d.coupled_newton import (
    CoupledLinearisation,
    CoupledNewtonConfig,
    solve_coupled_newton,
)
from fem_inhouse.spectral2d.green import B0Green2D, project_isotropic_plane_stress_tangent
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.spectral2d.krylov import solve_nonsymmetric_krylov
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior
from fem_inhouse.spectral2d.transform_factory import create_full_dirichlet_dsti_plan
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig


class GenericStructuralMicromorphicBatch:
    """Validation adapter for the generic structural MFront probe.

    This is deliberately local to the benchmark script.  It proves the M20
    cost of the four MFront tangent blocks before moving the adapter into the
    production bridge.
    """

    def __init__(self, library: Path, point_count: int) -> None:
        import mgis.behaviour as mgis

        self._mgis = mgis
        behaviour = mgis.load(
            str(library),
            "MicromorphicJ2GenericStructuralPlaneStressProbe",
            mgis.Hypothesis.Tridimensional,
        )
        # Keep the Python behaviour wrapper alive for the lifetime of the
        # manager.  MGIS's manager refers to the underlying behaviour object;
        # letting this local go out of scope can leave a dangling handle.
        self._behaviour = behaviour
        self._manager = mgis.MaterialDataManager(behaviour, point_count)
        properties = {
            "YoungModulus": 205.0e3,
            "PoissonRatio": 0.3,
            "InitialYieldStress": 250.0,
            "HardeningCoefficient": 380.0,
            "HardeningExponent": 0.245,
            "MicromorphicCouplingModulus": 2.0e3,
        }
        for state in (self._manager.s0, self._manager.s1):
            for name, value in properties.items():
                mgis.setMaterialProperty(state, name, value)
            mgis.setExternalStateVariable(state, "Temperature", 293.15)
        self._chi = np.zeros(point_count)
        self._has_trial = False

    @property
    def point_count(self) -> int:
        return self._manager.n

    def set_nonlocal_equivalent_plastic_strain(self, values: np.ndarray) -> None:
        supplied = np.asarray(values, dtype=float)
        if supplied.shape != (self.point_count,):
            raise ValueError("generic probe chi has an unexpected shape")
        if self._has_trial:
            self.revert()
        self._chi[:] = supplied

    def evaluate_in_plane(
        self,
        in_plane_strain: np.ndarray,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> InPlaneConstitutiveTrial:
        strain = np.asarray(in_plane_strain, dtype=float)
        if strain.shape != (self.point_count, 3):
            raise ValueError("generic probe strain has an unexpected shape")
        if self._has_trial:
            self.revert()
        gradients = np.zeros((self.point_count, 7), dtype=float)
        gradients[:, 0] = strain[:, 0]
        gradients[:, 1] = strain[:, 1]
        gradients[:, 3] = strain[:, 2] / np.sqrt(2.0)
        gradients[:, 6] = self._chi
        self._manager.s1.gradients[:, :] = gradients
        integration_type = (
            self._mgis.IntegrationType.IntegrationWithConsistentTangentOperator
            if consistent_tangent
            else self._mgis.IntegrationType.IntegrationWithoutTangentOperator
        )
        status = self._mgis.integrate(
            self._manager, integration_type, float(time_increment), 0, self.point_count
        )
        if status != 1:
            self.revert()
            raise RuntimeError(f"generic structural probe integration failed: {status}")
        self._has_trial = True
        forces = np.asarray(self._manager.s1.thermodynamic_forces)
        stress = forces[:, [0, 1, 3]].copy()
        stress[:, 2] /= np.sqrt(2.0)
        tangent = None
        dsigma_dchi = None
        dp_depsilon = None
        dp_dchi = None
        if consistent_tangent:
            # MGIS stores declared tangent blocks consecutively, not as one
            # row-major 7x7 matrix: 6x6, 6x1, 1x6, 1x1.
            blocks = np.asarray(self._manager.K)
            mechanical_block = blocks[:, :36].reshape(self.point_count, 6, 6)
            stress_chi_block = blocks[:, 36:42]
            source_strain_block = blocks[:, 42:48]
            source_chi_block = blocks[:, 48]
            tangent = mechanical_block[:, [0, 1, 3]][:, :, [0, 1, 3]].copy()
            tangent[:, 2, :] /= np.sqrt(2.0)
            tangent[:, :, 2] /= np.sqrt(2.0)
            dsigma_dchi = stress_chi_block[:, [0, 1, 3]].copy()
            dsigma_dchi[:, 2] /= np.sqrt(2.0)
            dp_depsilon = source_strain_block[:, [0, 1, 3]].copy()
            dp_depsilon[:, 2] /= np.sqrt(2.0)
            dp_dchi = source_chi_block.copy()
        observables = {
            "equivalent_plastic_strain": forces[:, 6].copy(),
            "generic_dsigma_dchi": dsigma_dchi,
            "generic_dp_depsilon": dp_depsilon,
            "generic_dp_dchi": dp_dchi,
        }
        return InPlaneConstitutiveTrial(
            stress_in_plane_mpa=stress,
            tangent_in_plane_mpa=tangent,
            observables=observables,
        )

    def revert(self) -> None:
        self._mgis.revert(self._manager)
        self._has_trial = False

    def commit(self) -> None:
        self._mgis.update(self._manager)
        self._has_trial = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        default=os.environ.get("MFRONT_BEHAVIOUR_LIBRARY"),
    )
    parser.add_argument(
        "--generic-library",
        type=Path,
        default=os.environ.get("MFRONT_GENERIC_BEHAVIOUR_LIBRARY"),
        help="validation-only generic structural MFront library",
    )
    parser.add_argument("--nx", type=int, default=3)
    parser.add_argument("--ny", type=int, default=3)
    parser.add_argument("--length-scale", type=float, default=0.8)
    parser.add_argument("--time-increment", type=float, default=1.0)
    parser.add_argument("--strain-step", type=float, default=1.0e-7)
    parser.add_argument("--chi-step", type=float, default=1.0e-7)
    parser.add_argument(
        "--krylov-relative-tolerance", type=float, default=1.0e-10
    )
    parser.add_argument(
        "--central-strain-coupling",
        action="store_true",
        help="use central instead of forward finite differences for dp/depsilon",
    )
    args = parser.parse_args()
    if args.library is None:
        parser.error("--library or MFRONT_BEHAVIOUR_LIBRARY is required")
    if args.nx < 2 or args.ny < 2:
        parser.error("nx and ny must be at least 2")

    grid = StructuredGrid2D(args.nx, args.ny, float(args.nx), float(args.ny))
    kinematics = TwoSubcellDiagnostic2D(grid)
    point_count = kinematics.material_point_count
    generic_mode = args.generic_library is not None
    if generic_mode:
        material = GenericStructuralMicromorphicBatch(args.generic_library, point_count)
    else:
        material = MFrontNativePlaneStressBatch(
            args.library,
            np.full(point_count, 250.0),
            np.full(point_count, 380.0),
            np.full(point_count, 0.245),
            behaviour_name="PixelMicromorphicLudwikJ2Plasticity",
            micromorphic_coupling_modulus_mpa=2_000.0,
        )
    # Calibrate the elastic spectral preconditioner from the actual virgin
    # plane-stress tangent instead of the unit test values used by the first
    # coupled pilot.
    virgin_trial = material.evaluate_in_plane(
        np.zeros((point_count, 3)),
        time_increment=args.time_increment,
        consistent_tangent=True,
    )
    virgin_tangent = np.asarray(virgin_trial.tangent_in_plane_mpa).reshape(
        grid.nx, grid.ny, 2, 3, 3
    )
    material.revert()
    lambda_0, mu_0, b0_projection_error = project_isotropic_plane_stress_tangent(
        virgin_tangent.mean(axis=(0, 1, 2))
    )
    plan = create_full_dirichlet_dsti_plan(grid, SpectralTransformConfig())
    green = B0Green2D(
        kinematics_reference_symbols(kinematics, plan),
        lambda_0=lambda_0,
        mu_0=mu_0,
    )
    mechanical_inverse = make_dst_b0_inverse(plan, green)
    nonlocal_inverse = make_dct_helmholtz_inverse(
        grid.pixel_shape,
        length_scale=args.length_scale,
        spacing_x=grid.spacing_x,
        spacing_y=grid.spacing_y,
    )
    nonlocal_operator = make_dct_helmholtz_operator(
        grid.pixel_shape,
        length_scale=args.length_scale,
        spacing_x=grid.spacing_x,
        spacing_y=grid.spacing_y,
    )

    boundary = np.zeros((*grid.node_shape, 2), dtype=float)
    x, y = grid.coordinates
    boundary[..., 0] = 0.008 * x[:, None] + 0.001 * y[None, :]
    boundary[..., 1] = 0.0003 * y[None, :] + 0.001 * x[:, None]
    initial_mechanical = np.zeros(2 * grid.interior_shape[0] * grid.interior_shape[1])
    initial_nonlocal = np.full(grid.nx * grid.ny, 1.0e-3)
    h_u = args.strain_step * max(grid.spacing_x, grid.spacing_y)
    h_chi = args.chi_step
    coupled_material_seconds = 0.0
    coupled_coupling_derivative_seconds = 0.0
    coupled_material_evaluations = 0
    coupled_coupling_probe_evaluations = 0
    coupled_response_evaluations = 0
    coupled_chi_derivative_seconds = 0.0
    coupled_strain_derivative_seconds = 0.0
    coupled_ruchi_action_seconds = 0.0
    coupled_g_u_action_seconds = 0.0
    coupled_g_chi_action_seconds = 0.0
    coupled_ruchi_action_calls = 0
    coupled_g_u_action_calls = 0
    coupled_g_chi_action_calls = 0

    def strain_from_mechanical(mechanical: np.ndarray) -> np.ndarray:
        full = boundary.copy()
        full[1:-1, 1:-1] += unpack_interior(mechanical, grid)[1:-1, 1:-1]
        return kinematics.strain_samples(full).reshape(-1, 3)

    def response(
        mechanical: np.ndarray, chi: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    ]:
        nonlocal coupled_material_seconds, coupled_material_evaluations
        nonlocal coupled_response_evaluations
        samples = strain_from_mechanical(mechanical)
        material.set_nonlocal_equivalent_plastic_strain(np.repeat(chi, 2))
        material_start = time.perf_counter()
        trial = material.evaluate_in_plane(
            samples,
            time_increment=args.time_increment,
            consistent_tangent=True,
        )
        coupled_material_seconds += time.perf_counter() - material_start
        coupled_material_evaluations += 1
        coupled_response_evaluations += 1
        stress_point = np.asarray(trial.stress_in_plane_mpa).copy()
        stress = stress_point.reshape(grid.nx, grid.ny, 2, 3)
        source = np.asarray(trial.observables["equivalent_plastic_strain"]).reshape(
            grid.nx, grid.ny, 2
        ).mean(axis=2)
        source_point = np.asarray(
            trial.observables["equivalent_plastic_strain"]
        ).copy()
        nodal_force = kinematics.divergence(stress)
        mechanical_residual = pack_interior(nodal_force)
        # Equivalent residual after left multiplication by the invertible
        # Helmholtz operator: H chi - p = 0.  H^{-1} remains the spectral
        # preconditioner, but is no longer applied to every residual/matvec.
        nonlocal_residual = nonlocal_operator(chi) - source.reshape(-1)
        tangent = np.asarray(trial.tangent_in_plane_mpa).reshape(
            grid.nx, grid.ny, 2, 3, 3
        )
        generic_couplings = None
        if generic_mode:
            generic_couplings = (
                np.asarray(trial.observables["generic_dsigma_dchi"]).copy(),
                np.asarray(trial.observables["generic_dp_depsilon"]).copy(),
                np.asarray(trial.observables["generic_dp_dchi"]).copy(),
            )
        material.revert()
        return (
            mechanical_residual,
            nonlocal_residual.reshape(-1),
            tangent,
            source_point,
            stress_point,
            generic_couplings,
        )

    cached_state: tuple[np.ndarray, np.ndarray] | None = None
    cached_response: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    ] | None = None

    def response_for_state(
        state: tuple[np.ndarray, np.ndarray]
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    ]:
        nonlocal cached_state, cached_response
        if (
            cached_state is None
            or not np.array_equal(state[0], cached_state[0])
            or not np.array_equal(state[1], cached_state[1])
        ):
            cached_state = (state[0].copy(), state[1].copy())
            cached_response = response(*state)
        assert cached_response is not None
        return cached_response

    def residual_only(
        state: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate only stress/source for convergence checks."""

        mechanical, chi = state
        samples = strain_from_mechanical(mechanical)
        material.set_nonlocal_equivalent_plastic_strain(np.repeat(chi, 2))
        trial = material.evaluate_in_plane(
            samples,
            time_increment=args.time_increment,
            consistent_tangent=False,
        )
        stress = np.asarray(trial.stress_in_plane_mpa).reshape(
            grid.nx, grid.ny, 2, 3
        )
        source = np.asarray(
            trial.observables["equivalent_plastic_strain"]
        ).reshape(grid.nx, grid.ny, 2).mean(axis=2)
        material.revert()
        return pack_interior(kinematics.divergence(stress)), (
            nonlocal_operator(chi) - source.reshape(-1)
        )

    def residual(
        state: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        current_response = response_for_state(state)
        return current_response[0], current_response[1]

    def local_coupling_derivatives(
        mechanical: np.ndarray,
        chi: np.ndarray,
        base_source: np.ndarray,
        base_stress: np.ndarray,
        generic_couplings: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return generic pointwise FD sensitivities for the current law.

        The adapter only sees stress and the selected scalar observable.  It
        therefore remains usable for crystal laws whose local unknowns and
        state-variable layout differ completely from J2.  The generic local
        implicit-block contract is validated separately and can replace this
        oracle once MFront exports ``F_z, F_q, y_z, y_q``.
        """

        nonlocal coupled_material_seconds, coupled_material_evaluations
        nonlocal coupled_coupling_probe_evaluations
        nonlocal coupled_chi_derivative_seconds, coupled_strain_derivative_seconds
        if generic_couplings is not None:
            return generic_couplings
        samples = strain_from_mechanical(mechanical)
        point_chi = np.repeat(chi, 2)
        chi_step = min(h_chi, 0.5 * float(np.min(chi)))
        if chi_step <= 0.0:
            raise ValueError("the local coupling probe requires strictly positive chi")

        def stress_source(
            chi_values: np.ndarray, strain_values: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            nonlocal coupled_material_seconds, coupled_material_evaluations
            nonlocal coupled_coupling_probe_evaluations
            material.set_nonlocal_equivalent_plastic_strain(chi_values)
            material_start = time.perf_counter()
            trial = material.evaluate_in_plane(
                strain_values,
                time_increment=args.time_increment,
                consistent_tangent=False,
            )
            coupled_material_seconds += time.perf_counter() - material_start
            coupled_material_evaluations += 1
            coupled_coupling_probe_evaluations += 1
            stress = np.asarray(trial.stress_in_plane_mpa).copy()
            source = np.asarray(
                trial.observables["equivalent_plastic_strain"]
            ).copy()
            material.revert()
            return stress, source

        def constitutive_response(
            strain_values: np.ndarray, chi_values: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            return stress_source(chi_values, strain_values)

        parameter = point_chi
        sensitivity = finite_difference_sensitivities(
            constitutive_response,
            samples,
            parameter,
            base_stress=base_stress,
            base_observable=base_source,
            strain_step=h_u,
            parameter_step=chi_step,
            central_parameter=True,
            forward_strain=not args.central_strain_coupling,
        )
        coupled_chi_derivative_seconds += sensitivity.parameter_seconds
        coupled_strain_derivative_seconds += sensitivity.strain_seconds
        return (
            sensitivity.stress_parameter,
            sensitivity.observable_strain,
            sensitivity.observable_parameter,
        )

    def evaluate(state: tuple[np.ndarray, np.ndarray]) -> CoupledLinearisation:
        nonlocal coupled_coupling_derivative_seconds
        mechanical, chi = state
        ru, g, tangent, base_source, base_stress, generic_couplings = response_for_state(state)
        nu = mechanical.size
        nc = chi.size
        coupling_derivative_start = time.perf_counter()
        dsigma_dchi, dp_depsilon, dp_dchi = local_coupling_derivatives(
            mechanical, chi, base_source, base_stress, generic_couplings
        )
        coupled_coupling_derivative_seconds += (
            time.perf_counter() - coupling_derivative_start
        )

        def ruu_action(value: np.ndarray) -> np.ndarray:
            displacement = unpack_interior(value, grid)
            strain_increment = kinematics.strain_samples(displacement).reshape(-1, 3)
            stress_increment = np.einsum(
                "...ij,...j->...i", tangent, strain_increment.reshape(grid.nx, grid.ny, 2, 3)
            )
            return pack_interior(kinematics.divergence(stress_increment))

        def ruchi_action(value: np.ndarray) -> np.ndarray:
            nonlocal coupled_ruchi_action_seconds, coupled_ruchi_action_calls
            action_start = time.perf_counter()
            point_value = np.repeat(value, 2)
            stress_increment = (dsigma_dchi * point_value[:, None]).reshape(
                grid.nx, grid.ny, 2, 3
            )
            result = pack_interior(kinematics.divergence(stress_increment))
            coupled_ruchi_action_seconds += time.perf_counter() - action_start
            coupled_ruchi_action_calls += 1
            return result

        def g_u_action(value: np.ndarray) -> np.ndarray:
            nonlocal coupled_g_u_action_seconds, coupled_g_u_action_calls
            action_start = time.perf_counter()
            displacement = unpack_interior(value, grid)
            strain_increment = kinematics.strain_samples(displacement).reshape(-1, 3)
            source_increment = np.einsum(
                "pi,pi->p", dp_depsilon, strain_increment
            ).reshape(grid.nx, grid.ny, 2).mean(axis=2)
            result = -source_increment.reshape(-1)
            coupled_g_u_action_seconds += time.perf_counter() - action_start
            coupled_g_u_action_calls += 1
            return result

        def g_chi_action(value: np.ndarray) -> np.ndarray:
            nonlocal coupled_g_chi_action_seconds, coupled_g_chi_action_calls
            action_start = time.perf_counter()
            source_increment = (
                dp_dchi.reshape(grid.nx, grid.ny, 2).mean(axis=2)
                * value.reshape(grid.pixel_shape)
            )
            result = nonlocal_operator(value) - source_increment.reshape(-1)
            coupled_g_chi_action_seconds += time.perf_counter() - action_start
            coupled_g_chi_action_calls += 1
            return result

        def combined_action(
            displacement_value: np.ndarray, chi_value: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            """Apply all four blocks while sharing the strain increment."""

            displacement = unpack_interior(displacement_value, grid)
            strain_increment = kinematics.strain_samples(displacement).reshape(
                grid.nx, grid.ny, 2, 3
            )
            chi_increment = chi_value.reshape(grid.nx, grid.ny, 1, 1)
            stress_increment = np.einsum(
                "...ij,...j->...i", tangent, strain_increment
            ) + dsigma_dchi.reshape(grid.nx, grid.ny, 2, 3) * chi_increment
            source_increment = (
                np.einsum(
                    "...i,...i->...",
                    dp_depsilon.reshape(grid.nx, grid.ny, 2, 3),
                    strain_increment,
                )
                + dp_dchi.reshape(grid.nx, grid.ny, 2) * chi_value.reshape(
                    grid.nx, grid.ny, 1
                )
            ).mean(axis=2)
            upper = pack_interior(kinematics.divergence(stress_increment))
            lower = nonlocal_operator(chi_value) - source_increment.reshape(-1)
            return upper, lower

        return CoupledLinearisation(
            mechanical_residual=ru,
            nonlocal_residual=g,
            actions=CoupledBlockActions(
                mechanical_size=nu,
                nonlocal_size=nc,
                ruu=ruu_action,
                ruchi=ruchi_action,
                g_u=g_u_action,
                g_chi=g_chi_action,
                mechanical_inverse=mechanical_inverse,
                nonlocal_inverse=nonlocal_inverse,
                combined=combined_action,
            ),
        )

    coupled_start = time.perf_counter()
    result = solve_coupled_newton(
        initial_mechanical,
        initial_nonlocal,
        evaluate,
        config=CoupledNewtonConfig(
            maximum_iterations=8,
            krylov_relative_tolerance=args.krylov_relative_tolerance,
        ),
        evaluate_residual=residual_only,
    )
    coupled_elapsed = time.perf_counter() - coupled_start

    # Honest staggered reference: every outer non-local iteration resolves the
    # mechanical equilibrium again at fixed chi.  Intermediate constitutive
    # trials are reverted; only the final converged trial is committed.
    if generic_mode:
        staggered_material = GenericStructuralMicromorphicBatch(
            args.generic_library, point_count
        )
    else:
        staggered_material = MFrontNativePlaneStressBatch(
            args.library,
            np.full(point_count, 250.0),
            np.full(point_count, 380.0),
            np.full(point_count, 0.245),
            behaviour_name="PixelMicromorphicLudwikJ2Plasticity",
            micromorphic_coupling_modulus_mpa=2_000.0,
        )

    def staggered_response(
        trial_material: object,
        mechanical: np.ndarray,
        chi: np.ndarray,
        *,
        consistent_tangent: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        trial_material.set_nonlocal_equivalent_plastic_strain(np.repeat(chi, 2))
        trial = trial_material.evaluate_in_plane(
            strain_from_mechanical(mechanical),
            time_increment=args.time_increment,
            consistent_tangent=consistent_tangent,
        )
        stress = np.asarray(trial.stress_in_plane_mpa).reshape(
            grid.nx, grid.ny, 2, 3
        )
        residual = pack_interior(kinematics.divergence(stress))
        source = np.asarray(
            trial.observables["equivalent_plastic_strain"]
        ).reshape(grid.nx, grid.ny, 2).mean(axis=2)
        tangent = (
            None
            if not consistent_tangent
            else np.asarray(trial.tangent_in_plane_mpa).reshape(
                grid.nx, grid.ny, 2, 3, 3
            )
        )
        trial_material.revert()
        return residual, source.reshape(-1), tangent

    staggered_start = time.perf_counter()
    staggered_mechanical = initial_mechanical.copy()
    staggered_nonlocal = initial_nonlocal.copy()
    staggered_outer_iterations = 0
    staggered_mechanical_iterations = 0
    staggered_krylov_iterations = 0
    staggered_mechanical_residual = float("inf")
    staggered_nonlocal_residual = float("inf")
    for outer in range(20):
        staggered_outer_iterations = outer + 1
        for _ in range(8):
            ru_st, source_st, tangent_st = staggered_response(
                staggered_material,
                staggered_mechanical,
                staggered_nonlocal,
                consistent_tangent=False,
            )
            staggered_mechanical_residual = float(np.linalg.norm(ru_st))
            if staggered_mechanical_residual <= 1.0e-10:
                break
            _, source_st, tangent_st = staggered_response(
                staggered_material,
                staggered_mechanical,
                staggered_nonlocal,
                consistent_tangent=True,
            )
            assert tangent_st is not None

            def mechanical_matvec(
                value: np.ndarray, tangent_reference: np.ndarray = tangent_st
            ) -> np.ndarray:
                displacement = unpack_interior(value, grid)
                strain_increment = kinematics.strain_samples(displacement).reshape(
                    grid.nx, grid.ny, 2, 3
                )
                stress_increment = np.einsum(
                    "...ij,...j->...i", tangent_reference, strain_increment
                )
                return pack_interior(kinematics.divergence(stress_increment))

            mechanical_operator = LinearOperator(
                (initial_mechanical.size, initial_mechanical.size),
                matvec=mechanical_matvec,
                dtype=np.float64,
            )
            correction, info, calls = solve_nonsymmetric_krylov(
                mechanical_operator,
                -ru_st,
                preconditioner=LinearOperator(
                    (initial_mechanical.size, initial_mechanical.size),
                    matvec=mechanical_inverse,
                    dtype=np.float64,
                ),
                method="gmres",
                rtol=args.krylov_relative_tolerance,
                maximum_iterations=200,
                restart=100,
            )
            if info != 0:
                raise RuntimeError(f"staggered GMRES failed with info={info}")
            staggered_mechanical += correction
            staggered_mechanical_iterations += 1
            staggered_krylov_iterations += calls

        updated_nonlocal = nonlocal_inverse(source_st)
        staggered_nonlocal_residual = float(
            np.linalg.norm(updated_nonlocal - staggered_nonlocal)
        )
        staggered_nonlocal = updated_nonlocal
        if (
            staggered_mechanical_residual <= 1.0e-10
            and staggered_nonlocal_residual <= 1.0e-10
        ):
            break

    # Commit the final staggered constitutive trial only.
    staggered_material.set_nonlocal_equivalent_plastic_strain(
        np.repeat(staggered_nonlocal, 2)
    )
    staggered_material.evaluate_in_plane(
        strain_from_mechanical(staggered_mechanical),
        time_increment=args.time_increment,
        consistent_tangent=False,
    )
    staggered_material.commit()
    staggered_elapsed = time.perf_counter() - staggered_start
    partitioned_material = MFrontNativePlaneStressBatch(
        args.library,
        np.full(point_count, 250.0),
        np.full(point_count, 380.0),
        np.full(point_count, 0.245),
        behaviour_name="PixelMicromorphicLudwikJ2Plasticity",
        micromorphic_coupling_modulus_mpa=2_000.0,
    )
    partitioned_start = time.perf_counter()
    partitioned = evaluate_nonlocal_fixed_point(
        partitioned_material,
        strain_from_mechanical(result.mechanical),
        time_increment=args.time_increment,
        element_shape=grid.pixel_shape,
        gauss_points_per_element=2,
        initial_nonlocal_peeq=initial_nonlocal.reshape(grid.pixel_shape),
        length_scale_mm=args.length_scale,
        spacing_x_mm=grid.spacing_x,
        spacing_y_mm=grid.spacing_y,
        coupling_modulus_mpa=2_000.0,
        relaxation=0.5,
        relaxation_strategy="aitken",
        minimum_relaxation=0.05,
        maximum_relaxation=0.8,
        relative_tolerance=1.0e-10,
        maximum_iterations=50,
        maximum_helmholtz_residual=1.0e-10,
    )
    partitioned_elapsed = time.perf_counter() - partitioned_start
    chi_difference = partitioned.nonlocal_peeq.reshape(-1) - result.nonlocal_field
    print(json.dumps({
        "grid": [args.nx, args.ny],
        "converged": result.converged,
        "newton_iterations": result.iterations,
        "initial_residual_norm": result.initial_residual_norm,
        "final_residual_norm": result.final_residual_norm,
        "krylov_iterations": list(result.krylov_iterations),
        "coupled_elapsed_seconds": coupled_elapsed,
        "partitioned_elapsed_seconds": partitioned_elapsed,
        "coupled_minus_partitioned_seconds": coupled_elapsed - partitioned_elapsed,
        "coupled_over_partitioned": coupled_elapsed / partitioned_elapsed,
        "b0_lambda": lambda_0,
        "b0_mu": mu_0,
        "b0_projection_error": b0_projection_error,
        "nonlocal_residual_form": "H_chi_minus_p",
        "coupled_material_seconds": coupled_material_seconds,
        "coupled_coupling_derivative_seconds": coupled_coupling_derivative_seconds,
        "coupled_material_evaluations": coupled_material_evaluations,
        "coupled_coupling_probe_evaluations": coupled_coupling_probe_evaluations,
        "coupled_response_evaluations": coupled_response_evaluations,
        "coupled_chi_derivative_seconds": coupled_chi_derivative_seconds,
        "coupled_strain_derivative_seconds": coupled_strain_derivative_seconds,
        "coupled_ruchi_action_seconds": coupled_ruchi_action_seconds,
        "coupled_g_u_action_seconds": coupled_g_u_action_seconds,
        "coupled_g_chi_action_seconds": coupled_g_chi_action_seconds,
        "coupled_ruchi_action_calls": coupled_ruchi_action_calls,
        "coupled_g_u_action_calls": coupled_g_u_action_calls,
        "coupled_g_chi_action_calls": coupled_g_chi_action_calls,
        "minimum_chi": float(np.min(result.nonlocal_field)),
        "maximum_chi": float(np.max(result.nonlocal_field)),
        "partitioned_iterations": partitioned.iterations,
        "partitioned_relative_residual": partitioned.relative_residual,
        "coupled_vs_partitioned_chi_linf": float(np.max(np.abs(chi_difference))),
        "coupled_vs_partitioned_chi_l2": float(np.linalg.norm(chi_difference)),
        "staggered_elapsed_seconds": staggered_elapsed,
        "staggered_outer_iterations": staggered_outer_iterations,
        "staggered_mechanical_iterations": staggered_mechanical_iterations,
        "staggered_krylov_iterations": staggered_krylov_iterations,
        "staggered_mechanical_residual": staggered_mechanical_residual,
        "staggered_nonlocal_residual": staggered_nonlocal_residual,
        "staggered_converged": (
            staggered_mechanical_residual <= 1.0e-10
            and staggered_nonlocal_residual <= 1.0e-10
        ),
        "coupled_vs_staggered_chi_linf": float(
            np.max(np.abs(staggered_nonlocal - result.nonlocal_field))
        ),
    }, indent=2))


def kinematics_reference_symbols(kinematics: TwoSubcellDiagnostic2D, plan: object) -> object:
    """Keep the pilot's symbol construction explicit and local."""

    return kinematics.reference_operator_symbols(plan)


if __name__ == "__main__":
    main()
