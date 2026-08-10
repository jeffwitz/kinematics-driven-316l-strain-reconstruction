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

from fem_inhouse.core.mfront_native import MFrontNativePlaneStressBatch
from fem_inhouse.core.nonlocal_plasticity import evaluate_nonlocal_fixed_point
from fem_inhouse.spectral2d.coupled_blocks import (
    CoupledBlockActions,
    make_dct_helmholtz_inverse,
    make_dst_b0_inverse,
)
from fem_inhouse.spectral2d.coupled_newton import (
    CoupledLinearisation,
    CoupledNewtonConfig,
    solve_coupled_newton,
)
from fem_inhouse.spectral2d.green import B0Green2D
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior
from fem_inhouse.spectral2d.transform_factory import create_full_dirichlet_dsti_plan
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        default=os.environ.get("MFRONT_BEHAVIOUR_LIBRARY"),
    )
    parser.add_argument("--nx", type=int, default=3)
    parser.add_argument("--ny", type=int, default=3)
    parser.add_argument("--length-scale", type=float, default=0.8)
    parser.add_argument("--time-increment", type=float, default=1.0)
    parser.add_argument("--strain-step", type=float, default=1.0e-7)
    parser.add_argument("--chi-step", type=float, default=1.0e-7)
    args = parser.parse_args()
    if args.library is None:
        parser.error("--library or MFRONT_BEHAVIOUR_LIBRARY is required")
    if args.nx < 2 or args.ny < 2:
        parser.error("nx and ny must be at least 2")

    grid = StructuredGrid2D(args.nx, args.ny, float(args.nx), float(args.ny))
    kinematics = TwoSubcellDiagnostic2D(grid)
    point_count = kinematics.material_point_count
    material = MFrontNativePlaneStressBatch(
        args.library,
        np.full(point_count, 250.0),
        np.full(point_count, 380.0),
        np.full(point_count, 0.245),
        behaviour_name="PixelMicromorphicLudwikJ2Plasticity",
        micromorphic_coupling_modulus_mpa=2_000.0,
    )
    plan = create_full_dirichlet_dsti_plan(grid, SpectralTransformConfig())
    green = B0Green2D(
        kinematics_reference_symbols(kinematics, plan),
        lambda_0=1.0,
        mu_0=1.0,
    )
    mechanical_inverse = make_dst_b0_inverse(plan, green)
    nonlocal_inverse = make_dct_helmholtz_inverse(
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

    def strain_from_mechanical(mechanical: np.ndarray) -> np.ndarray:
        full = boundary.copy()
        full[1:-1, 1:-1] += unpack_interior(mechanical, grid)[1:-1, 1:-1]
        return kinematics.strain_samples(full).reshape(-1, 3)

    def response(
        mechanical: np.ndarray, chi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        samples = strain_from_mechanical(mechanical)
        material.set_nonlocal_equivalent_plastic_strain(np.repeat(chi, 2))
        trial = material.evaluate_in_plane(
            samples,
            time_increment=args.time_increment,
            consistent_tangent=True,
        )
        stress = np.asarray(trial.stress_in_plane_mpa).reshape(grid.nx, grid.ny, 2, 3)
        source = np.asarray(trial.observables["equivalent_plastic_strain"]).reshape(
            grid.nx, grid.ny, 2
        ).mean(axis=2)
        nodal_force = kinematics.divergence(stress)
        mechanical_residual = pack_interior(nodal_force)
        filtered = nonlocal_inverse(source.reshape(-1)).reshape(grid.pixel_shape)
        nonlocal_residual = chi.reshape(grid.pixel_shape) - filtered
        tangent = np.asarray(trial.tangent_in_plane_mpa).reshape(
            grid.nx, grid.ny, 2, 3, 3
        )
        material.revert()
        return mechanical_residual, nonlocal_residual.reshape(-1), tangent

    def local_coupling_derivatives(
        mechanical: np.ndarray, chi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return pointwise derivatives needed by the off-diagonal blocks.

        The constitutive response is evaluated at the current trial state and
        reverted after every call.  Keeping these derivatives local avoids
        assembling dense global coupling matrices while retaining the exact
        TET2 and Helmholtz transfer operators in the block actions.
        """

        samples = strain_from_mechanical(mechanical)
        point_chi = np.repeat(chi, 2)
        chi_step = min(h_chi, 0.5 * float(np.min(chi)))
        if chi_step <= 0.0:
            raise ValueError("the local coupling probe requires strictly positive chi")

        def stress_source(
            chi_values: np.ndarray, strain_values: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            material.set_nonlocal_equivalent_plastic_strain(chi_values)
            trial = material.evaluate_in_plane(
                strain_values,
                time_increment=args.time_increment,
                consistent_tangent=False,
            )
            stress = np.asarray(trial.stress_in_plane_mpa).copy()
            source = np.asarray(
                trial.observables["equivalent_plastic_strain"]
            ).copy()
            material.revert()
            return stress, source

        stress_plus, source_plus = stress_source(point_chi + chi_step, samples)
        stress_minus, source_minus = stress_source(point_chi - chi_step, samples)
        dsigma_dchi = (stress_plus - stress_minus) / (2.0 * chi_step)
        dp_dchi = (source_plus - source_minus) / (2.0 * chi_step)

        dp_depsilon = np.empty((point_count, 3), dtype=float)
        for component in range(3):
            strain_plus = samples.copy()
            strain_minus = samples.copy()
            strain_plus[:, component] += h_u
            strain_minus[:, component] -= h_u
            _, source_plus = stress_source(point_chi, strain_plus)
            _, source_minus = stress_source(point_chi, strain_minus)
            dp_depsilon[:, component] = (source_plus - source_minus) / (2.0 * h_u)

        return dsigma_dchi, dp_depsilon, dp_dchi

    def evaluate(state: tuple[np.ndarray, np.ndarray]) -> CoupledLinearisation:
        mechanical, chi = state
        ru, g, tangent = response(mechanical, chi)
        nu = mechanical.size
        nc = chi.size
        dsigma_dchi, dp_depsilon, dp_dchi = local_coupling_derivatives(
            mechanical, chi
        )

        def ruu_action(value: np.ndarray) -> np.ndarray:
            displacement = unpack_interior(value, grid)
            strain_increment = kinematics.strain_samples(displacement).reshape(-1, 3)
            stress_increment = np.einsum(
                "...ij,...j->...i", tangent, strain_increment.reshape(grid.nx, grid.ny, 2, 3)
            )
            return pack_interior(kinematics.divergence(stress_increment))

        def ruchi_action(value: np.ndarray) -> np.ndarray:
            point_value = np.repeat(value, 2)
            stress_increment = (dsigma_dchi * point_value[:, None]).reshape(
                grid.nx, grid.ny, 2, 3
            )
            return pack_interior(kinematics.divergence(stress_increment))

        def g_u_action(value: np.ndarray) -> np.ndarray:
            displacement = unpack_interior(value, grid)
            strain_increment = kinematics.strain_samples(displacement).reshape(-1, 3)
            source_increment = np.einsum(
                "pi,pi->p", dp_depsilon, strain_increment
            ).reshape(grid.nx, grid.ny, 2).mean(axis=2)
            return -nonlocal_inverse(source_increment.reshape(-1))

        def g_chi_action(value: np.ndarray) -> np.ndarray:
            source_increment = (
                dp_dchi.reshape(grid.nx, grid.ny, 2).mean(axis=2)
                * value.reshape(grid.pixel_shape)
            )
            return value - nonlocal_inverse(source_increment.reshape(-1))

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
            ),
        )

    coupled_start = time.perf_counter()
    result = solve_coupled_newton(
        initial_mechanical,
        initial_nonlocal,
        evaluate,
        config=CoupledNewtonConfig(maximum_iterations=8),
    )
    coupled_elapsed = time.perf_counter() - coupled_start
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
        "minimum_chi": float(np.min(result.nonlocal_field)),
        "maximum_chi": float(np.max(result.nonlocal_field)),
        "partitioned_iterations": partitioned.iterations,
        "partitioned_relative_residual": partitioned.relative_residual,
        "coupled_vs_partitioned_chi_linf": float(np.max(np.abs(chi_difference))),
        "coupled_vs_partitioned_chi_l2": float(np.linalg.norm(chi_difference)),
    }, indent=2))


def kinematics_reference_symbols(kinematics: TwoSubcellDiagnostic2D, plan: object) -> object:
    """Keep the pilot's symbol construction explicit and local."""

    return kinematics.reference_operator_symbols(plan)


if __name__ == "__main__":
    main()
