#!/usr/bin/env python3
"""Compare monolithic and staggered generic micromorphic J2 on P43.

This driver uses the registered P43 DIC boundary history and the spatial J2
maps used by the TRI2 benchmark.  It deliberately keeps the generic MFront
adapter and the coupled matrix-free operators from the prototype separate
from the production crystal-plasticity drivers.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
from benchmark_tri2_j2_krylov import DATA_ROOT, DEFAULT_CROP, PIXEL_SIZE_MM
from prototype_coupled_j2_nonlocal import GenericStructuralMicromorphicBatch
from scipy.sparse.linalg import LinearOperator

from fem_inhouse.core.constitutive_sensitivities import finite_difference_sensitivities
from fem_inhouse.core.mfront_native import MFrontNativePlaneStressBatch
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


def _load_p43(
    crop: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x0, x1, y0, y1 = crop
    ux = np.load(DATA_ROOT / "displacement_x_mm.npy", mmap_mode="r")
    uy = np.load(DATA_ROOT / "displacement_y_mm.npy", mmap_mode="r")
    yield_stress = np.load(DATA_ROOT / "yield_stress_mpa.npy", mmap_mode="r")
    hardening = np.load(DATA_ROOT / "hardening_coefficient_mpa.npy", mmap_mode="r")
    boundary = np.stack((ux[x0 : x1 + 1, y0 : y1 + 1], uy[x0 : x1 + 1, y0 : y1 + 1]), axis=-1)
    history = np.stack([fraction * boundary for fraction in np.linspace(0.0, 1.0, 9)])
    return (
        history,
        np.asarray(yield_stress[x0:x1, y0:y1]).reshape(-1),
        np.asarray(hardening[x0:x1, y0:y1]).reshape(-1),
        boundary,
    )


class _ChunkedGenericMaterial:
    """Integrate P43 in small MGIS batches to isolate local failures."""

    def __init__(
        self,
        library: Path,
        yield_stress: np.ndarray,
        hardening: np.ndarray,
        chunk_size: int = 20,
    ) -> None:
        self._point_count = yield_stress.size
        self._chunks = []
        for start in range(0, self._point_count, chunk_size):
            stop = min(start + chunk_size, self._point_count)
            self._chunks.append(
                GenericStructuralMicromorphicBatch(
                    library,
                    stop - start,
                    yield_stress=yield_stress[start:stop],
                    hardening_coefficient=hardening[start:stop],
                )
            )

    @property
    def point_count(self) -> int:
        return self._point_count

    def set_nonlocal_equivalent_plastic_strain(self, values: np.ndarray) -> None:
        offset = 0
        for chunk in self._chunks:
            stop = offset + chunk.point_count
            chunk.set_nonlocal_equivalent_plastic_strain(values[offset:stop])
            offset = stop

    def evaluate_in_plane(
        self,
        strain: np.ndarray,
        *,
        time_increment: float,
        consistent_tangent: bool,
    ) -> InPlaneConstitutiveTrial:
        trials = []
        offset = 0
        for chunk in self._chunks:
            stop = offset + chunk.point_count
            trials.append(
                chunk.evaluate_in_plane(
                    strain[offset:stop],
                    time_increment=time_increment,
                    consistent_tangent=consistent_tangent,
                )
            )
            offset = stop
        stress = np.concatenate([trial.stress_in_plane_mpa for trial in trials])
        tangents = (
            None
            if not consistent_tangent
            else np.concatenate([trial.tangent_in_plane_mpa for trial in trials])
        )
        keys = trials[0].observables
        observables = {
            key: (
                None
                if trials[0].observables[key] is None
                else np.concatenate([trial.observables[key] for trial in trials])
            )
            for key in keys
        }
        return InPlaneConstitutiveTrial(
            stress_in_plane_mpa=stress,
            tangent_in_plane_mpa=tangents,
            observables=observables,
        )

    def revert(self) -> None:
        for chunk in self._chunks:
            chunk.revert()

    def commit(self) -> None:
        for chunk in self._chunks:
            chunk.commit()


def _make_material(
    library: Path, point_count: int, yield_stress: np.ndarray, hardening: np.ndarray
) -> _ChunkedGenericMaterial:
    return _ChunkedGenericMaterial(
        library,
        yield_stress=np.repeat(yield_stress, 2),
        hardening=np.repeat(hardening, 2),
    )


def _make_native_material(
    library: Path,
    point_count: int,
    yield_stress: np.ndarray,
    hardening: np.ndarray,
) -> MFrontNativePlaneStressBatch:
    return MFrontNativePlaneStressBatch(
        library,
        np.repeat(yield_stress, 2),
        np.repeat(hardening, 2),
        np.full(point_count, 0.245),
        behaviour_name="PixelMicromorphicLudwikJ2Plasticity",
        micromorphic_coupling_modulus_mpa=2_000.0,
    )


def _solve_sequence(
    *,
    library: Path,
    history: np.ndarray,
    yield_stress: np.ndarray,
    hardening: np.ndarray,
    grid: StructuredGrid2D,
    kinematics: TwoSubcellDiagnostic2D,
    green: B0Green2D,
    mechanical_inverse: object,
    nonlocal_inverse: object,
    nonlocal_operator: object,
    length_scale: float,
    krylov_tolerance: float,
    absolute_tolerance: float,
    method: str,
    backend: str,
) -> dict[str, object]:
    material_factory = _make_material if backend == "generic" else _make_native_material
    material = material_factory(library, kinematics.material_point_count, yield_stress, hardening)
    mechanical = np.zeros(2 * grid.interior_shape[0] * grid.interior_shape[1])
    chi = np.zeros(grid.nx * grid.ny)
    total_newton = 0
    krylov_counts: list[int] = []
    outer_counts: list[int] = []
    mechanical_counts: list[int] = []
    material_tangent_seconds = 0.0
    material_residual_seconds = 0.0
    tangent_evaluations = 0
    residual_evaluations = 0
    final_mechanical_residual_norm = float("nan")
    final_nonlocal_residual_norm = float("nan")
    started = time.perf_counter()

    for increment in range(1, len(history)):
        boundary = history[increment]

        def strain_from_mechanical(
            value: np.ndarray, boundary_value: np.ndarray = boundary
        ) -> np.ndarray:
            full = boundary_value.copy()
            full[1:-1, 1:-1] += unpack_interior(value, grid)[1:-1, 1:-1]
            return kinematics.strain_samples(full).reshape(-1, 3)

        def evaluate_material(value: np.ndarray, local_chi: np.ndarray, tangent: bool):
            nonlocal material_tangent_seconds, material_residual_seconds
            nonlocal tangent_evaluations, residual_evaluations
            start = time.perf_counter()
            try:
                material.set_nonlocal_equivalent_plastic_strain(
                    np.repeat(local_chi, 2)
                )
                trial = material.evaluate_in_plane(
                    strain_from_mechanical(value),
                    time_increment=1.0,
                    consistent_tangent=tangent,
                )
            except (RuntimeError, ValueError) as error:
                raise RuntimeError("constitutive trial is inadmissible") from error
            elapsed = time.perf_counter() - start
            if tangent:
                material_tangent_seconds += elapsed
                tangent_evaluations += 1
            else:
                material_residual_seconds += elapsed
                residual_evaluations += 1
            return trial

        def residual_only(state: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            trial = evaluate_material(state[0], state[1], False)
            stress = np.asarray(trial.stress_in_plane_mpa).reshape(grid.nx, grid.ny, 2, 3)
            source = (
                np.asarray(trial.observables["equivalent_plastic_strain"])
                .reshape(grid.nx, grid.ny, 2)
                .mean(axis=2)
            )
            material.revert()
            return pack_interior(kinematics.divergence(stress)), nonlocal_operator(
                state[1]
            ) - source.reshape(-1)

        def linearise(state: tuple[np.ndarray, np.ndarray]) -> CoupledLinearisation:
            value, local_chi = state
            trial = evaluate_material(value, local_chi, True)
            stress = np.asarray(trial.stress_in_plane_mpa).reshape(grid.nx, grid.ny, 2, 3)
            source = (
                np.asarray(trial.observables["equivalent_plastic_strain"])
                .reshape(grid.nx, grid.ny, 2)
                .mean(axis=2)
            )
            tangent = np.asarray(trial.tangent_in_plane_mpa).reshape(grid.nx, grid.ny, 2, 3, 3)
            if backend == "generic":
                dsigma_dchi = np.asarray(trial.observables["generic_dsigma_dchi"]).reshape(
                    grid.nx, grid.ny, 2, 3
                )
                dp_depsilon = np.asarray(trial.observables["generic_dp_depsilon"]).reshape(
                    grid.nx, grid.ny, 2, 3
                )
                dp_dchi = np.asarray(trial.observables["generic_dp_dchi"]).reshape(
                    grid.nx, grid.ny, 2
                )
            else:
                point_chi = np.repeat(local_chi, 2)

                def constitutive_response(
                    strain_values: np.ndarray, chi_values: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
                    material.set_nonlocal_equivalent_plastic_strain(chi_values)
                    probe = material.evaluate_in_plane(
                        strain_values,
                        time_increment=1.0,
                        consistent_tangent=False,
                    )
                    stress_probe = np.asarray(probe.stress_in_plane_mpa)
                    source_probe = np.asarray(probe.observables["equivalent_plastic_strain"])
                    material.revert()
                    return stress_probe, source_probe

                sensitivity = finite_difference_sensitivities(
                    constitutive_response,
                    strain_from_mechanical(value),
                    point_chi,
                    base_stress=np.asarray(trial.stress_in_plane_mpa),
                    base_observable=np.asarray(trial.observables["equivalent_plastic_strain"]),
                    strain_step=1.0e-7 * max(grid.spacing_x, grid.spacing_y),
                    parameter_step=1.0e-7,
                    central_parameter=False,
                    forward_strain=False,
                )
                dsigma_dchi = sensitivity.stress_parameter.reshape(grid.nx, grid.ny, 2, 3)
                dp_depsilon = sensitivity.observable_strain.reshape(grid.nx, grid.ny, 2, 3)
                dp_dchi = sensitivity.observable_parameter.reshape(grid.nx, grid.ny, 2)
            ru = pack_interior(kinematics.divergence(stress))
            g = nonlocal_operator(local_chi) - source.reshape(-1)
            material.revert()

            def combined(du: np.ndarray, dchi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                displacement = unpack_interior(du, grid)
                de = kinematics.strain_samples(displacement).reshape(grid.nx, grid.ny, 2, 3)
                dc = dchi.reshape(grid.nx, grid.ny, 1, 1)
                ds = np.einsum("...ij,...j->...i", tangent, de) + dsigma_dchi * dc
                dp = (
                    np.einsum("...i,...i->...", dp_depsilon, de)
                    + dp_dchi * dchi.reshape(grid.nx, grid.ny, 1)
                ).mean(axis=2)
                return pack_interior(kinematics.divergence(ds)), nonlocal_operator(
                    dchi
                ) - dp.reshape(-1)

            def mechanical_action(du: np.ndarray) -> np.ndarray:
                return combined(du, np.zeros_like(local_chi))[0]

            def coupling_action(dchi: np.ndarray) -> np.ndarray:
                return combined(np.zeros_like(value), dchi)[0]

            def source_mechanical_action(du: np.ndarray) -> np.ndarray:
                return combined(du, np.zeros_like(local_chi))[1]

            def source_chi_action(dchi: np.ndarray) -> np.ndarray:
                return combined(np.zeros_like(value), dchi)[1]

            return CoupledLinearisation(
                mechanical_residual=ru,
                nonlocal_residual=g,
                actions=CoupledBlockActions(
                    mechanical_size=value.size,
                    nonlocal_size=local_chi.size,
                    ruu=mechanical_action,
                    ruchi=coupling_action,
                    g_u=source_mechanical_action,
                    g_chi=source_chi_action,
                    mechanical_inverse=mechanical_inverse,
                    nonlocal_inverse=nonlocal_inverse,
                    combined=combined,
                ),
            )

        if method == "monolithic":
            result = solve_coupled_newton(
                mechanical,
                chi,
                linearise,
                config=CoupledNewtonConfig(
                    maximum_iterations=50,
                    krylov_relative_tolerance=krylov_tolerance,
                    relative_tolerance=0.0,
                    absolute_tolerance=absolute_tolerance,
                    evaluate_initial_residual=False,
                    line_search=True,
                    line_search_minimum_step=1.0 / 1024.0,
                    enforce_nonnegative_nonlocal=True,
                ),
                evaluate_residual=residual_only,
            )
            if not result.converged:
                raise RuntimeError(f"monolithic increment {increment} did not converge")
            mechanical = result.mechanical
            chi = result.nonlocal_field
            final_mechanical_residual_norm = result.final_mechanical_residual_norm
            final_nonlocal_residual_norm = result.final_nonlocal_residual_norm
            total_newton += result.iterations
            krylov_counts.extend(result.krylov_iterations)
            outer_counts.append(1)
            mechanical_counts.append(0)
        else:
            outer = 0
            mechanical_solves = 0
            while outer < 30:
                outer += 1
                ru, source = residual_only((mechanical, chi))
                ru_norm = float(np.linalg.norm(ru))
                if ru_norm > absolute_tolerance:
                    lin = linearise((mechanical, chi))
                    operator = LinearOperator(
                        (mechanical.size, mechanical.size),
                        matvec=lambda v, actions=lin.actions: actions.ruu(v),
                        dtype=float,
                    )
                    correction, info, calls = solve_nonsymmetric_krylov(
                        operator,
                        -ru,
                        preconditioner=LinearOperator(
                            (mechanical.size, mechanical.size),
                            matvec=mechanical_inverse,
                            dtype=float,
                        ),
                        method="gmres",
                        rtol=krylov_tolerance,
                        maximum_iterations=400,
                        restart=100,
                    )
                    if info != 0:
                        raise RuntimeError(f"staggered increment {increment} GMRES failed: {info}")
                    mechanical += correction
                    mechanical_solves += 1
                    krylov_counts.append(calls)
                    continue
                updated_chi = np.maximum(nonlocal_inverse(source), 0.0)
                chi_residual = float(np.linalg.norm(updated_chi - chi))
                chi = updated_chi
                if chi_residual <= absolute_tolerance:
                    break
            if outer >= 30:
                raise RuntimeError(f"staggered increment {increment} did not converge")
            outer_counts.append(outer)
            mechanical_counts.append(mechanical_solves)
            final_mechanical_residual_norm = ru_norm
            final_nonlocal_residual_norm = chi_residual

        material.set_nonlocal_equivalent_plastic_strain(np.repeat(chi, 2))
        material.evaluate_in_plane(
            strain_from_mechanical(mechanical), time_increment=1.0, consistent_tangent=False
        )
        material.commit()

    elapsed = time.perf_counter() - started
    return {
        "method": method,
        "elapsed_seconds": elapsed,
        "newton_iterations": total_newton,
        "krylov_iterations": krylov_counts,
        "krylov_total": int(sum(krylov_counts)),
        "outer_iterations": outer_counts,
        "mechanical_iterations": mechanical_counts,
        "material_tangent_evaluations": tangent_evaluations,
        "material_residual_evaluations": residual_evaluations,
        "material_tangent_seconds": material_tangent_seconds,
        "material_residual_seconds": material_residual_seconds,
        "final_mechanical_residual_norm": final_mechanical_residual_norm,
        "final_nonlocal_residual_norm": final_nonlocal_residual_norm,
        "final_mechanical": mechanical,
        "final_chi": chi,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("native", "generic"), default="native")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--generic-library", type=Path)
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=DEFAULT_CROP)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--path-substeps", type=int, default=1)
    parser.add_argument("--length-scale", type=float, default=0.8)
    parser.add_argument("--krylov-relative-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    library = args.generic_library if args.backend == "generic" else args.library
    if library is None:
        parser.error("--library is required for native or --generic-library for generic")
    crop = tuple(args.crop_nodes)
    mesh = crop[1] - crop[0]
    if mesh != crop[3] - crop[2] or args.increments > 8:
        raise SystemExit("P43 crop must be square and increments must be <= 8")
    history, yield_stress, hardening, _ = _load_p43(crop)
    history = history[: args.increments + 1]
    if args.path_substeps > 1:
        refined = [history[0]]
        for start, stop in pairwise(history):
            for fraction in np.linspace(
                1.0 / args.path_substeps, 1.0, args.path_substeps
            ):
                refined.append((1.0 - fraction) * start + fraction * stop)
        history = np.asarray(refined)
    grid = StructuredGrid2D(mesh, mesh, mesh * PIXEL_SIZE_MM, mesh * PIXEL_SIZE_MM)
    kinematics = TwoSubcellDiagnostic2D(grid)
    point_count = kinematics.material_point_count
    material_factory = _make_material if args.backend == "generic" else _make_native_material
    virgin = material_factory(library, point_count, yield_stress, hardening)
    virgin_trial = virgin.evaluate_in_plane(
        np.zeros((point_count, 3)), time_increment=1.0, consistent_tangent=True
    )
    tangent = np.asarray(virgin_trial.tangent_in_plane_mpa).reshape(mesh, mesh, 2, 3, 3)
    virgin.revert()
    lambda_0, mu_0, projection_error = project_isotropic_plane_stress_tangent(
        tangent.mean(axis=(0, 1, 2))
    )
    plan = create_full_dirichlet_dsti_plan(grid, SpectralTransformConfig())
    green = B0Green2D(kinematics.reference_operator_symbols(plan), lambda_0=lambda_0, mu_0=mu_0)
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
    started = time.perf_counter()
    mono = _solve_sequence(
        library=library,
        history=history,
        yield_stress=yield_stress,
        hardening=hardening,
        grid=grid,
        kinematics=kinematics,
        green=green,
        mechanical_inverse=mechanical_inverse,
        nonlocal_inverse=nonlocal_inverse,
        nonlocal_operator=nonlocal_operator,
        length_scale=args.length_scale,
        krylov_tolerance=args.krylov_relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
        method="monolithic",
        backend=args.backend,
    )
    stag = _solve_sequence(
        library=library,
        history=history,
        yield_stress=yield_stress,
        hardening=hardening,
        grid=grid,
        kinematics=kinematics,
        green=green,
        mechanical_inverse=mechanical_inverse,
        nonlocal_inverse=nonlocal_inverse,
        nonlocal_operator=nonlocal_operator,
        length_scale=args.length_scale,
        krylov_tolerance=args.krylov_relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
        method="staggered",
        backend=args.backend,
    )
    total = time.perf_counter() - started
    report = {
        "status": f"completed_coupled_{args.backend}_j2_p43",
        "backend": args.backend,
        "crop_nodes": list(crop),
        "mesh": [mesh, mesh],
        "increments": args.increments,
        "path_substeps": args.path_substeps,
        "effective_increments": len(history) - 1,
        "pixel_size_mm": PIXEL_SIZE_MM,
        "length_scale": args.length_scale,
        "krylov_relative_tolerance": args.krylov_relative_tolerance,
        "absolute_tolerance": args.absolute_tolerance,
        "b0_lambda": lambda_0,
        "b0_mu": mu_0,
        "b0_projection_error": projection_error,
        "total_elapsed_seconds": total,
        "monolithic": {k: v for k, v in mono.items() if k not in {"final_mechanical", "final_chi"}},
        "staggered": {k: v for k, v in stag.items() if k not in {"final_mechanical", "final_chi"}},
        "comparison": {
            "time_ratio_staggered_over_monolithic": stag["elapsed_seconds"]
            / mono["elapsed_seconds"],
            "mechanical_linf": float(
                np.max(np.abs(mono["final_mechanical"] - stag["final_mechanical"]))
            ),
            "chi_linf": float(np.max(np.abs(mono["final_chi"] - stag["final_chi"]))),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
