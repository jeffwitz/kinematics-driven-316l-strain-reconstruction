#!/usr/bin/env python3
"""Compare monolithic and staggered generic micromorphic J2 on P43.

This driver uses the registered P43 DIC boundary history and the spatial J2
maps used by the TRI2 benchmark.  It deliberately keeps the generic MFront
adapter and the coupled matrix-free operators from the prototype separate
from the production crystal-plasticity drivers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import pairwise
from pathlib import Path

import h5py
import numpy as np
from benchmark_tri2_j2_krylov import DATA_ROOT, DEFAULT_CROP, PIXEL_SIZE_MM
from prototype_coupled_j2_nonlocal import GenericStructuralMicromorphicBatch
from scipy.sparse.linalg import LinearOperator

from fem_inhouse.core.constitutive_sensitivities import finite_difference_sensitivities
from fem_inhouse.core.crystal_orientation import PixelOrientationProvider
from fem_inhouse.core.crystal_parameter_pairs import (
    PAIRED_PARAMETER_SET,
    resolve_paired_crystal_parameters,
)
from fem_inhouse.core.mfront_native import MFrontNativePlaneStressBatch
from fem_inhouse.core.nonlocal_plasticity import evaluate_nonlocal_fixed_point
from fem_inhouse.core.plane_stress_material import (
    InPlaneConstitutiveTrial,
)
from fem_inhouse.core.srix_generic import (
    MericGeneric3DCondensedPlaneStressBatch,
    MericGeneric3DMaterialPointBatch,
    SrixGeneric3DCondensedPlaneStressBatch,
    SrixGeneric3DMaterialPointBatch,
)
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
from fem_inhouse.spectral2d.step_control import (
    AdaptiveLoadStepController,
    AdaptiveStepConfig,
    LoadStepObservation,
)
from fem_inhouse.spectral2d.transform_factory import create_full_dirichlet_dsti_plan
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig


class ProductionNestedFailureError(RuntimeError):
    """Failure carrying a compact nested-Newton diagnostic payload."""

    def __init__(self, message: str, diagnostics: dict[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


DEFAULT_EBSD_ORIENTATION_H5 = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")


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


def _refine_history(history: np.ndarray, subdivisions: int) -> np.ndarray:
    """Refine every prescribed DIC segment by a global cutback factor."""
    if subdivisions < 1:
        raise ValueError("subdivisions must be positive")
    if subdivisions == 1:
        return history
    refined = [history[0]]
    for start, stop in pairwise(history):
        for fraction in np.linspace(1.0 / subdivisions, 1.0, subdivisions):
            refined.append((1.0 - fraction) * start + fraction * stop)
    return np.asarray(refined)


def _history_at_time_fractions(
    history: np.ndarray, fractions: np.ndarray
) -> np.ndarray:
    """Interpolate the prescribed DIC history at explicit normalized times."""
    values = np.asarray(fractions, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("time fractions must be a one-dimensional sequence")
    if not np.isfinite(values).all() or values[0] != 0.0 or values[-1] != 1.0:
        raise ValueError("time fractions must be finite and span exactly [0, 1]")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("time fractions must be strictly increasing")
    original = np.linspace(0.0, 1.0, len(history))
    for knot in original:
        if not np.any(np.isclose(values, knot, rtol=0.0, atol=1.0e-14)):
            raise ValueError(f"time fractions must preserve DIC knot {knot:.16g}")
    positions = values * (len(history) - 1)
    left = np.minimum(np.floor(positions).astype(int), len(history) - 2)
    local = positions - left
    local[values == 1.0] = 1.0
    return (1.0 - local).reshape((-1,) + (1,) * (history.ndim - 1)) * history[
        left
    ] + local.reshape((-1,) + (1,) * (history.ndim - 1)) * history[left + 1]


def _time_increment(history: np.ndarray, total_duration: float = 1.0) -> float:
    """Return the uniform duration of one transition in ``history``."""
    if history.ndim < 1 or history.shape[0] < 2:
        raise ValueError("history must contain at least two states")
    if not np.isfinite(total_duration) or total_duration <= 0.0:
        raise ValueError("total_duration must be finite and positive")
    return total_duration / float(history.shape[0] - 1)


def _load_ebsd_rotations(
    path: Path, crop: tuple[int, int, int, int], *, states_per_pixel: int
) -> tuple[np.ndarray, dict[str, object]]:
    """Load the co-registered P43 Bunge map and replicate it over cell states."""
    x0, x1, y0, y1 = crop
    with h5py.File(path, "r") as handle:
        angles = np.stack(
            [
                np.asarray(handle[f"orientation/{name}"][x0:x1, y0:y1], dtype=float)
                for name in ("phi1", "Phi", "phi2")
            ],
            axis=-1,
        )
    expected_shape = (x1 - x0, y1 - y0, 3)
    if angles.shape != expected_shape or not np.isfinite(angles).all():
        raise ValueError(f"invalid EBSD angle crop: expected {expected_shape}, got {angles.shape}")
    provider = PixelOrientationProvider.from_euler_bunge_deg(angles)
    point_count = (x1 - x0) * (y1 - y0) * states_per_pixel
    rotations = provider.rotations_global_to_material(point_count)
    return rotations, {
        "mode": "ebsd",
        "source_file": str(path.resolve()),
        "angles_sha256": hashlib.sha256(np.ascontiguousarray(angles).tobytes()).hexdigest(),
        "unique_orientations": int(np.unique(angles.reshape(-1, 3), axis=0).shape[0]),
        "states_per_pixel": states_per_pixel,
    }


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


def _make_srix_material(
    library: Path,
    point_count: int,
    coupling_modulus_mpa: float,
    rotations_global_to_material: np.ndarray,
):
    return _make_crystal_material(
        "srix", library, point_count, coupling_modulus_mpa, rotations_global_to_material
    )


def _make_meric_material(
    library: Path,
    point_count: int,
    coupling_modulus_mpa: float,
    rotations_global_to_material: np.ndarray,
):
    return _make_crystal_material(
        "meric", library, point_count, coupling_modulus_mpa, rotations_global_to_material
    )


def _make_crystal_material(
    model: str,
    library: Path,
    point_count: int,
    coupling_modulus_mpa: float,
    rotations_global_to_material: np.ndarray,
):
    law = "forest_rubin_srix" if model == "srix" else "meric_cailletaud"
    parameters, _ = resolve_paired_crystal_parameters(
        paired_parameter_set=PAIRED_PARAMETER_SET,
        law=law,
    )
    bridge_type = (
        SrixGeneric3DMaterialPointBatch if model == "srix" else MericGeneric3DMaterialPointBatch
    )
    condensed_type = (
        SrixGeneric3DCondensedPlaneStressBatch
        if model == "srix"
        else MericGeneric3DCondensedPlaneStressBatch
    )
    bridge = bridge_type(
        library,
        point_count=point_count,
        micromorphic_coupling_modulus_mpa=coupling_modulus_mpa,
        rotation_global_to_material=rotations_global_to_material,
        behaviour_parameters=parameters,
        thread_count=4,
    )
    return condensed_type(bridge)


def _make_native_material(
    library: Path,
    point_count: int,
    yield_stress: np.ndarray,
    hardening: np.ndarray,
    coupling_modulus_mpa: float,
) -> MFrontNativePlaneStressBatch:
    return MFrontNativePlaneStressBatch(
        library,
        np.repeat(yield_stress, 2),
        np.repeat(hardening, 2),
        np.full(point_count, 0.245),
        behaviour_name="PixelMicromorphicLudwikJ2Plasticity",
        micromorphic_coupling_modulus_mpa=coupling_modulus_mpa,
    )


def _make_coupled_material(
    *,
    library: Path,
    backend: str,
    material_model: str,
    point_count: int,
    yield_stress: np.ndarray,
    hardening: np.ndarray,
    coupling_modulus_mpa: float,
    crystal_rotations: np.ndarray | None,
):
    """Build one transactional material shared by all accepted subincrements."""
    if material_model == "meric":
        if crystal_rotations is None:
            raise RuntimeError("Méric requires an explicit crystal orientation field")
        return _make_meric_material(
            library,
            point_count,
            coupling_modulus_mpa,
            crystal_rotations,
        )
    if material_model == "srix":
        if crystal_rotations is None:
            raise RuntimeError("SRIX requires an explicit crystal orientation field")
        return _make_srix_material(
            library,
            point_count,
            coupling_modulus_mpa,
            crystal_rotations,
        )
    if backend == "generic":
        return _make_material(library, point_count, yield_stress, hardening)
    return _make_native_material(
        library,
        point_count,
        yield_stress,
        hardening,
        coupling_modulus_mpa,
    )


def _solve_production_nested_sequence(
    *,
    library: Path,
    history: np.ndarray,
    yield_stress: np.ndarray,
    hardening: np.ndarray,
    grid: StructuredGrid2D,
    kinematics: TwoSubcellDiagnostic2D,
    mechanical_inverse: object,
    nonlocal_inverse: object,
    length_scale: float,
    coupling_modulus_mpa: float,
    absolute_tolerance: float,
    total_duration: float = 1.0,
    time_increments: np.ndarray | None = None,
) -> dict[str, object]:
    """Run production-style coupling with local cutback and regrowth."""
    material = _make_native_material(
        library,
        kinematics.material_point_count,
        yield_stress,
        hardening,
        coupling_modulus_mpa,
    )
    mechanical = np.zeros(2 * grid.interior_shape[0] * grid.interior_shape[1])
    chi = np.zeros(grid.nx * grid.ny)
    newton_counts: list[int] = []
    fixed_point_counts: list[list[int]] = []
    krylov_counts: list[int] = []
    tangent_evaluations = 0
    residual_evaluations = 0
    local_cutbacks = 0
    accepted_subincrements = 0
    started = time.perf_counter()

    def solve_attempt(
        boundary: np.ndarray,
        time_increment: float,
        initial_mechanical: np.ndarray,
        initial_chi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int, list[int], list[int]]:
        local_mechanical = initial_mechanical.copy()
        local_chi = initial_chi.copy()
        attempt_krylov: list[int] = []
        last_line_search_failure: str | None = None
        last_line_search_diagnostic: dict[str, object] | None = None

        def strain_from_mechanical(value: np.ndarray) -> np.ndarray:
            full = boundary.copy()
            full[1:-1, 1:-1] += unpack_interior(value, grid)[1:-1, 1:-1]
            return kinematics.strain_samples(full).reshape(-1, 3)

        def nested_trial(value: np.ndarray, initial_chi: np.ndarray):
            nonlocal tangent_evaluations, residual_evaluations
            material.revert()
            evaluation = evaluate_nonlocal_fixed_point(
                material,
                strain_from_mechanical(value),
                time_increment=time_increment,
                element_shape=grid.pixel_shape,
                gauss_points_per_element=2,
                initial_nonlocal_peeq=initial_chi.reshape(grid.pixel_shape),
                length_scale_mm=length_scale,
                spacing_x_mm=grid.spacing_x,
                spacing_y_mm=grid.spacing_y,
                coupling_modulus_mpa=coupling_modulus_mpa,
                relaxation=0.5,
                relaxation_strategy="fixed",
                relative_tolerance=1.0e-6,
                maximum_iterations=15,
                maximum_helmholtz_residual=1.0e-10,
                element_order="C",
            )
            tangent_evaluations += 1
            residual_evaluations += 1
            return evaluation

        def residual_vector(trial: InPlaneConstitutiveTrial) -> np.ndarray:
            stress = np.asarray(trial.stress_in_plane_mpa).reshape(grid.nx, grid.ny, 2, 3)
            return pack_interior(kinematics.divergence(stress))

        def fixed_chi_residual(value: np.ndarray, fixed_chi: np.ndarray) -> np.ndarray:
            material.revert()
            material.set_nonlocal_equivalent_plastic_strain(np.repeat(fixed_chi, 2))
            trial = material.evaluate_in_plane(
                strain_from_mechanical(value),
                time_increment=time_increment,
                consistent_tangent=False,
            )
            residual = residual_vector(trial)
            material.revert()
            return residual

        increment_fixed_point_counts: list[int] = []
        for iteration in range(1, 51):
            evaluation = nested_trial(local_mechanical, local_chi)
            trial = evaluation.constitutive_trial
            stress = np.asarray(trial.stress_in_plane_mpa).reshape(grid.nx, grid.ny, 2, 3)
            ru = pack_interior(kinematics.divergence(stress))
            increment_fixed_point_counts.append(evaluation.iterations)
            if np.linalg.norm(ru) <= absolute_tolerance:
                local_chi = evaluation.nonlocal_peeq.reshape(-1).copy()
                material.commit()
                return (
                    local_mechanical,
                    local_chi,
                    iteration,
                    increment_fixed_point_counts,
                    attempt_krylov,
                )
            tangent = np.asarray(trial.tangent_in_plane_mpa).reshape(grid.nx, grid.ny, 2, 3, 3)
            base_mechanical = local_mechanical.copy()
            base_chi = evaluation.nonlocal_peeq.reshape(-1).copy()

            def mechanical_action(
                du: np.ndarray, tangent_value: np.ndarray = tangent
            ) -> np.ndarray:
                de = kinematics.strain_samples(unpack_interior(du, grid)).reshape(
                    grid.nx, grid.ny, 2, 3
                )
                ds = np.einsum("...ij,...j->...i", tangent_value, de)
                return pack_interior(kinematics.divergence(ds))

            operator = LinearOperator(
                (mechanical.size, mechanical.size), matvec=mechanical_action, dtype=float
            )
            correction, info, calls = solve_nonsymmetric_krylov(
                operator,
                -ru,
                preconditioner=LinearOperator(
                    (mechanical.size, mechanical.size), matvec=mechanical_inverse, dtype=float
                ),
                method="gmres",
                rtol=1.0e-4,
                maximum_iterations=400,
                restart=100,
            )
            if info != 0:
                raise RuntimeError(f"production nested GMRES failed: {info}")
            attempt_krylov.append(calls)
            current_norm = float(np.linalg.norm(ru))
            step = 1.0
            line_search_curve: list[dict[str, float]] = []
            while step >= 1.0 / 1024.0:
                candidate = base_mechanical + step * correction
                try:
                    candidate_eval = nested_trial(candidate, base_chi)
                except (RuntimeError, ValueError) as error:
                    last_line_search_failure = str(error)
                    step *= 0.5
                    continue
                candidate_stress = np.asarray(
                    candidate_eval.constitutive_trial.stress_in_plane_mpa
                ).reshape(grid.nx, grid.ny, 2, 3)
                candidate_norm = float(
                    np.linalg.norm(pack_interior(kinematics.divergence(candidate_stress)))
                )
                fixed_norm = float(np.linalg.norm(fixed_chi_residual(candidate, base_chi)))
                line_search_curve.append(
                    {
                        "alpha": float(step),
                        "nested_residual_norm": candidate_norm,
                        "fixed_chi_residual_norm": fixed_norm,
                    }
                )
                if candidate_norm < current_norm:
                    local_mechanical = candidate
                    local_chi = candidate_eval.nonlocal_peeq.reshape(-1).copy()
                    break
                step *= 0.5
            else:
                a_direction = mechanical_action(correction)
                base_trial_vectors: list[np.ndarray] = []
                for _ in range(5):
                    repeated_eval = nested_trial(base_mechanical, base_chi)
                    base_trial_vectors.append(residual_vector(repeated_eval.constitutive_trial))
                repeatability = {
                    "samples": len(base_trial_vectors),
                    "max_difference_vs_first": float(
                        max(
                            np.linalg.norm(vector - base_trial_vectors[0])
                            for vector in base_trial_vectors[1:]
                        )
                    ),
                    "rms_difference_vs_first": float(
                        np.sqrt(
                            np.mean(
                                [
                                    np.linalg.norm(vector - base_trial_vectors[0]) ** 2
                                    for vector in base_trial_vectors[1:]
                                ]
                            )
                        )
                    ),
                }
                strain_direction = kinematics.strain_samples(
                    unpack_interior(correction, grid)
                ).reshape(grid.nx, grid.ny, 2, 3)
                strain_max = float(np.max(np.abs(strain_direction)))
                controlled_amplitude_diagnostics: list[dict[str, object]] = []
                if strain_max > 0.0:
                    for target_strain in (1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9):
                        scaled_correction = correction * (target_strain / strain_max)
                        try:
                            plus_eval = nested_trial(base_mechanical + scaled_correction, base_chi)
                            minus_eval = nested_trial(base_mechanical - scaled_correction, base_chi)
                            nested_central = (
                                residual_vector(plus_eval.constitutive_trial)
                                - residual_vector(minus_eval.constitutive_trial)
                            ) / 2.0
                            plus_fixed = fixed_chi_residual(
                                base_mechanical + scaled_correction, base_chi
                            )
                            minus_fixed = fixed_chi_residual(
                                base_mechanical - scaled_correction, base_chi
                            )
                            fixed_central = (plus_fixed - minus_fixed) / 2.0
                            tangent_action = mechanical_action(scaled_correction)
                            tangent_norm = float(np.linalg.norm(tangent_action))
                            nested_norm = float(np.linalg.norm(nested_central))
                            fixed_norm = float(np.linalg.norm(fixed_central))
                            controlled_amplitude_diagnostics.append(
                                {
                                    "max_strain_amplitude": target_strain,
                                    "nested_relative_error_vs_A": float(
                                        np.linalg.norm(nested_central - tangent_action)
                                        / max(nested_norm, np.finfo(float).tiny)
                                    ),
                                    "fixed_chi_relative_error_vs_A": float(
                                        np.linalg.norm(fixed_central - tangent_action)
                                        / max(fixed_norm, np.finfo(float).tiny)
                                    ),
                                    "nested_cosine_vs_A": float(
                                        np.dot(nested_central, tangent_action)
                                        / max(nested_norm * tangent_norm, np.finfo(float).tiny)
                                    ),
                                    "fixed_chi_cosine_vs_A": float(
                                        np.dot(fixed_central, tangent_action)
                                        / max(fixed_norm * tangent_norm, np.finfo(float).tiny)
                                    ),
                                }
                            )
                        except (RuntimeError, ValueError) as error:
                            controlled_amplitude_diagnostics.append(
                                {
                                    "max_strain_amplitude": target_strain,
                                    "failure": str(error),
                                }
                            )
                alpha_diagnostics: list[dict[str, object]] = []
                for diagnostic_alpha in (1.0 / 512.0, 1.0 / 1024.0, 1.0 / 2048.0, 1.0 / 4096.0):
                    diagnostic_candidate = base_mechanical + diagnostic_alpha * correction
                    try:
                        diagnostic_eval = nested_trial(diagnostic_candidate, base_chi)
                        nested_vector = residual_vector(diagnostic_eval.constitutive_trial)
                        fixed_vector = fixed_chi_residual(diagnostic_candidate, base_chi)
                        nested_action = (nested_vector - ru) / diagnostic_alpha
                        fixed_action = (fixed_vector - ru) / diagnostic_alpha
                        a_norm = float(np.linalg.norm(a_direction))
                        nested_norm = float(np.linalg.norm(nested_action))
                        fixed_norm = float(np.linalg.norm(fixed_action))
                        alpha_diagnostics.append(
                            {
                                "alpha": diagnostic_alpha,
                                "nested_relative_error_vs_A": float(
                                    np.linalg.norm(nested_action - a_direction)
                                    / max(nested_norm, np.finfo(float).tiny)
                                ),
                                "fixed_chi_relative_error_vs_A": float(
                                    np.linalg.norm(fixed_action - a_direction)
                                    / max(fixed_norm, np.finfo(float).tiny)
                                ),
                                "nested_cosine_vs_A": float(
                                    np.dot(nested_action, a_direction)
                                    / max(nested_norm * a_norm, np.finfo(float).tiny)
                                ),
                                "fixed_chi_cosine_vs_A": float(
                                    np.dot(fixed_action, a_direction)
                                    / max(fixed_norm * a_norm, np.finfo(float).tiny)
                                ),
                                "nested_R_dot_Jdu": float(np.dot(ru, nested_action)),
                                "fixed_chi_R_dot_Jdu": float(np.dot(ru, fixed_action)),
                                "nested_normalized_R_dot_Jdu": float(
                                    np.dot(ru, nested_action)
                                    / max(np.linalg.norm(ru) * nested_norm, np.finfo(float).tiny)
                                ),
                                "fixed_chi_normalized_R_dot_Jdu": float(
                                    np.dot(ru, fixed_action)
                                    / max(np.linalg.norm(ru) * fixed_norm, np.finfo(float).tiny)
                                ),
                                "chi_response_per_alpha": float(
                                    np.linalg.norm(
                                        diagnostic_eval.nonlocal_peeq.reshape(-1) - base_chi
                                    )
                                    / diagnostic_alpha
                                ),
                            }
                        )
                    except (RuntimeError, ValueError) as error:
                        alpha_diagnostics.append({"alpha": diagnostic_alpha, "failure": str(error)})
                last_line_search_diagnostic = {
                    "current_residual_norm": current_norm,
                    "line_search_curve": line_search_curve,
                    "A_correction_norm": float(np.linalg.norm(a_direction)),
                    "repeatability": repeatability,
                    "controlled_amplitude_diagnostics": controlled_amplitude_diagnostics,
                    "secant_diagnostics": alpha_diagnostics,
                }
                detail = (
                    f"; last candidate failure: {last_line_search_failure}"
                    if last_line_search_failure is not None
                    else ""
                )
                raise ProductionNestedFailureError(
                    f"production nested line search failed{detail}",
                    last_line_search_diagnostic,
                )
        raise RuntimeError("production nested mechanical Newton did not converge")

    if time_increments is None:
        segment_durations = np.full(
            len(history) - 1, _time_increment(history, total_duration)
        )
    else:
        segment_durations = np.asarray(time_increments, dtype=float)
        if segment_durations.shape != (len(history) - 1,):
            raise ValueError("time_increments must match the history transitions")
    for increment, ((start_boundary, target_boundary), segment_duration) in enumerate(
        zip(pairwise(history), segment_durations, strict=True), start=1
    ):
        fraction = 0.0
        step_fraction = 1.0
        segment_cutbacks = 0
        last_failure: str | None = None
        last_failure_diagnostics: dict[str, object] = {}
        while fraction < 1.0 - 1.0e-14:
            next_fraction = min(1.0, fraction + step_fraction)
            boundary = (1.0 - next_fraction) * start_boundary + next_fraction * target_boundary
            saved_mechanical = mechanical.copy()
            saved_chi = chi.copy()
            try:
                (
                    mechanical,
                    chi,
                    iteration_count,
                    fp_counts,
                    attempt_krylov,
                ) = solve_attempt(
                    boundary,
                    segment_duration * (next_fraction - fraction),
                    saved_mechanical,
                    saved_chi,
                )
            except (RuntimeError, ValueError) as error:
                material.revert()
                mechanical = saved_mechanical
                chi = saved_chi
                local_cutbacks += 1
                segment_cutbacks += 1
                last_failure = str(error)
                if isinstance(error, ProductionNestedFailureError):
                    last_failure_diagnostics = error.diagnostics
                step_fraction *= 0.5
                if step_fraction < 1.0 / 1024.0:
                    diagnostics = {
                        **last_failure_diagnostics,
                        "increment": increment,
                        "fraction": fraction,
                        "segment_cutbacks": segment_cutbacks,
                        "time_increment": segment_duration * step_fraction,
                    }
                    raise ProductionNestedFailureError(
                        "production nested local cutback failed "
                        f"at increment {increment}, fraction={fraction:.16g}, "
                        f"segment_cutbacks={segment_cutbacks}, "
                        f"last_failure={last_failure}",
                        diagnostics,
                    ) from None
                continue
            fraction = next_fraction
            accepted_subincrements += 1
            newton_counts.append(iteration_count)
            fixed_point_counts.append(fp_counts)
            krylov_counts.extend(attempt_krylov)
            step_fraction = min(1.0 - fraction, 1.5 * step_fraction)

    material.revert()
    material.set_nonlocal_equivalent_plastic_strain(np.repeat(chi, 2))
    final_full = history[-1].copy()
    final_full[1:-1, 1:-1] += unpack_interior(mechanical, grid)[1:-1, 1:-1]
    final_trial = material.evaluate_in_plane(
        kinematics.strain_samples(final_full).reshape(-1, 3),
        time_increment=float(segment_durations[-1]),
        consistent_tangent=False,
    )
    final_stress = np.asarray(final_trial.stress_in_plane_mpa).reshape(grid.nx, grid.ny, 2, 3)
    final_peeq = (
        np.asarray(final_trial.observables["equivalent_plastic_strain"])
        .reshape(grid.nx, grid.ny, 2)
        .mean(axis=2)
    )
    material.revert()

    return {
        "method": "production-nested",
        "elapsed_seconds": time.perf_counter() - started,
        "newton_iterations": int(sum(newton_counts)),
        "newton_iterations_per_increment": newton_counts,
        "fixed_point_iterations_per_newton": fixed_point_counts,
        "krylov_iterations": krylov_counts,
        "krylov_total": int(sum(krylov_counts)),
        "material_tangent_evaluations": tangent_evaluations,
        "material_residual_evaluations": residual_evaluations,
        "local_cutbacks": local_cutbacks,
        "accepted_subincrements": accepted_subincrements,
        "final_mechanical": mechanical,
        "final_chi": chi,
        "final_stress": final_stress,
        "final_peeq": final_peeq,
        "final_source": final_peeq,
    }


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
    coupling_modulus_mpa: float,
    krylov_tolerance: float,
    absolute_tolerance: float,
    method: str,
    backend: str,
    staggered_relaxation: float,
    fd_strain_step: float,
    fd_chi_step: float,
    material_model: str = "j2",
    crystal_rotations: np.ndarray | None = None,
    total_duration: float = 1.0,
    time_increments: np.ndarray | None = None,
    progress_path: Path | None = None,
    material_instance: object | None = None,
    initial_mechanical: np.ndarray | None = None,
    initial_chi: np.ndarray | None = None,
) -> dict[str, object]:
    requested_material_model = material_model
    material_model = "srix" if material_model == "meric" else material_model
    source_key = "nonlocal_source" if material_model == "srix" else "equivalent_plastic_strain"
    if method == "production-nested":
        if backend != "native":
            raise RuntimeError("production-nested currently requires the native backend")
        return _solve_production_nested_sequence(
            library=library,
            history=history,
            yield_stress=yield_stress,
            hardening=hardening,
            grid=grid,
            kinematics=kinematics,
            mechanical_inverse=mechanical_inverse,
            nonlocal_inverse=nonlocal_inverse,
            length_scale=length_scale,
            coupling_modulus_mpa=coupling_modulus_mpa,
            absolute_tolerance=absolute_tolerance,
            total_duration=total_duration,
            time_increments=time_increments,
        )
    material = (
        _make_coupled_material(
            library=library,
            backend=backend,
            material_model=requested_material_model,
            point_count=kinematics.material_point_count,
            yield_stress=yield_stress,
            hardening=hardening,
            coupling_modulus_mpa=coupling_modulus_mpa,
            crystal_rotations=crystal_rotations,
        )
        if material_instance is None
        else material_instance
    )
    mechanical_size = 2 * grid.interior_shape[0] * grid.interior_shape[1]
    mechanical = (
        np.zeros(mechanical_size)
        if initial_mechanical is None
        else np.asarray(initial_mechanical, dtype=float).reshape(mechanical_size).copy()
    )
    chi = (
        np.zeros(grid.nx * grid.ny)
        if initial_chi is None
        else np.asarray(initial_chi, dtype=float).reshape(grid.nx * grid.ny).copy()
    )
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
    if time_increments is None:
        increment_durations = np.full(
            len(history) - 1, _time_increment(history, total_duration)
        )
    else:
        increment_durations = np.asarray(time_increments, dtype=float)
        if increment_durations.shape != (len(history) - 1,):
            raise ValueError("time_increments must match the history transitions")
        if not np.isfinite(increment_durations).all() or np.any(
            increment_durations <= 0.0
        ):
            raise ValueError("time_increments must be finite and positive")
        if not np.isclose(increment_durations.sum(), total_duration):
            raise ValueError("time_increments must sum to total_duration")
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("", encoding="utf-8")

    for increment in range(1, len(history)):
        boundary = history[increment]
        time_increment = float(increment_durations[increment - 1])

        def strain_from_mechanical(
            value: np.ndarray, boundary_value: np.ndarray = boundary
        ) -> np.ndarray:
            full = boundary_value.copy()
            full[1:-1, 1:-1] += unpack_interior(value, grid)[1:-1, 1:-1]
            return kinematics.strain_samples(full).reshape(-1, 3)

        def evaluate_material(
            value: np.ndarray,
            local_chi: np.ndarray,
            tangent: bool,
            _time_increment: float = time_increment,
        ):
            nonlocal material_tangent_seconds, material_residual_seconds
            nonlocal tangent_evaluations, residual_evaluations
            start = time.perf_counter()
            try:
                material.set_nonlocal_equivalent_plastic_strain(np.repeat(local_chi, 2))
                trial = material.evaluate_in_plane(
                    strain_from_mechanical(value),
                    time_increment=_time_increment,
                    consistent_tangent=tangent,
                )
            except (RuntimeError, ValueError) as error:
                raise RuntimeError(
                    f"constitutive trial is inadmissible: {error}"
                ) from error
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
                np.asarray(trial.observables[source_key])
                .reshape(grid.nx, grid.ny, 2)
                .mean(axis=2)
                .copy()
            )
            material.revert()
            return pack_interior(kinematics.divergence(stress)), nonlocal_operator(
                state[1]
            ) - source.reshape(-1)

        def source_only(state: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
            trial = evaluate_material(state[0], state[1], False)
            source = (
                np.asarray(trial.observables[source_key])
                .reshape(grid.nx, grid.ny, 2)
                .mean(axis=2)
                .copy()
            )
            material.revert()
            return source.reshape(-1)

        def linearise(
            state: tuple[np.ndarray, np.ndarray],
            *,
            include_coupling: bool = True,
            _time_increment: float = time_increment,
        ) -> CoupledLinearisation:
            value, local_chi = state
            trial = evaluate_material(value, local_chi, True)
            stress = np.asarray(trial.stress_in_plane_mpa).reshape(grid.nx, grid.ny, 2, 3)
            source = (
                np.asarray(trial.observables[source_key]).reshape(grid.nx, grid.ny, 2).mean(axis=2)
            )
            tangent = np.asarray(trial.tangent_in_plane_mpa).reshape(grid.nx, grid.ny, 2, 3, 3)
            if not include_coupling:
                # The staggered mechanical solve only consumes C_ee.  Do not
                # pay for the cross sensitivities that the monolithic method
                # needs but the staggered method never applies.
                dsigma_dchi = np.zeros((*tangent.shape[:-2], 3))
                dp_depsilon = np.zeros((*tangent.shape[:-2], 3))
                dp_dchi = np.zeros(tangent.shape[:-2])
            elif material_model == "srix":
                dsigma_dchi = np.asarray(trial.observables["generic_dsigma_dchi"]).reshape(
                    grid.nx, grid.ny, 2, 3
                )
                dp_depsilon = np.asarray(trial.observables["generic_dq_depsilon"]).reshape(
                    grid.nx, grid.ny, 2, 3
                )
                dp_dchi = np.asarray(trial.observables["generic_dq_dchi"]).reshape(
                    grid.nx, grid.ny, 2
                )
            elif backend == "generic":
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
                        time_increment=_time_increment,
                        consistent_tangent=False,
                    )
                    stress_probe = np.asarray(probe.stress_in_plane_mpa)
                    source_probe = np.asarray(probe.observables[source_key])
                    material.revert()
                    return stress_probe, source_probe

                sensitivity = finite_difference_sensitivities(
                    constitutive_response,
                    strain_from_mechanical(value),
                    point_chi,
                    base_stress=np.asarray(trial.stress_in_plane_mpa),
                    base_observable=np.asarray(trial.observables[source_key]),
                    strain_step=fd_strain_step,
                    parameter_step=fd_chi_step,
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
            chi_residual = float("inf")
            mechanical_converged = False
            while outer < 30:
                outer += 1
                mechanical_converged = False
                for _ in range(50):
                    ru, _ = residual_only((mechanical, chi))
                    ru_norm = float(np.linalg.norm(ru))
                    if ru_norm <= absolute_tolerance:
                        mechanical_converged = True
                        break
                    lin = linearise((mechanical, chi), include_coupling=False)
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
                    base_mechanical = mechanical.copy()
                    step = 1.0
                    accepted = False
                    while step >= 1.0 / 1024.0:
                        candidate = base_mechanical + step * correction
                        try:
                            candidate_ru, _ = residual_only((candidate, chi))
                        except (RuntimeError, ValueError):
                            step *= 0.5
                            continue
                        candidate_norm = float(np.linalg.norm(candidate_ru))
                        if candidate_norm < ru_norm:
                            mechanical = candidate
                            accepted = True
                            break
                        step *= 0.5
                    if not accepted:
                        raise RuntimeError(
                            f"staggered increment {increment} mechanical line search failed"
                        )
                    mechanical_solves += 1
                    total_newton += 1
                    krylov_counts.append(calls)
                if not mechanical_converged:
                    raise RuntimeError(
                        f"staggered increment {increment} mechanical Newton did not converge"
                    )
                source = source_only((mechanical, chi))
                filtered_chi = np.maximum(nonlocal_inverse(source), 0.0)
                updated_chi = (
                    1.0 - staggered_relaxation
                ) * chi + staggered_relaxation * filtered_chi
                chi_residual = float(
                    np.linalg.norm(updated_chi - chi) / max(np.linalg.norm(updated_chi), 1.0e-30)
                )
                chi = updated_chi
                if chi_residual <= 1.0e-6:
                    break
            if not mechanical_converged or chi_residual > 1.0e-6:
                raise RuntimeError(f"staggered increment {increment} did not converge")
            outer_counts.append(outer)
            mechanical_counts.append(mechanical_solves)
            final_mechanical_residual_norm = ru_norm
            final_nonlocal_residual_norm = chi_residual

        material.set_nonlocal_equivalent_plastic_strain(np.repeat(chi, 2))
        final_trial = material.evaluate_in_plane(
            strain_from_mechanical(mechanical),
            time_increment=time_increment,
            consistent_tangent=False,
        )
        material.commit()
        if progress_path is not None:
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "increment": increment,
                            "total_increments": len(history) - 1,
                            "elapsed_seconds": time.perf_counter() - started,
                            "newton_total": total_newton,
                            "krylov_total": int(sum(krylov_counts)),
                            "mechanical_residual_norm": final_mechanical_residual_norm,
                            "nonlocal_residual_norm": final_nonlocal_residual_norm,
                        }
                    )
                    + "\n"
                )

    elapsed = time.perf_counter() - started
    final_stress = np.asarray(final_trial.stress_in_plane_mpa).reshape(grid.nx, grid.ny, 2, 3)
    final_peeq = (
        np.asarray(
            final_trial.observables[
                "nonlocal_source" if material_model == "srix" else "equivalent_plastic_strain"
            ]
        )
        .reshape(grid.nx, grid.ny, 2)
        .mean(axis=2)
    )
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
        "final_stress": final_stress,
        "final_peeq": final_peeq,
        "final_source": final_peeq,
    }


def _solve_sequence_with_local_cutback(
    *,
    history: np.ndarray,
    total_duration: float = 1.0,
    progress_path: Path | None = None,
    **solver_kwargs: object,
) -> dict[str, object]:
    """Advance one shared material transaction with selective path cutbacks."""
    grid = solver_kwargs["grid"]
    kinematics = solver_kwargs["kinematics"]
    assert isinstance(grid, StructuredGrid2D)
    assert isinstance(kinematics, TwoSubcellDiagnostic2D)
    segment_count = len(history) - 1
    segment_duration = total_duration / float(segment_count)
    step_config = AdaptiveStepConfig(
        initial_increment_fraction=1.0,
        minimum_increment_fraction=1.0 / 1024.0,
        maximum_increment_fraction=1.0,
        increment_growth_factor=1.5,
        increment_cutback_factor=0.5,
        target_newton_iterations_min=50,
        target_newton_iterations_max=50,
        maximum_cutbacks_per_step=20,
    )
    material = _make_coupled_material(
        library=solver_kwargs["library"],
        backend=solver_kwargs["backend"],
        material_model=solver_kwargs["material_model"],
        point_count=kinematics.material_point_count,
        yield_stress=solver_kwargs["yield_stress"],
        hardening=solver_kwargs["hardening"],
        coupling_modulus_mpa=solver_kwargs["coupling_modulus_mpa"],
        crystal_rotations=solver_kwargs.get("crystal_rotations"),
    )
    mechanical = np.zeros(2 * grid.interior_shape[0] * grid.interior_shape[1])
    chi = np.zeros(grid.nx * grid.ny)
    accepted_fractions = [0.0]
    accepted_time_increments: list[float] = []
    rejected_attempts: list[dict[str, object]] = []
    accumulated_newton = 0
    accumulated_krylov: list[int] = []
    accumulated_outer: list[int] = []
    accumulated_mechanical: list[int] = []
    accumulated_tangent_evaluations = 0
    accumulated_residual_evaluations = 0
    accumulated_tangent_seconds = 0.0
    accumulated_residual_seconds = 0.0
    last_result: dict[str, object] | None = None
    attempt_index = 0
    started = time.perf_counter()
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("", encoding="utf-8")

    def record(payload: dict[str, object]) -> None:
        if progress_path is None:
            return
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload) + "\n")

    for segment_index, (segment_start, segment_stop) in enumerate(pairwise(history), start=1):
        controller = AdaptiveLoadStepController(step_config)
        local_fraction = 0.0
        accepted_boundary = np.asarray(segment_start).copy()
        while local_fraction < 1.0 - 1.0e-14:
            attempt_index += 1
            next_fraction = controller.propose(local_fraction)
            attempt_boundary = (1.0 - next_fraction) * segment_start + next_fraction * segment_stop
            attempt_duration = segment_duration * (next_fraction - local_fraction)
            global_start = (segment_index - 1 + local_fraction) / segment_count
            global_end = (segment_index - 1 + next_fraction) / segment_count
            attempt_history = np.stack((accepted_boundary, attempt_boundary))
            record(
                {
                    "event": "attempt_started",
                    "attempt": attempt_index,
                    "dic_segment": segment_index,
                    "start_fraction": global_start,
                    "end_fraction": global_end,
                    "time_increment": attempt_duration,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            try:
                result = _solve_sequence(
                    **solver_kwargs,
                    history=attempt_history,
                    total_duration=attempt_duration,
                    progress_path=None,
                    material_instance=material,
                    initial_mechanical=mechanical,
                    initial_chi=chi,
                )
            except (RuntimeError, ValueError) as error:
                material.revert()
                try:
                    decision = controller.reject(type(error).__name__)
                except RuntimeError as controller_error:
                    raise RuntimeError(
                        "local coupled cutback exhausted at "
                        f"DIC segment {segment_index}, fraction={local_fraction:.16g}; "
                        f"last failure: {error}"
                    ) from controller_error
                failure = {
                    "event": "rejected",
                    "attempt": attempt_index,
                    "dic_segment": segment_index,
                    "start_fraction": global_start,
                    "end_fraction": global_end,
                    "time_increment": attempt_duration,
                    "reason": str(error),
                    "cutbacks_for_step": decision.cutbacks_for_current_step,
                    "next_segment_fraction": decision.next_increment_fraction,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                rejected_attempts.append(failure)
                record(failure)
                continue

            mechanical = np.asarray(result["final_mechanical"]).copy()
            chi = np.asarray(result["final_chi"]).copy()
            accepted_boundary = attempt_boundary.copy()
            local_fraction = next_fraction
            accepted_fractions.append(global_end)
            accepted_time_increments.append(attempt_duration)
            accumulated_newton += int(result["newton_iterations"])
            accumulated_krylov.extend(result["krylov_iterations"])
            accumulated_outer.extend(result["outer_iterations"])
            accumulated_mechanical.extend(result["mechanical_iterations"])
            accumulated_tangent_evaluations += int(result["material_tangent_evaluations"])
            accumulated_residual_evaluations += int(result["material_residual_evaluations"])
            accumulated_tangent_seconds += float(result["material_tangent_seconds"])
            accumulated_residual_seconds += float(result["material_residual_seconds"])
            last_result = result
            decision = controller.accept(
                LoadStepObservation(
                    converged=True,
                    newton_iterations=int(result["newton_iterations"]),
                )
            )
            record(
                {
                    "event": "accepted",
                    "attempt": attempt_index,
                    "accepted_subincrement": len(accepted_time_increments),
                    "dic_segment": segment_index,
                    "start_fraction": global_start,
                    "end_fraction": global_end,
                    "time_increment": attempt_duration,
                    "next_segment_fraction": decision.next_increment_fraction,
                    "newton_total": accumulated_newton,
                    "krylov_total": int(sum(accumulated_krylov)),
                    "mechanical_residual_norm": result["final_mechanical_residual_norm"],
                    "nonlocal_residual_norm": result["final_nonlocal_residual_norm"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )

    if last_result is None:
        raise RuntimeError("adaptive coupled path accepted no increment")
    last_result.update(
        {
            "elapsed_seconds": time.perf_counter() - started,
            "newton_iterations": accumulated_newton,
            "krylov_iterations": accumulated_krylov,
            "krylov_total": int(sum(accumulated_krylov)),
            "outer_iterations": accumulated_outer,
            "mechanical_iterations": accumulated_mechanical,
            "material_tangent_evaluations": accumulated_tangent_evaluations,
            "material_residual_evaluations": accumulated_residual_evaluations,
            "material_tangent_seconds": accumulated_tangent_seconds,
            "material_residual_seconds": accumulated_residual_seconds,
            "local_cutbacks": len(rejected_attempts),
            "accepted_subincrements": len(accepted_time_increments),
            "accepted_time_fractions": accepted_fractions,
            "accepted_time_increments": accepted_time_increments,
            "rejected_attempts": rejected_attempts,
        }
    )
    return last_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("native", "generic"), default="native")
    parser.add_argument("--material", choices=("j2", "srix", "meric"), default="j2")
    parser.add_argument(
        "--method",
        choices=("monolithic", "staggered", "production-nested", "both"),
        default="both",
    )
    parser.add_argument("--library", type=Path)
    parser.add_argument("--generic-library", type=Path)
    parser.add_argument(
        "--ebsd-orientation-h5",
        type=Path,
        default=DEFAULT_EBSD_ORIENTATION_H5,
        help="co-registered P43 EBSD orientation dataset used by both crystal laws",
    )
    parser.add_argument(
        "--total-duration-seconds",
        type=float,
        help=(
            "physical duration of the prescribed history; when omitted, crystal runs "
            "use normalized unit pseudo-time and are not rate-comparison qualified"
        ),
    )
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=DEFAULT_CROP)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--path-substeps", type=int, default=1)
    parser.add_argument(
        "--time-fractions",
        nargs="+",
        type=float,
        help=(
            "explicit normalized accepted path, including 0, 1 and every DIC knot; "
            "used to replay local/nonlocal crystal runs on the identical time grid"
        ),
    )
    parser.add_argument(
        "--adaptive-path-cutback",
        action="store_true",
        help=(
            "cut back only the failed DIC segment transactionally, then regrow "
            "the accepted step while preserving every original DIC state"
        ),
    )
    parser.add_argument("--maximum-path-substeps", type=int, default=16)
    parser.add_argument("--length-scale", type=float, default=0.05888)
    parser.add_argument("--coupling-modulus-mpa", type=float, default=5168.0)
    parser.add_argument("--krylov-relative-tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=1.0e-6,
        help="mechanical residual tolerance (use 1e-8 or tighter for qualification)",
    )
    parser.add_argument("--staggered-relaxation", type=float, default=1.0)
    parser.add_argument("--fd-strain-step", type=float, default=1.0e-7)
    parser.add_argument("--fd-chi-step", type=float, default=1.0e-7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.material == "srix" and args.backend != "generic":
        raise SystemExit("SRIX requires --backend generic")
    if args.material == "meric" and args.backend != "generic":
        raise SystemExit("Méric requires --backend generic")
    if args.material == "meric" and args.method == "production-nested":
        raise SystemExit("Méric production-nested is not wired yet")
    library = args.generic_library if args.backend == "generic" else args.library
    if library is None:
        parser.error("--library is required for native or --generic-library for generic")
    crop = tuple(args.crop_nodes)
    mesh = crop[1] - crop[0]
    if mesh != crop[3] - crop[2] or args.increments > 8:
        raise SystemExit("P43 crop must be square and increments must be <= 8")
    history, yield_stress, hardening, _ = _load_p43(crop)
    history = history[: args.increments + 1]
    if args.path_substeps < 1 or args.maximum_path_substeps < args.path_substeps:
        raise SystemExit("path subdivision limits are inconsistent")
    if args.time_fractions is not None and args.adaptive_path_cutback:
        raise SystemExit("explicit time fractions and adaptive path cutback are exclusive")
    if not 0.0 < args.staggered_relaxation <= 1.0:
        raise SystemExit("staggered relaxation must lie in (0, 1]")
    if not np.isfinite(args.fd_strain_step) or args.fd_strain_step <= 0:
        raise SystemExit("fd strain step must be finite and positive")
    if not np.isfinite(args.fd_chi_step) or args.fd_chi_step <= 0:
        raise SystemExit("fd chi step must be finite and positive")
    if args.total_duration_seconds is not None and (
        not np.isfinite(args.total_duration_seconds) or args.total_duration_seconds <= 0
    ):
        raise SystemExit("total duration must be finite and positive")
    total_duration = 1.0 if args.total_duration_seconds is None else args.total_duration_seconds
    grid = StructuredGrid2D(mesh, mesh, mesh * PIXEL_SIZE_MM, mesh * PIXEL_SIZE_MM)
    kinematics = TwoSubcellDiagnostic2D(grid)
    point_count = kinematics.material_point_count
    crystal_rotations = None
    orientation_provenance: dict[str, object] | None = None
    paired_parameter_manifest: dict[str, object] | None = None
    if args.material in {"srix", "meric"}:
        crystal_rotations, orientation_provenance = _load_ebsd_rotations(
            args.ebsd_orientation_h5,
            crop,
            states_per_pixel=point_count // (mesh * mesh),
        )
        law = "forest_rubin_srix" if args.material == "srix" else "meric_cailletaud"
        _, paired_parameter_manifest = resolve_paired_crystal_parameters(
            paired_parameter_set=PAIRED_PARAMETER_SET,
            law=law,
        )
    if args.material == "srix":
        assert crystal_rotations is not None
        virgin = _make_srix_material(
            library,
            point_count,
            args.coupling_modulus_mpa,
            crystal_rotations,
        )
    elif args.material == "meric":
        assert crystal_rotations is not None
        virgin = _make_meric_material(
            library,
            point_count,
            args.coupling_modulus_mpa,
            crystal_rotations,
        )
    elif args.backend == "generic":
        virgin = _make_material(library, point_count, yield_stress, hardening)
    else:
        virgin = _make_native_material(
            library, point_count, yield_stress, hardening, args.coupling_modulus_mpa
        )
    explicit_fractions = (
        None if args.time_fractions is None else np.asarray(args.time_fractions, dtype=float)
    )
    if explicit_fractions is not None:
        try:
            prescribed_history = _history_at_time_fractions(history, explicit_fractions)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        prescribed_time_increments = np.diff(explicit_fractions) * total_duration
    else:
        prescribed_history = None
        prescribed_time_increments = None
    virgin_trial = virgin.evaluate_in_plane(
        np.zeros((point_count, 3)),
        time_increment=(
            float(prescribed_time_increments[0])
            if prescribed_time_increments is not None
            else _time_increment(history, total_duration)
        ),
        consistent_tangent=True,
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
    actual_path_substeps = args.path_substeps
    failure_diagnostics: dict[str, object] | None = None
    while True:
        trial_history = (
            prescribed_history
            if prescribed_history is not None
            else _refine_history(history, actual_path_substeps)
        )
        try:
            mono = None
            stag = None
            failures: list[str] = []
            common = dict(
                library=library,
                history=trial_history,
                yield_stress=yield_stress,
                hardening=hardening,
                grid=grid,
                kinematics=kinematics,
                green=green,
                mechanical_inverse=mechanical_inverse,
                nonlocal_inverse=nonlocal_inverse,
                nonlocal_operator=nonlocal_operator,
                length_scale=args.length_scale,
                coupling_modulus_mpa=args.coupling_modulus_mpa,
                krylov_tolerance=args.krylov_relative_tolerance,
                absolute_tolerance=args.absolute_tolerance,
                backend=args.backend,
                staggered_relaxation=args.staggered_relaxation,
                fd_strain_step=args.fd_strain_step,
                fd_chi_step=args.fd_chi_step,
                material_model=args.material,
                crystal_rotations=crystal_rotations,
                total_duration=total_duration,
                time_increments=prescribed_time_increments,
            )
            if args.method in ("monolithic", "both"):
                try:
                    coupled_solver = (
                        _solve_sequence_with_local_cutback
                        if args.adaptive_path_cutback
                        else _solve_sequence
                    )
                    mono = coupled_solver(
                        **common,
                        method="monolithic",
                        progress_path=args.output.with_suffix(".monolithic.progress.jsonl"),
                    )
                except RuntimeError as error:
                    if isinstance(error, ProductionNestedFailureError):
                        failure_diagnostics = error.diagnostics
                    failures.append(f"monolithic: {error}")
            if args.method in ("staggered", "production-nested", "both"):
                staggered_method = (
                    "production-nested" if args.method == "production-nested" else "staggered"
                )
                try:
                    staggered_solver = (
                        _solve_sequence_with_local_cutback
                        if args.adaptive_path_cutback and staggered_method != "production-nested"
                        else _solve_sequence
                    )
                    stag = staggered_solver(
                        **common,
                        method=staggered_method,
                        progress_path=args.output.with_suffix(
                            f".{staggered_method}.progress.jsonl"
                        ),
                    )
                except RuntimeError as error:
                    if isinstance(error, ProductionNestedFailureError):
                        failure_diagnostics = error.diagnostics
                    failures.append(f"{staggered_method}: {error}")
            if failures:
                raise RuntimeError("; ".join(failures))
            break
        except RuntimeError as error:
            if isinstance(error, ProductionNestedFailureError):
                failure_diagnostics = error.diagnostics
            if failure_diagnostics is not None:
                failure_path = args.output.with_suffix(".failure.json")
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                failure_path.write_text(
                    json.dumps(
                        {
                            "status": "failed",
                            "backend": args.backend,
                            "method": args.method,
                            "crop_nodes": list(crop),
                            "path_substeps": actual_path_substeps,
                            "diagnostics": failure_diagnostics,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            raise
    total = time.perf_counter() - started
    archive_result = mono if mono is not None else stag
    default_fractions = (
        explicit_fractions
        if explicit_fractions is not None
        else np.linspace(0.0, 1.0, len(trial_history))
    )
    accepted_time_fractions = np.asarray(
        default_fractions
        if archive_result is None
        else archive_result.get("accepted_time_fractions", default_fractions),
        dtype=float,
    )
    accepted_time_increments = np.diff(accepted_time_fractions) * total_duration
    time_path_sha256 = hashlib.sha256(
        np.ascontiguousarray(accepted_time_fractions).tobytes()
    ).hexdigest()
    report = {
        "status": f"completed_coupled_{args.backend}_{args.material}_p43",
        "backend": args.backend,
        "crop_nodes": list(crop),
        "mesh": [mesh, mesh],
        "increments": args.increments,
        "path_substeps": actual_path_substeps,
        "effective_increments": len(accepted_time_fractions) - 1,
        "pixel_size_mm": PIXEL_SIZE_MM,
        "length_scale": args.length_scale,
        "coupling_modulus_mpa": args.coupling_modulus_mpa,
        "krylov_relative_tolerance": args.krylov_relative_tolerance,
        "absolute_tolerance": args.absolute_tolerance,
        "staggered_relaxation": args.staggered_relaxation,
        "fd_strain_step": args.fd_strain_step,
        "fd_chi_step": args.fd_chi_step,
        "b0_lambda": lambda_0,
        "b0_mu": mu_0,
        "b0_projection_error": projection_error,
        "total_elapsed_seconds": total,
        "method": args.method,
        "material": args.material,
        "time_history": {
            "kind": (
                "prescribed_normalized_pseudo_time"
                if explicit_fractions is not None and args.total_duration_seconds is None
                else (
                    "prescribed_physical_time"
                    if explicit_fractions is not None
                    else (
                        "adaptive_normalized_pseudo_time"
                        if args.adaptive_path_cutback
                        and args.total_duration_seconds is None
                        else (
                            "adaptive_physical_time"
                            if args.adaptive_path_cutback
                            else (
                                "normalized_pseudo_time"
                                if args.total_duration_seconds is None
                                else "physical_time"
                            )
                        )
                    )
                )
            ),
            "total_duration": total_duration,
            "minimum_time_increment": float(np.min(accepted_time_increments)),
            "maximum_time_increment": float(np.max(accepted_time_increments)),
            "accepted_time_fractions": accepted_time_fractions.tolist(),
            "path_sha256": time_path_sha256,
            "physical_time_history": args.total_duration_seconds is not None,
        },
        "orientation": orientation_provenance,
        "paired_parameter_manifest": paired_parameter_manifest,
        "comparison_contract": (
            None
            if args.material == "j2"
            else {
                "same_elasticity": True,
                "same_slip_systems": True,
                "same_interaction_matrix": True,
                "same_hardening": True,
                "same_orientation_field": True,
                "physical_time_history": args.total_duration_seconds is not None,
                "scientific_rate_comparison_qualified": False,
            }
        ),
        "monolithic": (
            None
            if mono is None
            else {
                k: v
                for k, v in mono.items()
                if k
                not in {
                    "final_mechanical",
                    "final_chi",
                    "final_stress",
                    "final_peeq",
                    "final_source",
                }
            }
        ),
        "staggered": (
            None
            if stag is None
            else {
                k: v
                for k, v in stag.items()
                if k
                not in {
                    "final_mechanical",
                    "final_chi",
                    "final_stress",
                    "final_peeq",
                    "final_source",
                }
            }
        ),
        "comparison": (
            None
            if mono is None or stag is None
            else {
                "time_ratio_staggered_over_monolithic": stag["elapsed_seconds"]
                / mono["elapsed_seconds"],
                "mechanical_linf": float(
                    np.max(np.abs(mono["final_mechanical"] - stag["final_mechanical"]))
                ),
                "chi_linf": float(np.max(np.abs(mono["final_chi"] - stag["final_chi"]))),
            }
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    solution_path = args.output.with_suffix(".solution.npz")
    solution_arrays: dict[str, np.ndarray] = {}
    for label, result in (("monolithic", mono), ("staggered", stag)):
        if result is not None:
            solution_arrays[f"{label}_mechanical"] = result["final_mechanical"]
            solution_arrays[f"{label}_chi"] = result["final_chi"]
            if "final_stress" in result:
                solution_arrays[f"{label}_stress"] = result["final_stress"]
                solution_arrays[f"{label}_peeq"] = result["final_peeq"]
                solution_arrays[f"{label}_source"] = result["final_source"]
    if solution_arrays:
        solution_arrays.update(
            {
                "metadata_material": np.asarray(args.material),
                "metadata_coupling_modulus_mpa": np.asarray(args.coupling_modulus_mpa),
                "metadata_length_scale_mm": np.asarray(args.length_scale),
                "metadata_effective_increments": np.asarray(len(accepted_time_fractions) - 1),
                "metadata_time_increment": np.asarray(
                    accepted_time_increments[0]
                    if np.allclose(accepted_time_increments, accepted_time_increments[0])
                    else np.nan
                ),
                "metadata_time_history_kind": np.asarray(report["time_history"]["kind"]),
                "metadata_total_duration": np.asarray(total_duration),
                "metadata_time_path_sha256": np.asarray(time_path_sha256),
                "metadata_accepted_time_fractions": accepted_time_fractions,
                "metadata_physical_time_history": np.asarray(
                    args.total_duration_seconds is not None
                ),
                "metadata_orientation_sha256": np.asarray(
                    ""
                    if orientation_provenance is None
                    else orientation_provenance["angles_sha256"]
                ),
            }
        )
        np.savez_compressed(solution_path, **solution_arrays)
        report["solution_archive"] = str(solution_path)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
