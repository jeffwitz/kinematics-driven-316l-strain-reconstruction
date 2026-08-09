"""Causal localisation of the 52-vs-46 Newton penalty (GPS vs reference, M20).

Every global hypothesis is refuted: trial purity bit-identical, Jv exact to
1e-9, forcing quasi-neutral, crossed stress/tangent showing the Newton count
follows the TANGENT field, and B^T C B spectra identical to 3 digits. The
penalty is associated with the tangent field encountered on the trajectory.
This script locates WHICH material points carry the tangent difference that
loses the iterations, and traces it back to its constitutive origin.

Checkpoint: increment 6 (default) of the M20 EBSD run, first Newton iteration
where the GPS and reference residual histories start to separate.

Test A -- Newton direction difference: assemble J_G = B^T C_G B and
J_R = B^T C_R B from the recorded tangents, solve BOTH with the SAME GPS
residual r = R_GPS(u_k) by a direct solver, and report |du_G - du_R|/|du_R|,
cos(theta), and the nonlinear reductions rho_G = |R_GPS(u_k + du_G)| /
|R_GPS(u_k)| and rho_R = |R_GPS(u_k + du_R)| / |R_GPS(u_k)|. This must
reproduce locally the 52/46 mechanism.

Test B -- spatial ranking: per material point, delta_C = C_GPS - C_REF,
delta_eps = B du (the recorded Newton strain direction), delta_sigma =
delta_C @ delta_eps, score = |delta_sigma|. Ranking, pixel/subcell
coordinates, cumulative share of the total action, and how many points
explain 50/80/90/95% of the norm of (J_G - J_R) du.

Test C -- surgical tangent substitution: run full M20 with the GPS stress,
GPS state, GPS law and GPS sub-stepping, but the REFERENCE tangent on the
top-k ranked points only, k in {0, 1, 5, 10, 25, 50, 100, all}. The Newton
count must drop from 52 toward ~47 as k grows if the localisation is right.

Test D -- constitutive characterisation of the top points: EBSD orientation,
stress, total/elastic/transverse strain, g/p/a slips, active-system mask,
local iterations, sub-stepping, and the distance of each slip system to its
activity threshold.

Test E -- state transplant: on the top 5 points, evaluate C_GPS(S_G) and
C_REF(S_G) on the SAME constitutive state S_G (GPS state exported by
variable name, imported into the reference), and the symmetric pair on S_R.
If |C_GPS(S) - C_REF(S)| / |C_REF(S)| <= 1e-10 on both states, the tangent
formulation is definitively acquitted and the cause is different internal
states amplified by crystal plasticity sensitivity.

Usage:

    .venv/bin/python scripts/diagnose_gps_tangent_localisation.py \
        --output validation/_generated/performance/gps_tangent_localisation_m20.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CROP_20X20 = (1610, 1630, 1075, 1095)
EBSD_ORIENTATION_H5 = "/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5"
PAIRED_PARAMETER_SET = "316l_guilhem2013_nasri2018_meric_srix_rate_1e-3"
GPS = "mfront-native-generalised-plane-stress"
REFERENCE = "mfront-3d-condensed-plane-stress"
SUBSTITUTION_TOP_K = (0, 1, 5, 10, 25, 50, 100, None)


class RecordingMaterial:
    """Record every solver call: strain, stress, tangent, ISV, snapshots."""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.calls: list[dict[str, object]] = []
        self.committed_snapshots: list[object] = []
        self._last_substeps = 0

    def _record(
        self,
        strain: np.ndarray,
        trial: object,
        time_increment: float,
    ) -> None:
        tangent = getattr(trial, "tangent_in_plane_mpa", None)
        observables = getattr(trial, "observables", None) or {}
        substeps = int(getattr(self._inner, "_substep_uses", 0))
        self.calls.append(
            {
                "strain": np.asarray(strain, dtype=float).copy(),
                "stress": np.asarray(trial.stress_in_plane_mpa, dtype=float).copy(),
                "tangent": (
                    None
                    if tangent is None
                    else np.asarray(tangent, dtype=float).copy()
                ),
                "time_increment": float(time_increment),
                "substeps": substeps - self._last_substeps,
                "committed_before": len(self.committed_snapshots),
                "plastic_slip": (
                    None
                    if "plastic_slip" not in observables
                    else np.asarray(observables["plastic_slip"], dtype=float).copy()
                ),
                "equivalent_plastic_slip": (
                    None
                    if "equivalent_plastic_slip" not in observables
                    else np.asarray(
                        observables["equivalent_plastic_slip"], dtype=float
                    ).copy()
                ),
                "back_strain": (
                    None
                    if "back_strain" not in observables
                    else np.asarray(observables["back_strain"], dtype=float).copy()
                ),
                "substep_mask": getattr(self._inner, "last_substep_mask", None),
                "substep_divisions": getattr(self._inner, "last_substep_divisions", None),
                "shadow_diagnostics": getattr(
                    self._inner, "last_shadow_diagnostics", None
                ),
                "composite_fd_diagnostics": getattr(
                    self._inner, "last_composite_fd_diagnostics", None
                ),
            }
        )
        self._last_substeps = substeps

    def evaluate(
        self,
        in_plane_strain: np.ndarray,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> object:
        strain = np.asarray(in_plane_strain, dtype=float)
        trial = self._inner.evaluate(
            strain, time_increment=time_increment, consistent_tangent=consistent_tangent
        )
        self._record(strain, trial, time_increment)
        return trial

    def evaluate_in_plane(
        self,
        in_plane_strain: np.ndarray,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> object:
        strain = np.asarray(in_plane_strain, dtype=float)
        trial = self._inner.evaluate_in_plane(
            strain, time_increment=time_increment, consistent_tangent=consistent_tangent
        )
        self._record(strain, trial, time_increment)
        return trial

    def evaluate_in_plane_response(
        self,
        in_plane_strain: np.ndarray,
        *,
        time_increment: float,
        response_level: str = "tangent",
        consistent_tangent: bool = True,
    ) -> object:
        strain = np.asarray(in_plane_strain, dtype=float)
        from fem_inhouse.core.plane_stress_material import evaluate_in_plane_response

        trial = evaluate_in_plane_response(
            self._inner,
            strain,
            time_increment=time_increment,
            response_level=response_level,
            consistent_tangent=consistent_tangent,
        )
        self._record(strain, trial, time_increment)
        return trial

    def commit(self) -> None:
        self._inner.commit()
        self.committed_snapshots.append(self._inner.snapshot_state())

    def revert(self) -> None:
        self._inner.revert()

    @property
    def point_count(self) -> int:
        return self._inner.point_count

    @property
    def timing_statistics(self) -> object:
        return getattr(self._inner, "timing_statistics", None)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


def _build_material(
    backend: str,
    arguments: argparse.Namespace,
    grid: object,
    yield_stress: np.ndarray,
    coefficient: np.ndarray,
) -> object:
    from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch
    from scripts.qualify_crystal_tet2_p43 import _load_ebsd_orientation_crop

    constitutive_options = {
        "paired_parameter_set": arguments.paired_parameter_set,
        "crystal_orientation": {
            "mode": "ebsd",
            "euler_bunge_deg": _load_ebsd_orientation_crop(
                arguments.ebsd_orientation_h5, arguments.crop_nodes
            )[0],
        },
    }
    shadow_scope = getattr(arguments, "shadow_scope", None)
    if backend == GPS and shadow_scope is not None:
        constitutive_options.update(
            {
                "gps_shadow_tangent": True,
                "gps_shadow_tangent_scope": shadow_scope,
            }
        )
    if backend == GPS and bool(getattr(arguments, "composite_fd_tangent", False)):
        constitutive_options.update(
            {
                "gps_composite_fd_tangent": True,
                "gps_composite_fd_step": float(
                    getattr(arguments, "composite_fd_step", 1.0e-6)
                ),
            }
        )
    return create_plane_stress_material_batch(
        backend,
        np.repeat(yield_stress, 2),
        np.repeat(coefficient, 2),
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1_000,
        first_positive_plastic_strain=1.0e-6,
        mfront_library=arguments.library,
        mfront_threads=arguments.mfront_threads,
        mfront_behaviour_id="fcc_forest_rubin_srix",
        local_plane_stress_options={
            "local_condition_check_mode": "on_failure",
            "local_transverse_predictor": "tangent",
        },
        constitutive_options=constitutive_options,
    )


def _run_backend(
    backend: str,
    arguments: argparse.Namespace,
    grid: object,
    yield_stress: np.ndarray,
    coefficient: np.ndarray,
    boundary: np.ndarray,
) -> tuple[object, object, object]:
    from fem_inhouse.spectral2d.newton_two_state import (
        EBISpectralSolverConfig,
        solve_two_state_dirichlet_plane_stress,
    )
    from fem_inhouse.spectral2d.transforms import SpectralTransformConfig

    material = _build_material(backend, arguments, grid, yield_stress, coefficient)
    recording = RecordingMaterial(material)
    history = np.stack(
        [
            fraction * boundary
            for fraction in np.linspace(0.0, 1.0, arguments.increments + 1)
        ]
    )
    config = EBISpectralSolverConfig(
        relative_equilibrium_tolerance=1e-8,
        maximum_newton_iterations=arguments.maximum_newton_iterations,
        krylov_method="lgmres",
        krylov_recycling=True,
        linear_tolerance_mode="eisenstat_walker",
        verify_final_state=False,
        transform=SpectralTransformConfig(
            backend="fftw",
            workers=1,
            fftw_planner_effort="measure",
            fftw_planning_time_limit_s=2.0,
            fftw_use_wisdom=False,
        ),
    )
    result = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=recording,
        boundary_displacement_history=history[: arguments.increments + 1],
        config=config,
    )
    return material, recording, result


def _checkpoint_calls(
    recording: RecordingMaterial,
    increment: int,
) -> list[dict[str, object]]:
    committed = increment - 1
    return [
        call
        for call in recording.calls
        if call["committed_before"] == committed
    ]


def _shadow_runtime_summary(recording: RecordingMaterial) -> list[dict[str, object]]:
    """Make the pointwise shadow/substep telemetry JSON-safe."""

    rows: list[dict[str, object]] = []
    for index, call in enumerate(recording.calls):
        diagnostic = call.get("shadow_diagnostics")
        if not isinstance(diagnostic, dict):
            continue
        row: dict[str, object] = {
            "call": index,
            "substep": np.asarray(call["substep_mask"]).tolist(),
            "divisions": np.asarray(call["substep_divisions"]).tolist(),
            "scope": diagnostic.get("scope"),
            "tangent_relative_error": np.asarray(
                diagnostic.get("tangent_relative_error", [])
            ).tolist(),
        }
        state_differences = diagnostic.get("state_differences", {})
        row["state_differences"] = {
            str(name): np.asarray(values).tolist()
            for name, values in state_differences.items()
        }
        rows.append(row)
    return rows


def _composite_fd_runtime_summary(recording: RecordingMaterial) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, call in enumerate(recording.calls):
        diagnostic = call.get("composite_fd_diagnostics")
        if not isinstance(diagnostic, dict):
            continue
        entries = []
        for item in diagnostic.get("points", []):
            entries.append(
                {
                    "point": int(item["point"]),
                    "divisions": int(item["divisions"]),
                    "partition_unchanged": bool(item["partition_unchanged"]),
                    "tangent": np.asarray(item["tangent"]).tolist(),
                }
            )
        rows.append(
            {
                "call": index,
                "step": float(diagnostic["step"]),
                "points": entries,
            }
        )
    return rows


def _residual_at(
    material: object,
    snapshot: object,
    strain: np.ndarray,
    time_increment: float,
    grid: object,
) -> np.ndarray:
    """R(u) = div(sigma(eps(u))) re-integrated from the exact same snapshot."""

    from fem_inhouse.spectral2d.newton_two_state import TwoSubcellDiagnostic2D

    kinematics = TwoSubcellDiagnostic2D(grid)
    material.restore_state(snapshot)
    trial = material.evaluate(
        strain.reshape(-1, 3),
        time_increment=time_increment,
        consistent_tangent=True,
    )
    stress = np.asarray(trial.stress_in_plane_mpa).reshape(*grid.pixel_shape, 2, 3)
    buffer = np.empty((*grid.node_shape, 2), dtype=np.float64)
    kinematics.divergence_from_sample_stress_into(stress, buffer)
    return buffer


def _assemble_jacobian(
    tangent: np.ndarray,
    grid: object,
    plan: object,
    kinematics: object,
    workspace: object,
) -> np.ndarray:
    """Full B^T C B matrix from a recorded per-point tangent field."""

    from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior_into

    tangent_field = np.asarray(tangent).reshape(*grid.pixel_shape, 2, 3, 3)
    size = 2 * (grid.nx - 1) * (grid.ny - 1)

    def action(vector: np.ndarray) -> np.ndarray:
        unpack_interior_into(vector, grid, workspace.nodal_increment)
        delta_stress = np.einsum(
            "xyqij,xyqj->xyqi",
            tangent_field,
            kinematics.strain_samples(workspace.nodal_increment),
        )
        kinematics.divergence_from_sample_stress_into(
            delta_stress, workspace.nodal_force
        )
        return pack_interior(workspace.nodal_force)

    columns = []
    for index in range(size):
        basis = np.zeros(size, dtype=float)
        basis[index] = 1.0
        columns.append(action(basis))
    return np.stack(columns, axis=1)


def test_a(
    material_gps: object,
    recording_gps: RecordingMaterial,
    recording_ref: RecordingMaterial,
    grid: object,
    increment: int,
    arguments: argparse.Namespace,
) -> dict[str, object]:
    """Newton directions from J_G and J_R on the SAME GPS residual."""

    from scipy.linalg import solve

    from fem_inhouse.spectral2d.newton_ebi import pack_interior
    from fem_inhouse.spectral2d.newton_two_state import (
        TwoStateJacobianWorkspace,
        TwoSubcellDiagnostic2D,
    )
    from fem_inhouse.spectral2d.transform_factory import create_full_dirichlet_dsti_plan
    from fem_inhouse.spectral2d.transforms import SpectralTransformConfig

    calls_gps = _checkpoint_calls(recording_gps, increment)
    calls_ref = _checkpoint_calls(recording_ref, increment)
    if not calls_gps or not calls_ref:
        raise RuntimeError(f"no calls recorded at checkpoint increment {increment}")
    first_gps = calls_gps[0]
    first_ref = calls_ref[0]
    snapshot_gps = recording_gps.committed_snapshots[increment - 2]
    strain_k = np.asarray(first_gps["strain"], dtype=float)
    dt = float(first_gps["time_increment"])

    residual = _residual_at(
        material_gps, snapshot_gps, strain_k, dt, grid
    )
    r = pack_interior(residual)
    # The solver's own measure: the INTERIOR residual only. The full nodal
    # field carries the Dirichlet boundary, which is constant under interior
    # perturbations and would mask the reduction.
    interior_norm = float(np.linalg.norm(r))

    plan = create_full_dirichlet_dsti_plan(
        grid,
        SpectralTransformConfig(
            backend="fftw",
            workers=1,
            fftw_planner_effort="measure",
            fftw_planning_time_limit_s=2.0,
            fftw_use_wisdom=False,
        ),
    )
    kinematics = TwoSubcellDiagnostic2D(grid)
    workspace = TwoStateJacobianWorkspace.create(grid)
    jacobian_g = _assemble_jacobian(
        first_gps["tangent"], grid, plan, kinematics, workspace
    )
    jacobian_r = _assemble_jacobian(
        first_ref["tangent"], grid, plan, kinematics, workspace
    )
    du_g = solve(jacobian_g, -r)
    du_r = solve(jacobian_r, -r)

    # Nonlinear reductions of the GPS residual under both corrections, on the
    # INTERIOR norm (the solver's own measure).
    def reduction(du: np.ndarray) -> float:
        from fem_inhouse.spectral2d.newton_ebi import unpack_interior_into

        unpack_interior_into(du, grid, workspace.nodal_increment)
        delta_eps = kinematics.strain_samples(workspace.nodal_increment)
        residual_new = _residual_at(
            material_gps,
            snapshot_gps,
            (strain_k.reshape(*grid.pixel_shape, 2, 3) + delta_eps).reshape(-1, 3),
            dt,
            grid,
        )
        return float(
            np.linalg.norm(pack_interior(residual_new))
            / max(interior_norm, 1.0e-30)
        )

    rho_g = reduction(du_g)
    rho_r = reduction(du_r)
    norm_r = interior_norm
    cos_theta = float(
        np.dot(du_g, du_r) / max(np.linalg.norm(du_g) * np.linalg.norm(du_r), 1.0e-30)
    )
    return {
        "increment": increment,
        "norm_r": norm_r,
        "norm_du_g": float(np.linalg.norm(du_g)),
        "norm_du_r": float(np.linalg.norm(du_r)),
        "relative_direction_difference": float(
            np.linalg.norm(du_g - du_r) / max(np.linalg.norm(du_r), 1.0e-30)
        ),
        "cos_theta": cos_theta,
        "rho_gps": rho_g,
        "rho_ref": rho_r,
        "gps_residual_after_gps_correction": rho_g,
        "gps_residual_after_ref_correction": rho_r,
        "calls_gps": len(calls_gps),
        "calls_ref": len(calls_ref),
    }


def test_b(
    recording_gps: RecordingMaterial,
    recording_ref: RecordingMaterial,
    grid: object,
    increment: int,
) -> dict[str, object]:
    """Spatial ranking of the per-point tangent-action difference."""

    calls_gps = _checkpoint_calls(recording_gps, increment)
    calls_ref = _checkpoint_calls(recording_ref, increment)
    first_gps = calls_gps[0]
    first_ref = calls_ref[0]
    c_gps = np.asarray(first_gps["tangent"], dtype=float)
    c_ref = np.asarray(first_ref["tangent"], dtype=float)
    # The Newton strain direction: the first accepted step of the GPS run.
    if len(calls_gps) > 1:
        delta_eps = np.asarray(calls_gps[1]["strain"]) - np.asarray(
            first_gps["strain"]
        )
    else:
        delta_eps = np.zeros_like(c_gps[..., 0])
    delta_c = c_gps - c_ref
    delta_sigma = np.einsum("nij,nj->ni", delta_c, delta_eps)
    scores = np.linalg.norm(delta_sigma, axis=1)
    order = np.argsort(scores)[::-1]
    total = float(scores.sum())
    cumulative = np.cumsum(scores[order]) / max(total, 1.0e-30)
    thresholds = {}
    for fraction in (0.50, 0.80, 0.90, 0.95):
        count = int(np.searchsorted(cumulative, fraction) + 1)
        thresholds[f"{fraction:.2f}"] = count
    # The action (J_G - J_R) du in the nodal sense, for reference.
    action_norm = float(
        np.linalg.norm(np.einsum("nij,nj->ni", delta_c, delta_eps))
    )
    return {
        "increment": increment,
        "point_count": len(scores),
        "total_score": total,
        "action_delta_sigma_norm": action_norm,
        "points_for_fraction": thresholds,
        "ranking": [
            {
                "point": int(index),
                "pixel_x": int(index // 2 % grid.pixel_shape[0]),
                "pixel_y": int(index // 2 // grid.pixel_shape[0]),
                "subcell": int(index % 2),
                "score": float(scores[index]),
                "cumulative_fraction": float(cumulative[position]),
                "delta_sigma": delta_sigma[index].tolist(),
            }
            for position, index in enumerate(order)
        ],
    }


class SubstitutedTangentBackend:
    """GPS stress/state/law with the reference tangent on selected points."""

    def __init__(self, gps: object, reference: object, top_points: list[int]) -> None:
        self._gps = gps
        self._reference = reference
        self._top = set(top_points)
        self._stress_from = "gps"

    def _pair(
        self,
        in_plane_strain: np.ndarray,
        *,
        time_increment: float,
        consistent_tangent: bool,
    ) -> tuple[object, object]:
        gps_trial = self._gps.evaluate_in_plane(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )
        ref_trial = self._reference.evaluate_in_plane(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )
        return gps_trial, ref_trial

    def _cross(self, gps_trial: object, ref_trial: object, consistent_tangent: bool):
        from fem_inhouse.core.plane_stress_material import InPlaneConstitutiveTrial

        stress = gps_trial.stress_in_plane_mpa
        if consistent_tangent and ref_trial.tangent_in_plane_mpa is not None:
            tangent = gps_trial.tangent_in_plane_mpa.copy()
            points = np.fromiter(self._top, dtype=int)
            tangent[points, :, :] = ref_trial.tangent_in_plane_mpa[points, :, :]
        else:
            tangent = gps_trial.tangent_in_plane_mpa if consistent_tangent else None
        return InPlaneConstitutiveTrial(
            stress_in_plane_mpa=stress,
            tangent_in_plane_mpa=tangent,
            observables=gps_trial.observables,
            local_plane_stress_iterations=gps_trial.local_plane_stress_iterations,
        )

    def evaluate(
        self,
        in_plane_strain: np.ndarray,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> object:
        return self._cross(
            *self._pair(
                in_plane_strain,
                time_increment=time_increment,
                consistent_tangent=consistent_tangent,
            ),
            consistent_tangent,
        )

    def evaluate_in_plane(
        self,
        in_plane_strain: np.ndarray,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> object:
        return self.evaluate(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )

    def evaluate_in_plane_response(
        self,
        in_plane_strain: np.ndarray,
        *,
        time_increment: float,
        response_level: str = "tangent",
        consistent_tangent: bool = True,
    ) -> object:
        return self.evaluate(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )

    def complete_trial(self, trial: object) -> object:
        return self._gps.complete_trial(trial)

    def commit(self) -> None:
        self._gps.commit()
        self._reference.commit()

    def revert(self) -> None:
        self._gps.revert()
        self._reference.revert()

    def accept_global_trial(self) -> None:
        self._gps.accept_global_trial()
        self._reference.accept_global_trial()

    def snapshot_state(self) -> tuple[object, object]:
        return (self._gps.snapshot_state(), self._reference.snapshot_state())

    def restore_state(self, snapshot: tuple[object, object]) -> None:
        self._gps.restore_state(snapshot[0])
        self._reference.restore_state(snapshot[1])

    @property
    def point_count(self) -> int:
        return self._gps.point_count

    @property
    def backend_name(self) -> str:
        return f"gps-tangent-substituted-{len(self._top)}"

    @property
    def completion_strategy(self) -> str:
        return "gps_tangent_substituted"

    @property
    def linear_system_matrix_type(self) -> str:
        return self._gps.linear_system_matrix_type

    @property
    def statistics(self) -> object:
        return self._gps.statistics

    @property
    def timing_statistics(self) -> object:
        return self._gps.timing_statistics


def _run_substitution(
    top_points: list[int],
    arguments: argparse.Namespace,
    grid: object,
    yield_stress: np.ndarray,
    coefficient: np.ndarray,
    boundary: np.ndarray,
) -> dict[str, object]:
    from fem_inhouse.spectral2d.newton_two_state import (
        EBISpectralSolverConfig,
        solve_two_state_dirichlet_plane_stress,
    )
    from fem_inhouse.spectral2d.transforms import SpectralTransformConfig

    material = SubstitutedTangentBackend(
        _build_material(GPS, arguments, grid, yield_stress, coefficient),
        _build_material(REFERENCE, arguments, grid, yield_stress, coefficient),
        top_points,
    )
    history = np.stack(
        [
            fraction * boundary
            for fraction in np.linspace(0.0, 1.0, arguments.increments + 1)
        ]
    )
    config = EBISpectralSolverConfig(
        relative_equilibrium_tolerance=1e-8,
        maximum_newton_iterations=arguments.maximum_newton_iterations,
        krylov_method="lgmres",
        krylov_recycling=True,
        linear_tolerance_mode="eisenstat_walker",
        verify_final_state=False,
        transform=SpectralTransformConfig(
            backend="fftw",
            workers=1,
            fftw_planner_effort="measure",
            fftw_planning_time_limit_s=2.0,
            fftw_use_wisdom=False,
        ),
    )
    result = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=material,
        boundary_displacement_history=history[: arguments.increments + 1],
        config=config,
    )
    return {
        "top_k": len(top_points),
        "newton_iterations": int(sum(result.diagnostics.iterations_per_increment)),
        "accepted_increments": len(result.diagnostics.iterations_per_increment),
        "iterations_per_increment": list(result.diagnostics.iterations_per_increment),
        "succeeded": bool(np.isfinite(result.displacement).all()),
    }


def test_d(
    recording_gps: RecordingMaterial,
    recording_ref: RecordingMaterial,
    ranking: list[dict[str, object]],
    grid: object,
    increment: int,
    top_count: int = 20,
) -> dict[str, object]:
    """Constitutive state of the highest-scoring points, GPS vs reference."""

    calls_gps = _checkpoint_calls(recording_gps, increment)
    calls_ref = _checkpoint_calls(recording_ref, increment)
    first_gps = calls_gps[0]
    first_ref = calls_ref[0]
    rows = []
    for item in ranking[:top_count]:
        point = int(item["point"])
        rows.append(
            {
                "point": point,
                "pixel_x": item["pixel_x"],
                "pixel_y": item["pixel_y"],
                "subcell": item["subcell"],
                "score": item["score"],
                "gps_plastic_slip": (
                    None
                    if first_gps["plastic_slip"] is None
                    else first_gps["plastic_slip"][point].tolist()
                ),
                "ref_plastic_slip": (
                    None
                    if first_ref["plastic_slip"] is None
                    else first_ref["plastic_slip"][point].tolist()
                ),
                "gps_equivalent_slip": (
                    None
                    if first_gps["equivalent_plastic_slip"] is None
                    else first_gps["equivalent_plastic_slip"][point].tolist()
                ),
                "ref_equivalent_slip": (
                    None
                    if first_ref["equivalent_plastic_slip"] is None
                    else first_ref["equivalent_plastic_slip"][point].tolist()
                ),
                "gps_stress": first_gps["stress"][point].tolist(),
                "ref_stress": first_ref["stress"][point].tolist(),
            }
        )
    return {"top_count": top_count, "rows": rows}


def test_e(
    material_gps: object,
    material_ref: object,
    recording_gps: RecordingMaterial,
    recording_ref: RecordingMaterial,
    grid: object,
    increment: int,
    ranking: list[dict[str, object]],
    top_count: int = 5,
) -> dict[str, object]:
    """Transplant the GPS state into the reference and vice versa, top points."""

    results = []
    for item in ranking[:top_count]:
        point = int(item["point"])
        row: dict[str, object] = {"point": point, "score": item["score"]}
        # E1: C_GPS(S_G) vs C_REF(S_G) -- reference evaluated on the GPS state.
        row.update(
            _evaluate_on_transplanted_state(
                material_gps,
                material_ref,
                recording_gps,
                recording_ref,
                grid,
                increment,
                point,
                state_from="gps",
            )
        )
        # E2: C_GPS(S_R) vs C_REF(S_R) -- GPS evaluated on the reference state.
        row.update(
            _evaluate_on_transplanted_state(
                material_gps,
                material_ref,
                recording_gps,
                recording_ref,
                grid,
                increment,
                point,
                state_from="reference",
            )
        )
        results.append(row)
    return {"top_count": top_count, "rows": results}


def _evaluate_on_transplanted_state(
    material_gps: object,
    material_ref: object,
    recording_gps: RecordingMaterial,
    recording_ref: RecordingMaterial,
    grid: object,
    increment: int,
    point: int,
    *,
    state_from: str,
) -> dict[str, object]:
    """Evaluate both backends on the SAME transplanted state at one point.

    The donor state (GPS or reference) is exported from its committed manager
    by variable name and imported into the other backend's committed manager.
    The donor is evaluated from its own snapshot (unchanged); the recipient is
    evaluated from the transplanted state, so both tangents come from the SAME
    constitutive state at that point.
    """

    from fem_inhouse.core.mfront import _declared_internal_slices

    snapshot_gps = recording_gps.committed_snapshots[increment - 2]
    snapshot_ref = recording_ref.committed_snapshots[increment - 2]
    calls_gps = _checkpoint_calls(recording_gps, increment)
    strain = np.asarray(calls_gps[0]["strain"], dtype=float)
    dt = float(calls_gps[0]["time_increment"])

    donor = material_gps if state_from == "gps" else material_ref
    recipient = material_ref if state_from == "gps" else material_gps
    donor_snapshot = snapshot_gps if state_from == "gps" else snapshot_ref
    recipient_snapshot = snapshot_ref if state_from == "gps" else snapshot_gps

    # The MGIS native attributes live on the batch itself for the GPS backend
    # and on `_bridge` for the condensed reference.
    def native(obj: object) -> object:
        return getattr(obj, "_bridge", None) or obj

    donor_native = native(donor)
    recipient_native = native(recipient)
    donor_manager = donor_native._manager
    recipient_manager = recipient_native._manager
    donor_behaviour = donor_native._behaviour
    recipient_behaviour = recipient_native._behaviour
    donor_slices = _declared_internal_slices(
        donor_native._mgis,
        donor_behaviour,
        donor_native._mgis.Hypothesis.Tridimensional,
        donor_native._specification,
    )
    recipient_slices = _declared_internal_slices(
        recipient_native._mgis,
        recipient_behaviour,
        recipient_native._mgis.Hypothesis.Tridimensional,
        recipient_native._specification,
    )

    # Export the donor's committed internal state at the point, by variable
    # name. The gradient (total strain) is NOT transplanted: it is the input
    # of the evaluation, and both backends are evaluated at the same imposed
    # in-plane strain below -- the condensed reference must solve its own
    # transverse closure, and imposing the GPS transverses makes its local
    # Newton fail (verified).
    donor_state = np.asarray(donor_manager.s0.internal_state_variables)[point, :]
    exported: dict[str, np.ndarray] = {}
    for name, position in donor_slices.items():
        if name in recipient_slices:
            exported[name] = donor_state[position].copy()

    # Import into the recipient's committed manager, by name.
    recipient.restore_state(recipient_snapshot)
    recipient_state = np.asarray(recipient_manager.s0.internal_state_variables)
    for name, values in exported.items():
        position = recipient_slices[name]
        recipient_state[point, position] = values

    donor.restore_state(donor_snapshot)
    donor_trial = donor.evaluate(
        strain.reshape(-1, 3),
        time_increment=dt,
        consistent_tangent=True,
    )
    recipient_trial = recipient.evaluate(
        strain.reshape(-1, 3),
        time_increment=dt,
        consistent_tangent=True,
    )
    donor_tangent = np.asarray(donor_trial.tangent_in_plane_mpa)[point]
    recipient_tangent = np.asarray(recipient_trial.tangent_in_plane_mpa)[point]
    relative = float(
        np.linalg.norm(donor_tangent - recipient_tangent)
        / max(np.linalg.norm(recipient_tangent), 1.0e-30)
    )
    donor.restore_state(donor_snapshot)
    recipient.restore_state(recipient_snapshot)
    return {
        f"{state_from}_relative_tangent_difference": relative,
        f"{state_from}_donor_tangent": donor_tangent.tolist(),
        f"{state_from}_recipient_tangent": recipient_tangent.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=CROP_20X20)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--library", default="build/mfront/src/libBehaviour.so")
    parser.add_argument("--ebsd-orientation-h5", type=Path, default=Path(EBSD_ORIENTATION_H5))
    parser.add_argument("--paired-parameter-set", default=PAIRED_PARAMETER_SET)
    parser.add_argument("--mfront-threads", type=int, default=4)
    parser.add_argument("--maximum-newton-iterations", type=int, default=40)
    parser.add_argument("--checkpoint-increment", type=int, default=6)
    parser.add_argument(
        "--shadow-scope",
        choices=("all", "substepped", "non_substepped"),
        default=None,
        help="Use the runtime raw shadow tangent on the selected GPS points.",
    )
    parser.add_argument("--composite-fd-tangent", action="store_true")
    parser.add_argument("--composite-fd-step", type=float, default=1.0e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/_generated/performance/gps_tangent_localisation_m20.json"),
    )
    parser.add_argument(
        "--points-output",
        type=Path,
        default=Path("validation/_generated/performance/gps_tangent_localisation_points.csv"),
    )
    parser.add_argument(
        "--substitution-output",
        type=Path,
        default=Path("validation/_generated/performance/gps_tangent_substitution_m20.json"),
    )
    parser.add_argument("--skip-substitution", action="store_true")
    arguments = parser.parse_args()

    from scripts.benchmark_tri2_j2_krylov import _load_case

    mesh = arguments.crop_nodes[1] - arguments.crop_nodes[0]
    grid, _, yield_stress, coefficient, boundary = _load_case(
        mesh, arguments.crop_nodes
    )

    material_gps, recording_gps, result_gps = _run_backend(
        GPS, arguments, grid, yield_stress, coefficient, boundary
    )
    material_ref, recording_ref, result_ref = _run_backend(
        REFERENCE, arguments, grid, yield_stress, coefficient, boundary
    )

    increment = arguments.checkpoint_increment
    test_a_result = test_a(
        material_gps, recording_gps, recording_ref, grid, increment, arguments
    )
    test_b_result = test_b(recording_gps, recording_ref, grid, increment)
    test_d_result = test_d(
        recording_gps, recording_ref, test_b_result["ranking"], grid, increment
    )
    test_e_result = test_e(
        material_gps,
        material_ref,
        recording_gps,
        recording_ref,
        grid,
        increment,
        test_b_result["ranking"],
    )

    # CSV of the ranking.
    ranking = test_b_result["ranking"]
    if arguments.points_output is not None:
        with arguments.points_output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "rank",
                    "point",
                    "pixel_x",
                    "pixel_y",
                    "subcell",
                    "score",
                    "cumulative_fraction",
                ],
            )
            writer.writeheader()
            for position, item in enumerate(ranking, start=1):
                writer.writerow(
                    {
                        "rank": position,
                        "point": item["point"],
                        "pixel_x": item["pixel_x"],
                        "pixel_y": item["pixel_y"],
                        "subcell": item["subcell"],
                        "score": f"{item['score']:.6e}",
                        "cumulative_fraction": f"{item['cumulative_fraction']:.6f}",
                    }
                )

    payload = {
        "schema_version": 1,
        "configuration": {
            "crop_nodes": arguments.crop_nodes,
            "increments": arguments.increments,
            "mfront_threads": arguments.mfront_threads,
            "checkpoint_increment": increment,
            "composite_fd_tangent": arguments.composite_fd_tangent,
            "composite_fd_step": arguments.composite_fd_step,
        },
        "reference": {
            "gps_newton": int(sum(result_gps.diagnostics.iterations_per_increment)),
            "ref_newton": int(sum(result_ref.diagnostics.iterations_per_increment)),
            "gps_per_increment": list(result_gps.diagnostics.iterations_per_increment),
            "ref_per_increment": list(result_ref.diagnostics.iterations_per_increment),
        },
        "test_a": test_a_result,
        "test_b": {
            key: value
            for key, value in test_b_result.items()
            if key != "ranking"
        },
        "test_d": test_d_result,
        "test_e": test_e_result,
        "shadow_runtime": _shadow_runtime_summary(recording_gps),
        "composite_fd_runtime": _composite_fd_runtime_summary(recording_gps),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(
        f"reference: GPS {payload['reference']['gps_newton']} Newton, "
        f"REF {payload['reference']['ref_newton']} Newton | "
        f"checkpoint inc {increment}"
    )
    a = test_a_result
    print(
        f"test A: |du_G - du_R|/|du_R| = {a['relative_direction_difference']:.3e}, "
        f"cos(theta) = {a['cos_theta']:.6f}, "
        f"rho_GPS = {a['rho_gps']:.3e}, rho_REF = {a['rho_ref']:.3e}"
    )
    print(
        "test B: points for 50/80/90/95% of the action: "
        + " / ".join(
            f"{frac}: {count}"
            for frac, count in test_b_result["points_for_fraction"].items()
        )
    )
    print(f"test D: top points characterised, {len(test_d_result['rows'])} rows")
    for row in test_e_result["rows"]:
        print(
            f"  test E point {row['point']} (score {row['score']:.2e}): "
            f"on S_G {row.get('gps_relative_tangent_difference', float('nan')):.3e}, "
            f"on S_R {row.get('reference_relative_tangent_difference', float('nan')):.3e}"
        )

    if not arguments.skip_substitution:
        top_points = [
            int(item["point"]) for item in ranking
        ]
        substitution_records = []
        for k in SUBSTITUTION_TOP_K:
            selected = top_points if k is None else top_points[:k]
            record = _run_substitution(
                selected, arguments, grid, yield_stress, coefficient, boundary
            )
            substitution_records.append(record)
            print(
                f"substitution top {record['top_k']}: "
                f"{record['newton_iterations']} Newton"
            )
        substitution_payload = {
            "schema_version": 1,
            "configuration": {
                "crop_nodes": arguments.crop_nodes,
                "increments": arguments.increments,
                "checkpoint_increment": increment,
            },
            "records": substitution_records,
        }
        arguments.substitution_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.substitution_output.write_text(
            json.dumps(substitution_payload, indent=2, sort_keys=True) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
