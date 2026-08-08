"""Global Jv against finite differences on a real deep-increment snapshot.

Every local-level hypothesis for the 85-vs-57 Newton penalty is eliminated:
tangent identical to 1e-16 at matched states, sub-stepping path acquitted,
closure tolerance and epsilon eliminated, and the GPS trial is PURE
(evaluate(A)-evaluate(B)-evaluate(A) bit-identical, accept_global_trial
neutral -- scripts/diagnose_gps_trial_purity.py). What has NOT been tested is
the JACOBIAN THE SOLVER APPLIES, assembled from the material tangent through
the spectral kinematics: `tangent_action(v) = div(C . B v)`.

This script compares that operator against a genuine finite difference on a
REAL deep increment (5-8, where the GPS convergence leaves the reference's):

    Jv_solver = div(C_k . B v)         -- what the global Newton applies,
    Jv_FD     = (R(u + h v) - R(u - h v)) / (2h),

with R(u) = div(sigma(eps(u))) evaluated by REINTEGRATING the material from
the exact same committed snapshot S_{n-1} before each side, so the FD sees
the true constitutive response (sub-stepping, rotations, cache and all).

The direction v is the real one: the strain difference between two
consecutive global-Newton iterations of the recorded run, so B v = eps_{k+1}
- eps_k is known exactly without unpacking the operator.

Diagnosis:
    reference  ||Jv - Jv_FD||/||Jv_FD|| ~ 1e-5  (healthy consistent tangent)
    GPS        ||Jv - Jv_FD||/||Jv_FD|| ~ 1e-1  (the applied matrix is wrong)

Usage:

    .venv/bin/python scripts/diagnose_jv_global_fd.py \
        --output validation/_generated/performance/jv_global_fd.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CROP_20X20 = (1610, 1630, 1075, 1095)
EBSD_ORIENTATION_H5 = "/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5"
PAIRED_PARAMETER_SET = "316l_guilhem2013_nasri2018_meric_srix_rate_1e-3"
PERTURBATIONS = (1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)


class RecordingMaterial:
    """Record every solver call: strain, stress, tangent, dt, substeps."""

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


def _run_backend(
    backend: str,
    arguments: argparse.Namespace,
) -> dict[str, object]:
    from scripts.benchmark_tri2_j2_krylov import _load_case
    from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch
    from fem_inhouse.spectral2d.newton_two_state import (
        EBISpectralSolverConfig,
        solve_two_state_dirichlet_plane_stress,
    )
    from fem_inhouse.spectral2d.transforms import SpectralTransformConfig

    from scripts.qualify_crystal_tet2_p43 import _load_ebsd_orientation_crop

    mesh = arguments.crop_nodes[1] - arguments.crop_nodes[0]
    grid, _, yield_stress, coefficient, boundary = _load_case(mesh, arguments.crop_nodes)
    history = np.stack(
        [fraction * boundary for fraction in np.linspace(0.0, 1.0, arguments.increments + 1)]
    )
    material = create_plane_stress_material_batch(
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
        constitutive_options={
            "paired_parameter_set": arguments.paired_parameter_set,
            "crystal_orientation": {
                "mode": "ebsd",
                "euler_bunge_deg": _load_ebsd_orientation_crop(
                    arguments.ebsd_orientation_h5, arguments.crop_nodes
                )[0],
            },
        },
    )
    recording = RecordingMaterial(material)
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
    calls = recording.calls
    snapshots = recording.committed_snapshots
    if len(snapshots) < 2:
        raise RuntimeError("fewer than two committed increments; cannot replay")
    # The replay increment: the LAST increment with at least two material
    # evaluations, so B v = eps_{k+1} - eps_k is a genuine Newton direction.
    replay_index = None
    replay_calls: list[dict[str, object]] = []
    for committed in range(len(snapshots) - 1, -1, -1):
        increment_calls = [
            call for call in calls if call["committed_before"] == committed
        ]
        if len(increment_calls) >= 2:
            replay_index = committed
            replay_calls = increment_calls
            break
    if replay_index is None:
        raise RuntimeError("no increment with two material evaluations to replay")
    committed_snapshot = snapshots[replay_index - 1]

    from fem_inhouse.spectral2d.newton_two_state import TwoSubcellDiagnostic2D

    kinematics = TwoSubcellDiagnostic2D(grid)
    divergence_buffer = np.empty((*grid.node_shape, 2), dtype=np.float64)

    def divergence_of(stress: np.ndarray) -> np.ndarray:
        kinematics.divergence_from_sample_stress_into(
            stress, divergence_buffer
        )
        return divergence_buffer.copy()

    eps_k = np.asarray(replay_calls[0]["strain"], dtype=float)
    eps_k1 = np.asarray(replay_calls[1]["strain"], dtype=float)
    tangent_k = np.asarray(replay_calls[0]["tangent"], dtype=float)
    dt = float(replay_calls[0]["time_increment"])
    delta_eps = eps_k1 - eps_k
    delta_norm = float(np.linalg.norm(delta_eps))
    direction = delta_eps / max(delta_norm, 1.0e-30)
    direction = direction.reshape(*grid.pixel_shape, 2, 3)

    # Jv_solver: what the global Newton applies, div(C_k . B v), with the
    # tangent of the first iteration and the real direction of the second.
    delta_stress = np.einsum(
        "xyqij,xyqj->xyqi", tangent_k.reshape(*grid.pixel_shape, 2, 3, 3), direction
    )
    jv_solver = divergence_of(delta_stress)

    # Jv_FD: re-integrate the material from the exact same committed snapshot
    # before each side. The solver's own evaluate returns in-plane stress per
    # material point; reshape to the two-subcell field before the divergence.
    def residual_at(strain: np.ndarray) -> np.ndarray:
        trial = material.evaluate(
            strain.reshape(-1, 3),
            time_increment=dt,
            consistent_tangent=True,
        )
        stress = np.asarray(trial.stress_in_plane_mpa).reshape(*grid.pixel_shape, 2, 3)
        return divergence_of(stress)

    per_h: dict[str, float] = {}
    for h in PERTURBATIONS:
        material.restore_state(committed_snapshot)
        r_plus = residual_at(eps_k.reshape(*grid.pixel_shape, 2, 3) + h * direction)
        material.restore_state(committed_snapshot)
        r_minus = residual_at(eps_k.reshape(*grid.pixel_shape, 2, 3) - h * direction)
        material.restore_state(committed_snapshot)
        jv_fd = (r_plus - r_minus) / (2.0 * h)
        norm_fd = float(np.linalg.norm(jv_fd))
        per_h[f"{h:.0e}"] = float(np.linalg.norm(jv_solver - jv_fd)) / max(
            norm_fd, 1.0e-30
        )
    material.revert()
    total_newton = int(sum(result.diagnostics.iterations_per_increment))
    return {
        "backend": backend,
        "newton_iterations": total_newton,
        "replay_increment": replay_index + 1,
        "replay_calls": len(replay_calls),
        "delta_strain_norm": float(delta_norm),
        "time_increment": dt,
        "relative_errors_per_h": per_h,
        "jv_solver_norm": float(np.linalg.norm(jv_solver)),
        "substeps_first_call": int(replay_calls[0]["substeps"]),
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/_generated/performance/jv_global_fd.json"),
    )
    arguments = parser.parse_args()
    backends = (
        "mfront-native-generalised-plane-stress",
        "mfront-3d-condensed-plane-stress",
    )
    records = [_run_backend(backend, arguments) for backend in backends]
    payload = {
        "schema_version": 1,
        "configuration": {
            "crop_nodes": arguments.crop_nodes,
            "increments": arguments.increments,
            "mfront_threads": arguments.mfront_threads,
            "perturbations": list(PERTURBATIONS),
        },
        "backends": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for record in records:
        print(
            f"{record['backend']}: {record['newton_iterations']} Newton | "
            f"replay inc {record['replay_increment']} ({record['replay_calls']} calls) | "
            f"||Jv|| {record['jv_solver_norm']:.3e} | "
            + " | ".join(
                f"h={h}: {err:.2e}" for h, err in record["relative_errors_per_h"].items()
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
