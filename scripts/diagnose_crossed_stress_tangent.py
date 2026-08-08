"""Cross the GPS and reference stress/tangent and count Newton iterations.

Every local-level hypothesis for the 85-vs-57 penalty is eliminated: the GPS
trial is PURE (bit-identical re-evaluations), accept_global_trial is neutral,
the assembled global Jv matches the finite difference to 1e-9, and a tight
linear forcing costs only 2 iterations at M20. What remains is the choice of
the pair (stress, tangent) the solver sees:

| stress/state | tangent    | interpretation                        |
| ------------ | ---------- | ------------------------------------- |
| GPS          | GPS        | 85 (the reference run)                |
| GPS          | REF        | is the derivative innocent?           |
| REF          | GPS        | is the residual innocent?             |
| REF          | REF        | 57 (the baseline)                     |

The two backends are evaluated at the SAME imposed strain from their own
committed states -- which the trajectories make genuinely different at the
deep increments -- and the returned trial combines the stress of one with the
tangent of the other. If GPS-stress + REF-tangent still needs 85 Newton, the
derivative is acquitted and the problem lives in the GPS residual/trial
function; if REF-stress + GPS-tangent still needs 57, the same conclusion
falls on the residual side.

Usage:

    .venv/bin/python scripts/diagnose_crossed_stress_tangent.py \
        --output validation/_generated/performance/crossed_stress_tangent.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CROP_20X20 = (1610, 1630, 1075, 1095)
EBSD_ORIENTATION_H5 = "/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5"
PAIRED_PARAMETER_SET = "316l_guilhem2013_nasri2018_meric_srix_rate_1e-3"

GPS = "mfront-native-generalised-plane-stress"
REFERENCE = "mfront-3d-condensed-plane-stress"


class CrossedBackend:
    """Evaluate both backends at the same strain; combine stress and tangent.

    The two backends are committed independently, so each integrates from its
    own history: this is exactly the situation at the deep increments, where
    the GPS and reference trajectories genuinely differ. `complete_trial`
    delegates to the backend that supplied the stress, so the recorded final
    fields stay coherent with the stress the solver saw.
    """

    def __init__(self, gps: object, reference: object, *, stress_from: str) -> None:
        if stress_from not in {"gps", "reference"}:
            raise ValueError("stress_from must be 'gps' or 'reference'")
        self._gps = gps
        self._reference = reference
        self._stress_from = stress_from

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

    def evaluate(
        self,
        in_plane_strain: np.ndarray,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> object:
        gps_trial, ref_trial = self._pair(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )
        return self._cross(gps_trial, ref_trial, consistent_tangent)

    def evaluate_in_plane(
        self,
        in_plane_strain: np.ndarray,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> object:
        gps_trial, ref_trial = self._pair(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )
        return self._cross(gps_trial, ref_trial, consistent_tangent)

    def evaluate_in_plane_response(
        self,
        in_plane_strain: np.ndarray,
        *,
        time_increment: float,
        response_level: str = "tangent",
        consistent_tangent: bool = True,
    ) -> object:
        gps_trial, ref_trial = self._pair(
            in_plane_strain,
            time_increment=time_increment,
            consistent_tangent=consistent_tangent,
        )
        return self._cross(gps_trial, ref_trial, consistent_tangent)

    def _cross(
        self,
        gps_trial: object,
        ref_trial: object,
        consistent_tangent: bool,
    ) -> object:
        from fem_inhouse.core.plane_stress_material import InPlaneConstitutiveTrial

        if self._stress_from == "gps":
            stress_trial, tangent_trial = gps_trial, ref_trial
        else:
            stress_trial, tangent_trial = ref_trial, gps_trial
        return InPlaneConstitutiveTrial(
            stress_in_plane_mpa=stress_trial.stress_in_plane_mpa,
            tangent_in_plane_mpa=(
                tangent_trial.tangent_in_plane_mpa
                if consistent_tangent
                else None
            ),
            observables=stress_trial.observables,
            local_plane_stress_iterations=stress_trial.local_plane_stress_iterations,
        )

    def complete_trial(self, trial: object) -> object:
        if self._stress_from == "gps":
            return self._gps.complete_trial(trial)
        return self._reference.complete_trial(trial)

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
        return f"crossed-{self._stress_from}-stress"

    @property
    def completion_strategy(self) -> str:
        return "crossed_stress_tangent"

    @property
    def linear_system_matrix_type(self) -> str:
        return self._gps.linear_system_matrix_type

    @property
    def statistics(self) -> object:
        return self._gps.statistics

    @property
    def timing_statistics(self) -> object:
        gps_timing = self._gps.timing_statistics
        ref_timing = self._reference.timing_statistics
        if gps_timing is None:
            return ref_timing
        if ref_timing is None:
            return gps_timing
        from dataclasses import fields

        from fem_inhouse.core.mfront import MFrontTimingStatistics

        combined = MFrontTimingStatistics()
        for field in fields(MFrontTimingStatistics):
            value = getattr(gps_timing, field.name, None)
            other = getattr(ref_timing, field.name, None)
            if isinstance(value, (int, float)) and isinstance(other, (int, float)):
                setattr(combined, field.name, value + other)
        return combined


def _run_crossed(
    stress_from: str,
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

    def build(backend: str) -> object:
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

    material = CrossedBackend(
        build(GPS),
        build(REFERENCE),
        stress_from=stress_from,
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
        "stress_from": stress_from,
        "newton_iterations": int(sum(result.diagnostics.iterations_per_increment)),
        "accepted_increments": len(result.diagnostics.iterations_per_increment),
        "iterations_per_increment": list(result.diagnostics.iterations_per_increment),
        "solver_succeeded": bool(np.isfinite(result.displacement).all()),
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
        default=Path("validation/_generated/performance/crossed_stress_tangent.json"),
    )
    arguments = parser.parse_args()
    records = [
        _run_crossed("gps", arguments),
        _run_crossed("reference", arguments),
    ]
    payload = {
        "schema_version": 1,
        "configuration": {
            "crop_nodes": arguments.crop_nodes,
            "increments": arguments.increments,
            "mfront_threads": arguments.mfront_threads,
        },
        "backends": records,
        "reference_points": {
            "gps_gps": None,
            "gps_ref": None,
            "ref_gps": None,
            "ref_ref": None,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for record in records:
        print(
            f"{record['stress_from']}-stress + other-tangent: "
            f"{record['newton_iterations']} Newton across "
            f"{record['accepted_increments']} increments"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
