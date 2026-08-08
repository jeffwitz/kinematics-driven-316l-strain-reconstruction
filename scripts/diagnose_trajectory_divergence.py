"""Find the first instant where the GPS and reference trajectories diverge.

The sub-stepping path and the tangent are acquitted for the 85-vs-57 Newton
penalty; the remaining suspects are the trial/state handling. This script
runs both backends on the same case through a recording wrapper, aligning the
two global-Newton evaluate sequences, and reports:

- k*, the first evaluate index where the recorded states differ;
- what differs first: the imposed strain (a solver-path divergence), the
  returned stress (an evaluation difference), or the internal variables
  g/p/a and the closure strains (an integration difference);
- the sub-stepping flag at the divergence.

Diagnosis framework:

- Case A: stresses differ while the committed internal states are still
  identical -> trial/transaction handling;
- Case B: g, p, a differ first -> different integration paths;
- Case C: everything identical before a commit, diverging after it ->
  commit/accept_global_trial/trial promotion.

Usage:

    .venv/bin/python scripts/diagnose_trajectory_divergence.py \
        --output validation/_generated/performance/trajectory_divergence.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

CROP = (1610, 1630, 1075, 1095)
EBSD_ORIENTATION_H5 = "/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5"
PAIRED_PARAMETER_SET = "316l_guilhem2013_nasri2018_meric_srix_rate_1e-3"


class RecordingMaterial:
    """Wrap a plane-stress batch and record every evaluate.

    Records per call: the in-plane strain (Kelvin), the returned in-plane
    stress, the internal state variables (elastic strain, slips, hardening,
    closure strains) read from the batch's manager, and the sub-stepping
    counter of the GPS bridge when present.
    """

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.records: list[dict[str, object]] = []
        self._last_substeps = 0

    def _record(self, strain: np.ndarray, trial: object) -> None:
        record: dict[str, object] = {
            "strain": np.asarray(strain, dtype=float).copy(),
            "stress": np.asarray(trial.stress_in_plane_mpa, dtype=float).copy(),
        }
        residual = getattr(trial, "plane_stress_residual_mpa", None)
        if residual is not None:
            record["residual"] = np.asarray(residual, dtype=float).copy()
        observables = getattr(trial, "observables", None) or {}
        if "plastic_slip" in observables:
            record["slip"] = np.asarray(observables["plastic_slip"], dtype=float).copy()
        if "equivalent_plastic_slip" in observables:
            record["peeq_slip"] = np.asarray(
                observables["equivalent_plastic_slip"], dtype=float
            ).copy()
        manager = self._manager()
        if manager is not None:
            record["isv"] = np.asarray(manager.s1.internal_state_variables).copy()
        substeps = int(getattr(self._inner, "_substep_uses", 0))
        record["substeps"] = substeps - self._last_substeps
        self._last_substeps = substeps
        self.records.append(record)

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
        self._record(strain, trial)
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
        self._record(strain, trial)
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
        self._record(strain, trial)
        return trial

    def _manager(self) -> object | None:
        inner = self._inner
        manager = getattr(inner, "_manager", None)
        if manager is None:
            bridge = getattr(inner, "_bridge", None)
            manager = getattr(bridge, "_manager", None)
        return manager

    def commit(self) -> None:
        self._inner.commit()

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


def _run(
    backend: str, arguments: argparse.Namespace
) -> tuple[RecordingMaterial, dict[str, object]]:
    from scripts.benchmark_tri2_j2_krylov import _load_case
    from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch
    from fem_inhouse.spectral2d.newton_two_state import (
        EBISpectralSolverConfig,
        solve_two_state_dirichlet_plane_stress,
    )
    from fem_inhouse.spectral2d.transforms import SpectralTransformConfig

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
                "euler_bunge_deg": _load_ebsd(arguments.ebsd_orientation_h5, arguments.crop_nodes),
            },
        },
    )
    recording = RecordingMaterial(material)
    config = EBISpectralSolverConfig(
        relative_equilibrium_tolerance=1e-8,
        maximum_newton_iterations=arguments.maximum_newton_iterations,
        krylov_method="lgmres",
        krylov_recycling=True,
        gmres_relative_tolerance=1e-8,
        gmres_restart=50,
        lgmres_inner_m=30,
        lgmres_outer_k=3,
        linear_tolerance_mode="eisenstat_walker",
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
        boundary_displacement_history=history,
        config=config,
    )
    return recording, {"newton_iterations": int(sum(result.diagnostics.iterations_per_increment))}


def _load_ebsd(path: Path, crop: tuple[int, int, int, int]) -> np.ndarray:
    from scripts.qualify_crystal_tet2_p43 import _load_ebsd_orientation_crop

    angles, _ = _load_ebsd_orientation_crop(path, crop)
    return angles


def _shared_isv(isv: np.ndarray) -> np.ndarray:
    """The layout differs between the backends: the GPS inserts three
    closure components between the slips and the hardening. Compare the
    shared variables (eel, g, p, a) by dropping the closure block."""

    if isv.shape[-1] >= 45:  # GPS: eel(6) g(12) closure(3) p(12) a(12) [+ localIterations]
        return np.concatenate((isv[..., :18], isv[..., 21:45]), axis=-1)
    return isv


def _first_divergence(
    reference: RecordingMaterial, candidate: RecordingMaterial
) -> dict[str, object] | None:
    common = min(len(reference.records), len(candidate.records))
    for index in range(common):
        ref = reference.records[index]
        cand = candidate.records[index]
        strain_same = np.array_equal(ref["strain"], cand["strain"])
        stress_same = np.allclose(ref["stress"], cand["stress"], rtol=0.0, atol=1e-10)
        isv_same = True
        slip_same = True
        if "isv" in ref and "isv" in cand:
            isv_same = np.array_equal(
                _shared_isv(np.asarray(ref["isv"])), _shared_isv(np.asarray(cand["isv"]))
            )
        if "slip" in ref and "slip" in cand:
            slip_same = np.allclose(ref["slip"], cand["slip"], rtol=0.0, atol=1e-12)
        if not (strain_same and stress_same and isv_same and slip_same):
            detail: dict[str, object] = {
                "index": index,
                "strain_differs": not strain_same,
                "stress_differs": not stress_same,
                "isv_differs": not isv_same,
                "slip_differs": not slip_same,
            }
            stress_diff = np.abs(np.asarray(ref["stress"]) - np.asarray(cand["stress"]))
            detail["stress_max_abs_diff_mpa"] = float(stress_diff.max())
            detail["stress_mean_abs_diff_mpa"] = float(stress_diff.mean())
            detail["stress_reference_max_mpa"] = float(np.max(np.abs(np.asarray(ref["stress"]))))
            if "isv" in ref and "isv" in cand:
                shared_diff = np.abs(_shared_isv(np.asarray(ref["isv"])) - _shared_isv(np.asarray(cand["isv"])))
                detail["isv_max_abs_diff"] = float(shared_diff.max())
                detail["isv_mean_abs_diff"] = float(shared_diff.mean())
            detail["reference_substeps"] = int(ref.get("substeps", 0))
            detail["candidate_substeps"] = int(cand.get("substeps", 0))
            return detail
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=CROP)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--ebsd-orientation-h5", type=Path, default=Path(EBSD_ORIENTATION_H5))
    parser.add_argument("--paired-parameter-set", default=PAIRED_PARAMETER_SET)
    parser.add_argument("--library", default=os.environ.get("MFRONT_BEHAVIOUR_LIBRARY"))
    parser.add_argument("--mfront-threads", type=int, default=4)
    parser.add_argument("--maximum-newton-iterations", type=int, default=40)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/_generated/performance/trajectory_divergence.json"),
    )
    arguments = parser.parse_args()
    if not arguments.library:
        parser.error("--library is required")

    reference, reference_summary = _run(
        "mfront-3d-condensed-plane-stress", arguments
    )
    candidate, candidate_summary = _run(
        "mfront-native-generalised-plane-stress", arguments
    )
    divergence = _first_divergence(reference, candidate)
    report = {
        "configuration": {
            "crop_nodes": arguments.crop_nodes,
            "increments": arguments.increments,
        },
        "reference_evaluations": len(reference.records),
        "candidate_evaluations": len(candidate.records),
        "reference_newton_iterations": reference_summary["newton_iterations"],
        "candidate_newton_iterations": candidate_summary["newton_iterations"],
        "first_divergence": divergence,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value)
        + "\n"
    )
    print(f"reference: {reference_summary['newton_iterations']} Newton, "
          f"{len(reference.records)} evaluations")
    print(f"candidate: {candidate_summary['newton_iterations']} Newton, "
          f"{len(candidate.records)} evaluations")
    if divergence is None:
        print("no divergence in the common prefix")
    else:
        print(f"first divergence at evaluate {divergence['index']}: "
              f"strain={divergence['strain_differs']}, stress={divergence['stress_differs']}, "
              f"isv={divergence['isv_differs']}, slip={divergence['slip_differs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
