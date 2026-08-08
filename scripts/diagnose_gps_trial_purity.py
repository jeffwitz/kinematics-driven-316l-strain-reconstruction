"""Purity of the GPS trial: is evaluate strictly a function of (s0, eps_trial)?

The 85-vs-57 Newton penalty survives every local-level hypothesis: the
tangent is identical to 1e-16 at matched states, the sub-stepping path is
acquitted by the corrected halved-reference test, the closure tolerance and
the GPS epsilon are eliminated. What remains is the trial/state handling
(Case C of the trajectory-divergence framework): a cache, an s1, a predictor
or a sub-stepping state leaking from one global evaluation to the next.

The solver runs many successive evaluations during Newton/Krylov/line-search,
while the usual constitutive tests do evaluate -> commit. This script
performs the decisive purity protocol on the GPS backend AND on the condensed
reference as a control:

Test 1 (no commit between calls), from the exact same committed state S_n:

    evaluate(A) -> A1
    evaluate(B) -> B
    evaluate(A) -> A2

    impose A1 == A2 on stress, tangent, g[12], p[12], a[12], transverse
    strains, the sub-stepping decision and the number of divisions.

Test 1b (snapshot variant):

    snapshot
    evaluate(A)
    evaluate(B)
    restore(snapshot)
    evaluate(A) -> A2'

    impose A1 == A2'.

Test 2 (accept_global_trial without commit):

    s0 -> evaluate(A) -> accept_global_trial() -> evaluate(B)
    against
    s0 -> evaluate(B)

    impose the response to B identical (stress, internal variables, tangent):
    accept_global_trial may improve a predictor but must never transform the
    committed physical history.

A and B are taken from the solver's OWN calls of the last increment of a real
run (A = last accepted strain, B = the first trial of the same increment), so
the test exercises exactly the evaluation pattern the global Newton makes.

Usage:

    .venv/bin/python scripts/diagnose_gps_trial_purity.py \
        --output validation/_generated/performance/trial_purity.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

CROP_20X20 = (1610, 1630, 1075, 1095)
EBSD_ORIENTATION_H5 = "/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5"
PAIRED_PARAMETER_SET = "316l_guilhem2013_nasri2018_meric_srix_rate_1e-3"

#: Any deviation above this on the compared arrays fails the purity test.
#: The user's protocol: "if A2 != A1 even at 1e-10, we have probably found it."
PURITY_TOLERANCE = 1.0e-10


class RecordingMaterial:
    """Wrap a plane-stress batch, record every solver call.

    Records per call the in-plane strain and the sub-stepping counters so the
    test can later replay A = the last accepted strain and B = an earlier
    trial of the same increment, and can verify the sub-stepping decision
    (did this call halve, and how far?) is itself part of the compared state.
    """

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.calls: list[dict[str, object]] = []
        self.committed_snapshots: list[object] = []
        self._last_substeps = 0
        self._last_divisions = 0
        self._last_cache_hits = 0
        self._last_cache_misses = 0

    def _record(self, strain: np.ndarray) -> None:
        substeps = int(getattr(self._inner, "_substep_uses", 0))
        divisions = int(getattr(self._inner, "_substep_divisions_max", 0))
        cache_hits = int(getattr(self._inner, "_cache_hits", 0))
        cache_misses = int(getattr(self._inner, "_cache_misses", 0))
        self.calls.append(
            {
                "strain": np.asarray(strain, dtype=float).copy(),
                "substeps": substeps - self._last_substeps,
                "divisions": divisions - self._last_divisions,
                "cache_hits": cache_hits - self._last_cache_hits,
                "cache_misses": cache_misses - self._last_cache_misses,
                "committed_before": len(self.committed_snapshots),
            }
        )
        self._last_substeps = substeps
        self._last_divisions = divisions
        self._last_cache_hits = cache_hits
        self._last_cache_misses = cache_misses

    def _record_committed(self) -> None:
        self.committed_snapshots.append(self._inner.snapshot_state())

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
        self._record(strain)
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
        self._record(strain)
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
        self._record(strain)
        return trial

    def commit(self) -> None:
        self._inner.commit()
        self._record_committed()

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


def _max_abs(a: object, b: object) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _snapshot_trial(trial: object) -> dict[str, np.ndarray]:
    """Deep-copy every compared quantity of a trial."""

    out: dict[str, np.ndarray] = {
        "stress": np.asarray(trial.stress_in_plane_mpa, dtype=float).copy(),
    }
    tangent = getattr(trial, "tangent_in_plane_mpa", None)
    if tangent is not None:
        out["tangent"] = np.asarray(tangent, dtype=float).copy()
    residual = getattr(trial, "plane_stress_residual_mpa", None)
    if residual is not None:
        out["residual"] = np.asarray(residual, dtype=float).copy()
    full_stress = getattr(trial, "full_stress_tensor_mpa", None)
    if full_stress is not None:
        out["full_stress"] = np.asarray(full_stress, dtype=float).copy()
    full_strain = getattr(trial, "full_strain_tensor", None)
    if full_strain is not None:
        out["full_strain"] = np.asarray(full_strain, dtype=float).copy()
        # The transverse strains: components (zz, xz, yz) of the total tensor.
        out["transverse"] = np.stack(
            (full_strain[..., 2, 2], full_strain[..., 0, 2], full_strain[..., 1, 2]),
            axis=-1,
        ).copy()
    plastic = getattr(trial, "plastic_strain_tensor", None)
    if plastic is not None:
        out["plastic_strain"] = np.asarray(plastic, dtype=float).copy()
    observables = getattr(trial, "observables", None) or {}
    for name in ("plastic_slip", "equivalent_plastic_slip", "back_strain"):
        if name in observables:
            out[name] = np.asarray(observables[name], dtype=float).copy()
    if "accumulated_slip" in observables:
        out["accumulated_slip"] = np.asarray(observables["accumulated_slip"], dtype=float).copy()
    return out


def _compare(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: _max_abs(value, right[name]) for name, value in left.items()}


def _purity_protocol(
    material: object,
    strain_a: np.ndarray,
    strain_b: np.ndarray,
    time_increment: float,
) -> dict[str, object]:
    """Test 1 and 1b: evaluate(A)-evaluate(B)-evaluate(A), with and without restore."""

    # Test 1: three evaluations, no commit between them.
    first = _snapshot_trial(
        material.evaluate(strain_a, time_increment=time_increment)
    )
    material.evaluate(strain_b, time_increment=time_increment)
    again = _snapshot_trial(
        material.evaluate(strain_a, time_increment=time_increment)
    )
    test1 = _compare(first, again)
    # A1 == A2 with a tolerance of 0 on the sub-stepping decision is the test;
    # the decision counters must be part of the comparison, and they are not
    # inside the trial, so they are compared via a fresh evaluation pair below.

    # Test 1b: snapshot, A, B, restore, A. The trial left by test 1 must be
    # discarded first: the condensed backend refuses to snapshot while a
    # trial is active.
    material.revert()
    snapshot = material.snapshot_state()
    material.evaluate(strain_a, time_increment=time_increment)
    material.evaluate(strain_b, time_increment=time_increment)
    material.restore_state(snapshot)
    restored = _snapshot_trial(
        material.evaluate(strain_a, time_increment=time_increment)
    )
    test1b = _compare(first, restored)
    return {"test1": test1, "test1b": test1b}


def _substep_decision_protocol(
    material: object,
    strain_a: np.ndarray,
    strain_b: np.ndarray,
    time_increment: float,
) -> dict[str, object]:
    """Is the sub-stepping decision itself stable across re-evaluations?

    The counters are read around each call: the decision (did it sub-step?),
    the number of divisions and the cache outcome must be identical for the
    same (s0, eps) regardless of what happened in between.
    """

    def counters() -> dict[str, int]:
        return {
            "substeps": int(getattr(material, "_substep_uses", 0)),
            "divisions": int(getattr(material, "_substep_divisions_max", 0)),
            "cache_hits": int(getattr(material, "_cache_hits", 0)),
            "cache_misses": int(getattr(material, "_cache_misses", 0)),
        }

    before = counters()
    material.evaluate(strain_a, time_increment=time_increment)
    after_a1 = counters()
    material.evaluate(strain_b, time_increment=time_increment)
    material.evaluate(strain_a, time_increment=time_increment)
    after_a2 = counters()
    decision_a1 = {k: after_a1[k] - before[k] for k in before}
    decision_a2 = {k: after_a2[k] - after_a1[k] for k in before}
    return {
        "a1": decision_a1,
        "a2": decision_a2,
        "identical": decision_a1 == decision_a2,
    }


def _accept_global_trial_protocol(
    material: object,
    strain_a: np.ndarray,
    strain_b: np.ndarray,
    time_increment: float,
) -> dict[str, object]:
    """Test 2: does accept_global_trial() transform the response to B?"""

    snapshot = material.snapshot_state()
    baseline = _snapshot_trial(material.evaluate(strain_b, time_increment=time_increment))
    material.restore_state(snapshot)
    material.evaluate(strain_a, time_increment=time_increment)
    material.accept_global_trial()
    with_accept = _snapshot_trial(
        material.evaluate(strain_b, time_increment=time_increment)
    )
    material.restore_state(snapshot)
    return _compare(baseline, with_accept)


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
    total_newton = int(sum(result.diagnostics.iterations_per_increment))
    # A and B must both be FORWARD evaluations from the same committed state:
    # replaying an earlier trial from the final committed state asks the law
    # to go backward, and the rate-independent law refuses or diverges (the
    # known zero/negative-increment trap). The faithful protocol replays the
    # solver's OWN last increment: S_{n-1} is the committed state before it,
    # A is the converged target of that increment, B an earlier trial of it.
    if len(snapshots) < 2:
        raise RuntimeError("the run committed fewer than two increments; cannot replay")
    last_committed = len(snapshots) - 1
    last_calls = [call for call in calls if call["committed_before"] == last_committed]
    if not last_calls:
        raise RuntimeError("no solver calls recorded for the last increment")
    strain_a = np.asarray(last_calls[-1]["strain"], dtype=float)
    if len(last_calls) > 1:
        strain_b = np.asarray(last_calls[0]["strain"], dtype=float)
    else:
        strain_b = strain_a + 1.0e-4 * np.abs(strain_a).max()
    if np.array_equal(strain_a, strain_b):
        strain_b = strain_a + 1.0e-4 * np.abs(strain_a).max()
    time_increment = float(1.0 / arguments.increments)
    material.restore_state(snapshots[last_committed - 1])
    purity = _purity_protocol(material, strain_a, strain_b, time_increment)
    material.restore_state(snapshots[last_committed - 1])
    substep = _substep_decision_protocol(material, strain_a, strain_b, time_increment)
    material.restore_state(snapshots[last_committed - 1])
    accept = _accept_global_trial_protocol(material, strain_a, strain_b, time_increment)
    return {
        "backend": backend,
        "newton_iterations": total_newton,
        "solver_calls": len(calls),
        "replayed_increment": last_committed,
        "strain_a_norm": float(np.linalg.norm(strain_a)),
        "strain_b_norm": float(np.linalg.norm(strain_b)),
        "strain_ab_diff": float(np.linalg.norm(strain_a - strain_b)),
        "purity_test1": purity["test1"],
        "purity_test1b": purity["test1b"],
        "substep_decision": substep,
        "accept_global_trial": accept,
        "purity_pass": all(
            value <= PURITY_TOLERANCE for value in purity["test1"].values()
        )
        and all(value <= PURITY_TOLERANCE for value in purity["test1b"].values())
        and bool(substep["identical"]),
        "accept_pass": all(
            value <= PURITY_TOLERANCE for value in accept.values()
        ),
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
        default=Path("validation/_generated/performance/trial_purity.json"),
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
            "purity_tolerance": PURITY_TOLERANCE,
        },
        "backends": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for record in records:
        verdict = "PURE" if record["purity_pass"] else "IMPURE"
        accept_verdict = "NEUTRAL" if record["accept_pass"] else "TRANSFORMATIVE"
        worst1 = max(record["purity_test1"].values(), default=0.0)
        worst1b = max(record["purity_test1b"].values(), default=0.0)
        worst2 = max(record["accept_global_trial"].values(), default=0.0)
        print(
            f"{record['backend']}: {record['newton_iterations']} Newton, "
            f"{record['solver_calls']} calls | purity {verdict} "
            f"(test1 {worst1:.2e}, test1b {worst1b:.2e}) | "
            f"substep decision {'identical' if record['substep_decision']['identical'] else 'DIFFERENT'} "
            f"| accept_global_trial {accept_verdict} ({worst2:.2e})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
