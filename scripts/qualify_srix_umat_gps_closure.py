"""Qualify the UMAT-closure generalised plane stress backend.

Executes cases C1, C2, C3 of
validation/srix_umat_gps_closure_preregistration.md, before which this script
must not be run. The reference backend is
``mfront-3d-condensed-plane-stress``; the candidate is
``mfront-native-generalised-plane-stress`` (Fcc316LForestRubinSrixGps).

Acceptance: A1 (in-plane stress and transverse strains, relative L2),
A2 (condensed tangent, relative L2), A3 (closure residual) and A6
(finite-difference tangent at every increment) -- the reference agreement
was restored as a primary criterion on 2026-08-08 after the "two branches"
reading was shown to be an artefact of a strain bookkeeping bug (6bfaf86).

Usage:

    bash scripts/build_mfront_behaviour.sh
    .venv/bin/python scripts/qualify_srix_umat_gps_closure.py \
        --output validation/_generated/performance/srix_umat_gps_closure.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Frozen load history (preregistration C1..C3): proportional uniaxial tension
# in the GLOBAL frame, 12 increments to two percent, covering the elastic
# range, the first activation and established multi-system plasticity. The
# exact numbers below are the registered history; do not change them without
# amending the preregistration.
MAX_STRAIN = 0.02
INCREMENTS = 12
HISTORY = np.array(
    [(i / INCREMENTS) * MAX_STRAIN * np.array([1.0, -0.4, 0.0]) for i in range(1, INCREMENTS + 1)]
)

#: Cases: (name, Bunge angles in degrees). None means the identity.
CASES: dict[str, tuple[float, float, float] | None] = {
    "C1_identity": None,
    "C2_bunge_35_20_15": (35.0, 20.0, 15.0),
    "C3_bunge_54_45_10": (54.7, 45.0, 10.0),
}

#: Acceptance criteria of the preregistration.
A1_TOLERANCE = 1.0e-9  # in-plane stress and transverse strains, relative L2
A2_TOLERANCE = 1.0e-8  # condensed tangent, relative L2
A3_TOLERANCE_MPA = 1.0e-6  # global transverse stress residual, max absolute
A6_TOLERANCE = 1.0e-6  # finite-difference tangent cross-check, relative

#: Finite-difference perturbation for the tangent cross-check (relative).
FD_PERTURBATION = 1.0e-6


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(candidate) - np.asarray(reference))
        / max(np.linalg.norm(np.asarray(reference)), 1.0e-30)
    )


def _transverse_strain_tensor(full_strain_tensor: np.ndarray) -> np.ndarray:
    """Return the global transverse strain triple (zz, xz, yz)."""

    return np.stack(
        (
            full_strain_tensor[:, 2, 2],
            full_strain_tensor[:, 0, 2],
            full_strain_tensor[:, 1, 2],
        ),
        axis=-1,
    )


def _make_batch(
    backend: str,
    *,
    library: str,
    parameter_set: str,
    orientation: tuple[float, float, float] | None,
    parameters: dict[str, float] | None = None,
) -> object:
    from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch

    options: dict[str, object] = {"parameter_set": parameter_set}
    if orientation is not None:
        from fem_inhouse.core.crystal_orientation import rotation_from_euler_bunge_deg

        options["crystal_orientation"] = {
            "mode": "homogeneous",
            "matrix": np.asarray(
                rotation_from_euler_bunge_deg(*orientation), dtype=float
            ).tolist(),
        }
    if parameters is not None:
        options["parameters"] = parameters
    return create_plane_stress_material_batch(
        backend,
        np.full((1, 1), 250.0),
        np.full((1, 1), 500.0),
        0.245,
        young_modulus_mpa=205000.0,
        poisson_ratio=0.3,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1000,
        first_positive_plastic_strain=1e-6,
        mfront_library=library,
        mfront_threads=1,
        mfront_behaviour_id="fcc_forest_rubin_srix",
        constitutive_options=options,
    )


def _run_history(
    batch: object,
    history: np.ndarray,
) -> dict[str, list[float]]:
    """Drive one batch through the history, committing every increment."""

    records: dict[str, list[float]] = {
        "in_plane_stress_relative_l2": [],
        "in_plane_stress_max_abs_mpa": [],
        "transverse_strain_relative_l2": [],
        "transverse_strain_max_abs": [],
        "tangent_relative_l2": [],
        "tangent_max_abs": [],
        "global_transverse_residual_max_mpa": [],
    }
    for in_plane in history:
        trial = batch.evaluate(
            np.tile(in_plane, (1, 1)), time_increment=1.0 / INCREMENTS
        )
        records["in_plane_stress_max_abs_mpa"].append(
            float(np.max(np.abs(trial.stress_in_plane_mpa)))
        )
        records["transverse_strain_max_abs"].append(
            float(np.max(np.abs(_transverse_strain_tensor(trial.full_strain_tensor))))
        )
        records["tangent_max_abs"].append(
            float(np.max(np.abs(np.asarray(trial.tangent_in_plane_mpa))))
        )
        records["global_transverse_residual_max_mpa"].append(
            float(np.max(np.abs(np.asarray(trial.plane_stress_residual_mpa))))
        )
        batch.commit()
    return records


def _compare(
    candidate_batch: object,
    reference_batch: object,
    history: np.ndarray,
    *,
    tangent: bool,
) -> dict[str, float]:
    """Run both batches through the history and return the registered metrics."""

    in_plane_stress_errors: list[float] = []
    transverse_strain_errors: list[float] = []
    tangent_errors: list[float] = []
    for in_plane in history:
        strain = np.tile(in_plane, (1, 1))
        candidate = candidate_batch.evaluate(strain, time_increment=1.0 / INCREMENTS)
        reference = reference_batch.evaluate(strain, time_increment=1.0 / INCREMENTS)
        in_plane_stress_errors.append(
            _relative_error(candidate.stress_in_plane_mpa, reference.stress_in_plane_mpa)
        )
        transverse_strain_errors.append(
            _relative_error(
                _transverse_strain_tensor(candidate.full_strain_tensor),
                _transverse_strain_tensor(reference.full_strain_tensor),
            )
        )
        if tangent:
            tangent_errors.append(
                _relative_error(
                    candidate.tangent_in_plane_mpa, reference.tangent_in_plane_mpa
                )
            )
        candidate_batch.commit()
        reference_batch.commit()
    return {
        "in_plane_stress_relative_l2_max": max(in_plane_stress_errors),
        "transverse_strain_relative_l2_max": max(transverse_strain_errors),
        "tangent_relative_l2_max": max(tangent_errors) if tangent else float("nan"),
    }


def _finite_difference_tangent_check(
    batch: object,
    in_plane: np.ndarray,
    *,
    time_increment: float,
) -> float:
    """Compare the returned condensed tangent with central differences.

    Returns the maximum relative error across the six probes. The batch is
    left in its committed state.
    """

    strain = np.tile(in_plane, (1, 1))
    base = batch.evaluate(strain, time_increment=time_increment)
    tangent_returned = np.asarray(base.tangent_in_plane_mpa)[0]
    base_stress = np.asarray(base.stress_in_plane_mpa)[0]
    fd_tangent = np.zeros((3, 3), dtype=float)
    for column in range(3):
        plus = strain.copy()
        minus = strain.copy()
        plus[:, column] += FD_PERTURBATION
        minus[:, column] -= FD_PERTURBATION
        stress_plus = np.asarray(
            batch.evaluate(plus, time_increment=time_increment).stress_in_plane_mpa
        )[0]
        stress_minus = np.asarray(
            batch.evaluate(minus, time_increment=time_increment).stress_in_plane_mpa
        )[0]
        fd_tangent[:, column] = (stress_plus - stress_minus) / (2 * FD_PERTURBATION)
    batch.revert()
    # The FD perturbation acts on the in-plane strain in engineering storage
    # and the tangent is engineering too; the returned tangent is used as-is.
    scale = max(np.max(np.abs(fd_tangent)), 1.0e-30)
    _ = base_stress
    return float(np.max(np.abs(tangent_returned - fd_tangent)) / scale)


def _main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--library",
        default=os.environ.get("MFRONT_BEHAVIOUR_LIBRARY"),
        help="compiled MFront behaviour library",
    )
    parser.add_argument(
        "--parameter-set",
        default="316l_srix_transposed_from_nasri2018_rate_1e-3",
        help="registered SRIX parameter set",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/_generated/performance/srix_umat_gps_closure.json"),
        help="JSON output path",
    )
    arguments = parser.parse_args()
    if not arguments.library or not Path(arguments.library).is_file():
        parser.error("--library is required and must exist (set MFRONT_BEHAVIOUR_LIBRARY)")
    library = str(Path(arguments.library).resolve())

    report: dict[str, object] = {
        "preregistration": "validation/srix_umat_gps_closure_preregistration.md",
        "parameter_set": arguments.parameter_set,
        "history_increments": INCREMENTS,
        "history_max_strain": MAX_STRAIN,
        "cases": {},
    }
    summary: list[str] = []
    all_accepted = True
    for case_name, orientation in CASES.items():
        reference = _make_batch(
            "mfront-3d-condensed-plane-stress",
            library=library,
            parameter_set=arguments.parameter_set,
            orientation=orientation,
        )
        candidate = _make_batch(
            "mfront-native-generalised-plane-stress",
            library=library,
            parameter_set=arguments.parameter_set,
            orientation=orientation,
        )
        comparison = _compare(candidate, reference, HISTORY, tangent=True)
        case_report: dict[str, object] = dict(comparison)
        # Amendment 1 (2026-08-07) is RETRACTED (2026-08-08): the "two
        # branches" it documented were an artefact of the bridge applying the
        # total strain as an increment, fixed in 6bfaf86. The reference
        # agreement is a primary acceptance criterion again: A1 (in-plane
        # stress and transverse strains, relative L2) and A2 (condensed
        # tangent, relative L2), alongside the closed-system checks A3
        # (closure residual) and A6 (finite-difference tangent).
        case_report["a1_in_plane_stress_relative_l2"] = comparison[
            "in_plane_stress_relative_l2_max"
        ]
        case_report["a1_transverse_strain_relative_l2"] = comparison[
            "transverse_strain_relative_l2_max"
        ]
        case_report["a2_tangent_relative_l2"] = comparison["tangent_relative_l2_max"]
        candidate_probe = _make_batch(
            "mfront-native-generalised-plane-stress",
            library=library,
            parameter_set=arguments.parameter_set,
            orientation=orientation,
        )
        records = _run_history(candidate_probe, HISTORY)
        case_report["a3_global_transverse_residual_max_mpa"] = max(
            records["global_transverse_residual_max_mpa"]
        )
        # A6 was VACUOUS until 2026-08-07. `_run_history` commits every
        # increment, so probing at `HISTORY[-1]` afterwards asked the law for
        # the response to a strain increment of exactly ZERO -- the guarded
        # elastic branch -- and compared it against the elastic tangent. It
        # agreed to 1e-9 and reported a pass while the real plastic tangent was
        # wrong by a factor of ten. The check now runs at EVERY increment, from
        # the state committed before it, so every probe carries a genuine
        # non-zero increment, and the criterion is the worst of them.
        tangent_probe = _make_batch(
            "mfront-native-generalised-plane-stress",
            library=library,
            parameter_set=arguments.parameter_set,
            orientation=orientation,
        )
        fd_errors: list[float] = []
        for step in HISTORY:
            fd_errors.append(
                _finite_difference_tangent_check(
                    tangent_probe, step, time_increment=1.0 / INCREMENTS
                )
            )
            tangent_probe.evaluate(
                np.atleast_2d(step), time_increment=1.0 / INCREMENTS
            )
            tangent_probe.commit()
        case_report["a6_fd_tangent_relative_error"] = max(fd_errors)
        case_report["a6_fd_tangent_relative_error_per_increment"] = fd_errors
        accepted = (
            case_report["a1_in_plane_stress_relative_l2"] <= A1_TOLERANCE
            and case_report["a1_transverse_strain_relative_l2"] <= A1_TOLERANCE
            and case_report["a2_tangent_relative_l2"] <= A2_TOLERANCE
            and case_report["a3_global_transverse_residual_max_mpa"] <= A3_TOLERANCE_MPA
            and case_report["a6_fd_tangent_relative_error"] <= A6_TOLERANCE
        )
        case_report["accepted"] = bool(accepted)
        all_accepted = all_accepted and accepted
        report["cases"][case_name] = case_report
        summary.append(
            f"{case_name}: a1={case_report['a1_in_plane_stress_relative_l2']:.3e} "
            f"a2={case_report['a2_tangent_relative_l2']:.3e} "
            f"a3={case_report['a3_global_transverse_residual_max_mpa']:.3e} "
            f"a6={case_report['a6_fd_tangent_relative_error']:.3e} "
            f"{'ACCEPTED' if accepted else 'REJECTED'}"
        )

    # Case C2b (material-frame closure deviation) is removed: its mechanism,
    # the `GpsClosureFrame` parameter, was declared but never implemented in
    # the integrator and is deleted with the consolidation of 2026-08-08. A
    # falsifier that cannot run must not look registered.

    report["accepted"] = bool(all_accepted)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print("\n".join(summary))
    print(f"overall: {'ACCEPTED' if all_accepted else 'REJECTED'}")
    print(f"output: {arguments.output}")
    return 0 if all_accepted else 1


if __name__ == "__main__":
    raise SystemExit(_main())
