"""Compare legacy and Generic SRIX on the same real EBSD orientation crop.

This is a deliberately small, reproducible integration check.  It uses the
co-registered orientation fields from ``CP_dataset.h5`` and the same scalar
accumulated-slip Helmholtz coupling for both material backends.  It is not a
production P43 benchmark; its purpose is to prove that the Generic bridge
preserves the legacy solution on a non-synthetic orientation map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

from fem_inhouse.examples import reduced_biaxial_case
from fem_inhouse.solver import run_case_study

try:
    from scripts.benchmark_tri2_j2_krylov import _load_case
except ModuleNotFoundError:  # Direct script execution.
    from benchmark_tri2_j2_krylov import _load_case  # type: ignore[no-redef]

DEFAULT_EBSD = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")


def _load_angles(
    path: Path, crop: tuple[int, int, int, int]
) -> tuple[np.ndarray, dict[str, object]]:
    x0, x1, y0, y1 = crop
    with h5py.File(path, "r") as handle:
        fields = [
            np.asarray(handle[f"orientation/{name}"][x0:x1, y0:y1], dtype=float)
            for name in ("phi1", "Phi", "phi2")
        ]
        shape = tuple(handle["orientation/phi1"].shape)
    angles = np.stack(fields, axis=-1)
    if angles.shape != (x1 - x0, y1 - y0, 3):
        raise ValueError(f"unexpected EBSD crop shape: {angles.shape}")
    return angles, {
        "source_file": str(path),
        "source_shape": list(shape),
        "crop_nodes": list(crop),
        "angles_shape": list(angles.shape),
        "angles_sha256": hashlib.sha256(np.ascontiguousarray(angles).tobytes()).hexdigest(),
    }


def _run(
    case,
    *,
    backend: str,
    library: str,
    behaviour: str,
    angles: np.ndarray,
    increments: int,
    tolerance: float,
    boundary_history: np.ndarray,
    displacement_x: np.ndarray,
    displacement_y: np.ndarray,
    yield_stress: np.ndarray,
    hardening: np.ndarray,
):
    solver = replace(
        case.config.solver,
        constitutive_backend=backend,
        mfront_library=library,
        mfront_behaviour_id=behaviour,
        constitutive_options={
            "crystal_orientation": {
                "mode": "ebsd",
                "euler_bunge_deg": angles.tolist(),
            }
        },
        increments=increments,
        residual_tolerance=tolerance,
        max_newton_iterations=20,
        minimum_step_divisor=32,
        mfront_threads=1,
    )
    nonlocal_config = replace(
        case.config.nonlocal_plasticity,
        enabled=True,
        length_scale_mm=0.05888,
        coupling_modulus_mpa=100.0,
        criterion="accumulated_slip_helmholtz",
        relative_tolerance=1e-6,
        maximum_iterations=15,
    )
    started = time.perf_counter()
    result = run_case_study(
        replace(case.config, solver=solver, nonlocal_plasticity=nonlocal_config),
        displacement_x_mm=displacement_x,
        displacement_y_mm=displacement_y,
        yield_stress_mpa=yield_stress,
        hardening_coefficient_mpa=hardening,
        boundary_displacement_history_mm=boundary_history,
    )
    return result, time.perf_counter() - started


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(a)), 1e-30)
    return float(np.linalg.norm(a - b) / denominator)


def _median_mad(values: list[float]) -> tuple[float, float]:
    samples = np.asarray(values, dtype=float)
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    return median, mad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ebsd-orientation-h5", type=Path, default=DEFAULT_EBSD)
    parser.add_argument(
        "--crop-nodes", nargs=4, type=int, default=(1610, 1613, 1075, 1078)
    )
    parser.add_argument("--increments", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    legacy_library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    generic_library = os.environ.get("SRIX_GENERIC_MFRONT_BEHAVIOUR_LIBRARY")
    if not legacy_library or not generic_library:
        raise SystemExit(
            "set MFRONT_BEHAVIOUR_LIBRARY and SRIX_GENERIC_MFRONT_BEHAVIOUR_LIBRARY"
        )
    crop = tuple(args.crop_nodes)
    angles, provenance = _load_angles(args.ebsd_orientation_h5, crop)
    mesh = angles.shape[0]
    case = reduced_biaxial_case(nx=mesh, ny=mesh)
    _, _history, yield_stress, hardening, boundary = _load_case(mesh, crop)
    def runs(backend, library, behaviour):
        return [
            _run(
                case,
                backend=backend,
                library=library,
                behaviour=behaviour,
                angles=angles,
                increments=args.increments,
                tolerance=args.tolerance,
                boundary_history=np.stack(
                    [fraction * boundary for fraction in np.linspace(0.0, 1.0, args.increments + 1)]
                ),
                displacement_x=boundary[..., 0],
                displacement_y=boundary[..., 1],
                yield_stress=yield_stress.reshape(mesh, mesh),
                hardening=hardening.reshape(mesh, mesh),
            )
            for _ in range(args.repeats)
        ]

    try:
        legacy_runs = runs(
            "mfront-3d-condensed-plane-stress", legacy_library, "fcc_forest_rubin_srix"
        )
    except Exception as error:
        raise SystemExit(f"legacy backend failed on P43 crop: {error}") from error
    try:
        generic_runs = runs(
            "mfront-srix-generic-plane-stress",
            generic_library,
            "fcc_forest_rubin_srix_generic_validation",
        )
    except Exception as error:
        raise SystemExit(f"Generic backend failed on P43 crop: {error}") from error
    legacy, legacy_times = legacy_runs[-1][0], [run[1] for run in legacy_runs]
    generic, generic_times = generic_runs[-1][0], [run[1] for run in generic_runs]
    legacy_median, legacy_mad = _median_mad(legacy_times)
    generic_median, generic_mad = _median_mad(generic_times)

    report = {
        "status": "ok",
        "provenance": provenance,
        "increments": args.increments,
        "tolerance": args.tolerance,
        "repeats": args.repeats,
        "legacy": {
            "elapsed_seconds_last": legacy_times[-1],
            "elapsed_seconds_samples": legacy_times,
            "elapsed_seconds_median": legacy_median,
            "elapsed_seconds_mad": legacy_mad,
            "converged_increments": legacy.diagnostics.converged_increments,
            "cutbacks": legacy.diagnostics.cutbacks,
            "maximum_plane_stress_residual_mpa": (
                legacy.diagnostics.maximum_gauss_point_plane_stress_residual_mpa
            ),
        },
        "generic": {
            "elapsed_seconds_last": generic_times[-1],
            "elapsed_seconds_samples": generic_times,
            "elapsed_seconds_median": generic_median,
            "elapsed_seconds_mad": generic_mad,
            "converged_increments": generic.diagnostics.converged_increments,
            "cutbacks": generic.diagnostics.cutbacks,
            "maximum_plane_stress_residual_mpa": (
                generic.diagnostics.maximum_gauss_point_plane_stress_residual_mpa
            ),
        },
        "relative_errors": {
            "displacement": _relative_error(generic.displacement_mm, legacy.displacement_mm),
            "stress": _relative_error(generic.stress_mpa, legacy.stress_mpa),
            "accumulated_slip": _relative_error(generic.cumulated_slip, legacy.cumulated_slip),
            "nonlocal_source": _relative_error(
                generic.nonlocal_equivalent_plastic_strain,
                legacy.nonlocal_equivalent_plastic_strain,
            ),
        },
        "timing_ratio_generic_over_legacy_median": generic_median / legacy_median,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
