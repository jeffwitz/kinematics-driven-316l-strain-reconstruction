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
        displacement_x_mm=case.displacement_x_mm,
        displacement_y_mm=case.displacement_y_mm,
        yield_stress_mpa=case.yield_stress_mpa,
        hardening_coefficient_mpa=case.hardening_coefficient_mpa,
    )
    return result, time.perf_counter() - started


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(a)), 1e-30)
    return float(np.linalg.norm(a - b) / denominator)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ebsd-orientation-h5", type=Path, default=DEFAULT_EBSD)
    parser.add_argument(
        "--crop-nodes", nargs=4, type=int, default=(1610, 1613, 1075, 1078)
    )
    parser.add_argument("--increments", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    legacy_library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    generic_library = os.environ.get("SRIX_GENERIC_MFRONT_BEHAVIOUR_LIBRARY")
    if not legacy_library or not generic_library:
        raise SystemExit(
            "set MFRONT_BEHAVIOUR_LIBRARY and SRIX_GENERIC_MFRONT_BEHAVIOUR_LIBRARY"
        )
    crop = tuple(args.crop_nodes)
    angles, provenance = _load_angles(args.ebsd_orientation_h5, crop)
    case = reduced_biaxial_case(nx=angles.shape[0], ny=angles.shape[1])
    legacy, legacy_time = _run(
        case,
        backend="mfront-3d-condensed-plane-stress",
        library=legacy_library,
        behaviour="fcc_forest_rubin_srix",
        angles=angles,
        increments=args.increments,
        tolerance=args.tolerance,
    )
    generic, generic_time = _run(
        case,
        backend="mfront-srix-generic-plane-stress",
        library=generic_library,
        behaviour="fcc_forest_rubin_srix_generic_validation",
        angles=angles,
        increments=args.increments,
        tolerance=args.tolerance,
    )

    report = {
        "status": "ok",
        "provenance": provenance,
        "increments": args.increments,
        "tolerance": args.tolerance,
        "legacy": {
            "elapsed_seconds": legacy_time,
            "converged_increments": legacy.diagnostics.converged_increments,
            "cutbacks": legacy.diagnostics.cutbacks,
            "maximum_plane_stress_residual_mpa": (
                legacy.diagnostics.maximum_gauss_point_plane_stress_residual_mpa
            ),
        },
        "generic": {
            "elapsed_seconds": generic_time,
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
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
