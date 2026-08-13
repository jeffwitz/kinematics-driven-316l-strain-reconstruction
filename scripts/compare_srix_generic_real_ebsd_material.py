"""Compare the two SRIX material bridges on a real EBSD orientation crop.

Unlike the global solver comparison, this diagnostic needs neither MKL nor
PyPardiso. It isolates the constitutive/plane-stress bridge and reports the
cost and response of one batch evaluation at a controlled plastic state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np

from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch

DEFAULT_EBSD = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")


def _angles(path: Path, crop: tuple[int, int, int, int]) -> tuple[np.ndarray, dict[str, object]]:
    x0, x1, y0, y1 = crop
    with h5py.File(path, "r") as handle:
        values = [
            np.asarray(handle[f"orientation/{name}"][x0:x1, y0:y1], dtype=float)
            for name in ("phi1", "Phi", "phi2")
        ]
        source_shape = list(handle["orientation/phi1"].shape)
    result = np.stack(values, axis=-1)
    return result, {
        "source_file": str(path),
        "source_shape": source_shape,
        "crop_nodes": list(crop),
        "angles_shape": list(result.shape),
        "angles_sha256": hashlib.sha256(np.ascontiguousarray(result).tobytes()).hexdigest(),
    }


def _material(
    *, backend: str, library: str, behaviour: str, angles: np.ndarray
):
    points = angles.shape[0] * angles.shape[1]
    return create_plane_stress_material_batch(
        backend,
        np.full(points, 124.0),
        np.full(points, 380.0),
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.3,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1000,
        first_positive_plastic_strain=1e-6,
        mfront_library=library,
        mfront_threads=1,
        mfront_behaviour_id=behaviour,
        nonlocal_coupling_modulus_mpa=100.0,
        constitutive_options={
            "crystal_orientation": {
                "mode": "ebsd",
                "euler_bunge_deg": angles.tolist(),
            }
        },
    )


def _evaluate(material, strain: np.ndarray, chi: np.ndarray, repeats: int):
    times: list[float] = []
    complete = None
    for _ in range(repeats):
        material.set_nonlocal_equivalent_plastic_strain(chi)
        started = time.perf_counter()
        response = material.evaluate_in_plane_response(
            strain,
            time_increment=1.0,
            response_level="complete",
            consistent_tangent=True,
        )
        complete = (
            response
            if hasattr(response, "full_stress_tensor_mpa")
            else material.complete_trial(response)
        )
        times.append(time.perf_counter() - started)
        material.revert()
    if complete is None:  # pragma: no cover - repeats is validated by the CLI
        raise RuntimeError("no constitutive trial was evaluated")
    source = np.asarray(complete.observables["accumulated_slip"], dtype=float)
    return complete, times, source


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(a)), 1e-30))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ebsd-orientation-h5", type=Path, default=DEFAULT_EBSD)
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=(1610, 1613, 1075, 1078))
    parser.add_argument("--repeats", type=int, default=5)
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

    angles, provenance = _angles(args.ebsd_orientation_h5, tuple(args.crop_nodes))
    points = angles.shape[0] * angles.shape[1]
    strain = np.tile(np.array([4.0e-3, -1.2e-3, 8.0e-4]), (points, 1))
    chi = np.linspace(0.0, 8.0e-4, points)
    legacy = _material(
        backend="mfront-3d-condensed-plane-stress",
        library=legacy_library,
        behaviour="fcc_forest_rubin_srix",
        angles=angles,
    )
    generic = _material(
        backend="mfront-srix-generic-plane-stress",
        library=generic_library,
        behaviour="fcc_forest_rubin_srix_generic_validation",
        angles=angles,
    )
    legacy_trial, legacy_times, legacy_source = _evaluate(legacy, strain, chi, args.repeats)
    generic_trial, generic_times, generic_source = _evaluate(generic, strain, chi, args.repeats)
    legacy_median = float(np.median(legacy_times))
    generic_median = float(np.median(generic_times))
    report = {
        "status": "ok",
        "provenance": provenance,
        "points": points,
        "strain": [4.0e-3, -1.2e-3, 8.0e-4],
        "chi_range": [float(chi.min()), float(chi.max())],
        "repeats": args.repeats,
        "legacy": {
            "times_seconds": legacy_times,
            "median_seconds": legacy_median,
            "plane_stress_residual_max_mpa": float(
                np.max(np.abs(legacy_trial.plane_stress_residual_mpa))
            ),
        },
        "generic": {
            "times_seconds": generic_times,
            "median_seconds": generic_median,
            "plane_stress_residual_max_mpa": float(
                np.max(np.abs(generic_trial.plane_stress_residual_mpa))
            ),
        },
        "relative_errors": {
            "stress": _relative(
                generic_trial.stress_in_plane_mpa, legacy_trial.stress_in_plane_mpa
            ),
            "accumulated_slip": _relative(generic_source, legacy_source),
            "tangent": _relative(
                generic_trial.tangent_in_plane_mpa, legacy_trial.tangent_in_plane_mpa
            ),
        },
        "timing_ratio_generic_over_legacy": generic_median / legacy_median,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
