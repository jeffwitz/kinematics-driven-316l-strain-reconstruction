"""Diagnose the 85-vs-57 global Newton penalty on real EBSD states.

The UMAT GPS backend agrees with the reference at the material-point level
(A1/A2 ~ 1e-11 on three orientations) yet needs 85 global Newton iterations
against the reference's 57 on P43 100x100. Two untested candidates remain: a
tangent less exact at orientations and states the three-case qualification
does not cover, or a genuine trajectory divergence. This script samples the
real states of the archived M100 run -- per-element strain derived from the
displacement field, orientation from the EBSD crop -- and measures, per
sample:

- the UMAT returned tangent against finite differences (its own quality);
- the UMAT tangent against the reference condensed tangent (the Jv operator
  the two global Newtons actually use).

The per-point committed internal state is not archived, so each sample is
evaluated as a single increment from the virgin state; the strain LEVEL and
the ORIENTATION are the ones actually encountered, which is what the penalty
hypotheses are about.

Usage:

    .venv/bin/python scripts/diagnose_gps_tangent_on_ebsd_states.py \
        --fields validation/_generated/performance/srix_p43_100x100_umat_gps/mfront_3d_condensed_plane_stress.fields.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _per_element_strain(displacement: np.ndarray) -> np.ndarray:
    """Central-difference strain at element centres from the nodal field.

    ``displacement`` is (nx+1, ny+1, 2); the return is (nx, ny, 3) with the
    engineering in-plane triple (exx, eyy, gxy).
    """

    from scripts.benchmark_tri2_j2_krylov import PIXEL_SIZE_MM

    ux = displacement[..., 0]
    uy = displacement[..., 1]
    # The spectral strain is du/dx with x in mm: the index gradient must be
    # divided by the pixel spacing.
    grad_x = np.gradient(ux, axis=0) / PIXEL_SIZE_MM
    grad_y = np.gradient(uy, axis=1) / PIXEL_SIZE_MM
    shear_x = np.gradient(uy, axis=0) / PIXEL_SIZE_MM
    shear_y = np.gradient(ux, axis=1) / PIXEL_SIZE_MM
    # Element centres from the nodal gradient.
    exx = 0.5 * (grad_x[:-1, :-1] + grad_x[1:, 1:])
    eyy = 0.5 * (grad_y[:-1, :-1] + grad_y[1:, 1:])
    gxy = 0.5 * (shear_x[:-1, :-1] + shear_x[1:, 1:]) + 0.5 * (
        shear_y[:-1, :-1] + shear_y[1:, 1:]
    )
    return np.stack((exx, eyy, gxy), axis=-1)


def _sample_indices(
    strain: np.ndarray, count: int, seed: int = 20260808
) -> np.ndarray:
    """Uniform sample plus a band-weighted one (the zones dominating the
    residual)."""

    rng = np.random.default_rng(seed)
    flat = strain.reshape(-1, 3)
    norm = np.linalg.norm(flat, axis=-1)
    uniform = rng.choice(flat.shape[0], size=count // 2, replace=False)
    weights = norm / max(norm.sum(), 1.0e-30)
    weighted = rng.choice(flat.shape[0], size=count - count // 2, replace=False, p=weights)
    return np.concatenate((uniform, weighted))


def _fd_tangent(
    batch: object, strain: np.ndarray, time_increment: float, perturbation: float = 1e-6
) -> np.ndarray:
    """Central-difference tangent of the returned in-plane stress."""

    fd = np.zeros((3, 3))
    for column in range(3):
        plus = strain.copy()
        minus = strain.copy()
        plus[column] += perturbation
        minus[column] -= perturbation
        stress_plus = np.asarray(
            batch.evaluate(np.atleast_2d(plus), time_increment=time_increment).stress_in_plane_mpa
        )[0]
        stress_minus = np.asarray(
            batch.evaluate(np.atleast_2d(minus), time_increment=time_increment).stress_in_plane_mpa
        )[0]
        fd[:, column] = (stress_plus - stress_minus) / (2 * perturbation)
    return fd


def _main() -> int:
    import os

    from fem_inhouse.core.crystal_orientation import rotation_from_euler_bunge_deg
    from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument(
        "--library", default=os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    )
    parser.add_argument(
        "--ebsd-orientation-h5",
        type=Path,
        default=Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5"),
    )
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if not arguments.library:
        parser.error("--library is required")

    crop = (1570, 1670, 1035, 1135)
    from scripts.qualify_crystal_tet2_p43 import _load_ebsd_orientation_crop

    angles, _ = _load_ebsd_orientation_crop(arguments.ebsd_orientation_h5, crop)
    with np.load(arguments.fields) as fields:
        displacement = np.asarray(fields["displacement"])
    strain = _per_element_strain(displacement)
    indices = _sample_indices(strain, arguments.samples)
    sampled = strain.reshape(-1, 3)[indices]
    rows, columns = np.unravel_index(indices, strain.shape[:2])

    def make(backend: str) -> object:
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
            mfront_library=arguments.library,
            mfront_threads=1,
            mfront_behaviour_id="fcc_forest_rubin_srix",
            constitutive_options={
                "parameter_set": "316l_srix_transposed_from_nasri2018_rate_1e-3",
                "crystal_orientation": {
                    "mode": "homogeneous",
                    "matrix": np.asarray(
                        rotation_from_euler_bunge_deg(35.0, 20.0, 15.0), dtype=float
                    ).tolist(),
                },
            },
        )

    records = []
    for index, (row, column, sample) in enumerate(zip(rows, columns, sampled, strict=True)):
        orientation = angles[row, column]
        options: dict[str, object] = {
            "parameter_set": "316l_srix_transposed_from_nasri2018_rate_1e-3",
            "crystal_orientation": {
                "mode": "homogeneous",
                "matrix": np.asarray(
                    rotation_from_euler_bunge_deg(*orientation), dtype=float
                ).tolist(),
            },
        }
        umat = create_plane_stress_material_batch(
            "mfront-native-generalised-plane-stress",
            np.full((1, 1), 250.0),
            np.full((1, 1), 500.0),
            0.245,
            young_modulus_mpa=205000.0,
            poisson_ratio=0.3,
            hardening_mode="ludwik",
            plastic_strain_max=0.2,
            plastic_table_points=1000,
            first_positive_plastic_strain=1e-6,
            mfront_library=arguments.library,
            mfront_threads=1,
            mfront_behaviour_id="fcc_forest_rubin_srix",
            constitutive_options=options,
        )
        reference = create_plane_stress_material_batch(
            "mfront-3d-condensed-plane-stress",
            np.full((1, 1), 250.0),
            np.full((1, 1), 500.0),
            0.245,
            young_modulus_mpa=205000.0,
            poisson_ratio=0.3,
            hardening_mode="ludwik",
            plastic_strain_max=0.2,
            plastic_table_points=1000,
            first_positive_plastic_strain=1e-6,
            mfront_library=arguments.library,
            mfront_threads=1,
            mfront_behaviour_id="fcc_forest_rubin_srix",
            constitutive_options=options,
        )
        # Drive each batch through a short proportional history to the sampled
        # strain: a single jump at the final-state strain level is not a state
        # the material sees, and the reference's closure does not converge on
        # it from the virgin state.
        steps = 8
        time_increment = 1.0 / steps
        try:
            for fraction in np.linspace(1.0 / steps, 1.0, steps):
                umat.evaluate(np.atleast_2d(fraction * sample), time_increment=time_increment)
                umat.commit()
                reference.evaluate(np.atleast_2d(fraction * sample), time_increment=time_increment)
                reference.commit()
            umat_trial = umat.evaluate(np.atleast_2d(sample), time_increment=time_increment)
            ref_trial = reference.evaluate(np.atleast_2d(sample), time_increment=time_increment)
            umat_tangent = np.asarray(umat_trial.tangent_in_plane_mpa)[0]
            ref_tangent = np.asarray(ref_trial.tangent_in_plane_mpa)[0]
            fd = _fd_tangent(umat, sample, time_increment)
        except Exception as error:
            records.append(
                {
                    "index": int(index),
                    "row": int(row),
                    "column": int(column),
                    "strain_norm": float(np.linalg.norm(sample)),
                    "failure": f"{type(error).__name__}: {str(error)[:120]}",
                }
            )
            continue
        umat.revert()
        scale = max(np.max(np.abs(fd)), 1.0e-30)
        records.append(
            {
                "index": int(index),
                "row": int(row),
                "column": int(column),
                "strain_norm": float(np.linalg.norm(sample)),
                "umat_fd_relative_error": float(np.max(np.abs(umat_tangent - fd)) / scale),
                "umat_vs_reference_relative": float(
                    np.max(np.abs(umat_tangent - ref_tangent))
                    / max(np.max(np.abs(ref_tangent)), 1.0e-30)
                ),
            }
        )

    fd_errors = np.array([r["umat_fd_relative_error"] for r in records])
    ref_errors = np.array([r["umat_vs_reference_relative"] for r in records])
    report = {
        "samples": arguments.samples,
        "umat_fd_relative_error": {
            "min": float(fd_errors.min()),
            "median": float(np.median(fd_errors)),
            "p90": float(np.percentile(fd_errors, 90)),
            "max": float(fd_errors.max()),
            "over_1e-6": int(np.sum(fd_errors > 1.0e-6)),
        },
        "umat_vs_reference_relative": {
            "min": float(ref_errors.min()),
            "median": float(np.median(ref_errors)),
            "p90": float(np.percentile(ref_errors, 90)),
            "max": float(ref_errors.max()),
            "over_1e-6": int(np.sum(ref_errors > 1.0e-6)),
        },
        "points": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"UMAT FD tangent error: median {np.median(fd_errors):.2e}, "
          f"p90 {np.percentile(fd_errors, 90):.2e}, max {fd_errors.max():.2e}, "
          f"{int(np.sum(fd_errors > 1e-6))}/{arguments.samples} over 1e-6")
    print(f"UMAT vs reference tangent: median {np.median(ref_errors):.2e}, "
          f"p90 {np.percentile(ref_errors, 90):.2e}, max {ref_errors.max():.2e}, "
          f"{int(np.sum(ref_errors > 1e-6))}/{arguments.samples} over 1e-6")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
