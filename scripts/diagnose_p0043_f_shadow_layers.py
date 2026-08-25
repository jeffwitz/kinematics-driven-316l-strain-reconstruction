#!/usr/bin/env python3
"""Locate the first F-mapping projected-shadow/FD disagreement by layer.

This intentionally follows only one observable SVD mode over the first few
accepted increments.  It compares the fixed-strain forcing and tangent solve
against two full centred F forwards before propagating the shadow histories.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from fem_inhouse.core.plane_stress_material import evaluate_in_plane_response
from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET, get_parameter_set
from fem_inhouse.identification.srix_parameter_coordinates import SrixTheta9
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior
from scripts.qualify_srix_p0043_synthetic_smoke import (
    CROP,
    _factory,
    _forward,
    _load_inputs,
    _make_path,
)
from scripts.qualify_srix_regm_twin import PIXEL_SIZE_MM
from scripts.qualify_srix_svd_shadow import _solve_tangent

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "validation/reference_data/p0043_f_shadow_layer_diagnostic_v1"
SVD = ROOT / "validation/reference_data/p0043_f_mapping_reidentification_v1/svd_f.json"


def _relative(actual: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    return float(np.linalg.norm(actual - reference) / max(denom, np.finfo(float).tiny))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left.ravel(), right.ravel()) / max(denominator, np.finfo(float).tiny))


def _scaled_norm(values: np.ndarray, scale: np.ndarray) -> float:
    return float(
        np.linalg.norm(values) / max(float(np.linalg.norm(scale)), np.finfo(float).tiny)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--increments", type=int, default=4)
    parser.add_argument("--mode", type=int, default=0, help="zero-based SVD mode")
    parser.add_argument("--step", type=float, default=1.5e-3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)

    measured_macro, angles, _ = _load_inputs(CROP)
    path = _make_path(measured_macro, 4)
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )
    eta = SrixTheta9.from_parameter_set(
        get_parameter_set(DEFAULT_PARAMETER_SET)
    ).log_coordinates()
    right_vectors = np.asarray(
        json.loads(SVD.read_text(encoding="utf-8"))["right_singular_vectors"]
    )
    direction = right_vectors[:, args.mode]
    step = float(args.step)

    # These three non-linear trajectories are the directional FD oracle.  The
    # one-mode shadow below uses their fields only for a layer-by-layer check.
    base, base_timing = _forward(
        SrixTheta9.from_log_coordinates(eta), path, angles, library, args.threads, "F"
    )
    plus, plus_timing = _forward(
        SrixTheta9.from_log_coordinates(eta + step * direction),
        path,
        angles,
        library,
        args.threads,
        "F",
    )
    minus, minus_timing = _forward(
        SrixTheta9.from_log_coordinates(eta - step * direction),
        path,
        angles,
        library,
        args.threads,
        "F",
    )

    pixels = angles.shape[0]
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    kinematics = TwoSubcellDiagnostic2D(grid)
    factory = _factory(angles, library, args.threads, "F")
    shadow_plus = factory(
        SrixTheta9.from_log_coordinates(eta + step * direction).as_runtime_overrides()
    )
    shadow_minus = factory(
        SrixTheta9.from_log_coordinates(eta - step * direction).as_runtime_overrides()
    )

    def tangent_action(tangent: np.ndarray, vector: np.ndarray) -> np.ndarray:
        nodal = unpack_interior(vector, grid)
        strain = kinematics.strain_samples(nodal)
        stress = np.einsum("xyqij,xyqj->xyqi", tangent, strain)
        return pack_interior(kinematics.divergence_from_sample_stress(stress))

    records: list[dict[str, float | int]] = []
    for index, accepted in enumerate(base[: args.increments], start=1):
        strain = np.asarray(accepted.sample_strain, dtype=np.float64)
        tangent = np.asarray(accepted.algorithmic_tangent_in_plane_mpa, dtype=np.float64)
        plus_trial = evaluate_in_plane_response(
            shadow_plus, strain.reshape(-1, 3), time_increment=accepted.time_increment,
            response_level="tangent", consistent_tangent=True,
        )
        minus_trial = evaluate_in_plane_response(
            shadow_minus, strain.reshape(-1, 3), time_increment=accepted.time_increment,
            response_level="tangent", consistent_tangent=True,
        )
        flat_difference = (
            np.asarray(plus_trial.stress_in_plane_mpa)
            - np.asarray(minus_trial.stress_in_plane_mpa)
        ) / (2.0 * step)

        # The spectral forward evaluates and reshapes two-state material values
        # in C order; retain an F candidate only as a diagnostic control.
        difference_c = flat_difference.reshape(*grid.pixel_shape, 2, 3)
        difference_f = flat_difference.reshape(*grid.pixel_shape, 2, 3, order="F")
        rhs_c = -pack_interior(kinematics.divergence_from_sample_stress(difference_c))
        rhs_f = -pack_interior(kinematics.divergence_from_sample_stress(difference_f))
        shadow_plus.revert()
        shadow_minus.revert()

        vector_c = _solve_tangent(grid, kinematics, tangent, rhs_c)
        sensitivity_c = unpack_interior(vector_c, grid)
        fd_sensitivity = (
            np.asarray(plus[index - 1].displacement)
            - np.asarray(minus[index - 1].displacement)
        ) / (2.0 * step)
        fd_interior = pack_interior(fd_sensitivity)
        tangent_defect = tangent_action(tangent, fd_interior) - rhs_c
        records.append({
            "increment": index,
            "start_fraction": float(accepted.start_fraction),
            "end_fraction": float(accepted.end_fraction),
            "forcing_c_vs_f_relative": _relative(rhs_c, rhs_f),
            "forcing_c_vs_f_cosine": _cosine(rhs_c, rhs_f),
            "tangent_fd_defect_relative": _scaled_norm(tangent_defect, rhs_c),
            "u_shadow_vs_fd_relative": _relative(pack_interior(sensitivity_c), fd_interior),
            "u_shadow_vs_fd_cosine": _cosine(pack_interior(sensitivity_c), fd_interior),
        })

        strain_sensitivity = kinematics.strain_samples(sensitivity_c)
        for shadow, sign in ((shadow_plus, 1.0), (shadow_minus, -1.0)):
            evaluate_in_plane_response(
                shadow, (strain + sign * step * strain_sensitivity).reshape(-1, 3),
                time_increment=accepted.time_increment,
                response_level="residual", consistent_tangent=False,
            )
            shadow.commit()

    report = {
        "element_order": "F",
        "spectral_material_batch_order": "C",
        "mode": args.mode,
        "step": step,
        "increments": records,
        "forward_timing": {"base": base_timing, "plus": plus_timing, "minus": minus_timing},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(records, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
