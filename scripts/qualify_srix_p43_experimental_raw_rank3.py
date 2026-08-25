#!/usr/bin/env python3
"""Identify experimental P43 SRIX from raw displacement mismatch only."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np

from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET
from fem_inhouse.identification.srix_equilibrium_gap import SrixTheta4
from fem_inhouse.identification.svd_parameter_basis import svd_parameter_basis
from scripts.qualify_srix_p43_experimental_rank3 import (
    START_Z,
    _matrix,
    _run_start,
)
from scripts.qualify_srix_p0043_synthetic_smoke import (
    CROP,
    _factory,
    _forward,
    _git,
    _load_inputs,
    _make_path,
    _vector,
)
from scripts.qualify_srix_regm_twin import _theta_from_preset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation/reference_data/p0043_experimental_raw_femu_m20_v1"
H = 1.5e-3


def _theta_from_payload(payload: dict[str, float]) -> SrixTheta4:
    return SrixTheta4(
        tau0_mpa=payload["tau0_mpa"],
        r_mpa=payload["R_mpa"],
        q_mpa=payload["Q_mpa"],
        b=payload["b"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=24)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)

    measured_macro, angles, provenance = _load_inputs(CROP)
    path = _make_path(measured_macro, 4)
    scored = tuple(4 * index for index in range(1, 9))
    target = [np.asarray(step.boundary, dtype=np.float64).copy() for step in path]
    prior = _theta_from_preset()
    eta_ref = prior.log_coordinates()
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )
    factory = _factory(angles, library, args.threads)
    prior_fields, prior_timing = _forward(prior, path, angles, library, args.threads)
    prior_residual = _vector(prior_fields, scored, target)
    prior_matrix = _matrix(
        prior_fields,
        scored,
        angles,
        prior,
        factory,
        library,
        args.threads,
        residual_scale=1.0,
    )
    basis = svd_parameter_basis(prior_matrix, fixed_rank=3)

    starts = [("nominal", np.zeros(3, dtype=np.float64))]
    starts.extend((name, np.asarray(values, dtype=np.float64)) for name, values in START_Z.items())
    results = []
    for name, initial_z in starts:
        results.append(
            _run_start(
                name,
                initial_z,
                eta_ref,
                basis,
                target,
                path,
                scored,
                angles,
                library,
                args.threads,
                args.max_nfev,
                residual_scale=1.0,
                optimizer_tolerances=(1.0e-10, 1.0e-12, 1.0e-12),
            )
        )

    best = min(results, key=lambda item: item["cost"]["final_displacement_rms_mm"])
    best_theta = _theta_from_payload(best["identified"])
    best_fields, best_timing = _forward(best_theta, path, angles, library, args.threads)
    final_matrix = _matrix(
        best_fields,
        scored,
        angles,
        best_theta,
        factory,
        library,
        args.threads,
        residual_scale=1.0,
    )
    final_basis = svd_parameter_basis(final_matrix, fixed_rank=3)
    bound = float(np.log(4.0))
    bound_flags = {
        item["name"]: [
            bool(abs(value - bound) < 1.0e-7 or abs(value + bound) < 1.0e-7)
            for value in item["z_final"]
        ]
        for item in results
    }
    gate = {
        "all_optimizers_converged": all(item["optimizer"]["success"] for item in results),
        "all_costs_decreased": all(
            item["cost"]["final_displacement_rms_mm"]
            < item["cost"]["initial_displacement_rms_mm"]
            for item in results
        ),
        "any_parameter_bound_active": any(any(flags) for flags in bound_flags.values()),
    }
    gate["m100_authorized"] = bool(
        gate["all_optimizers_converged"]
        and gate["all_costs_decreased"]
        and not gate["any_parameter_bound_active"]
    )
    report = {
        "schema_version": 1,
        "method": "P43 experimental raw-displacement FEMU rank-three SVD identification M20",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "crop": list(CROP),
        "path_steps": len(path),
        "scored_steps": list(scored),
        "observation_weighting": "none",
        "noise_model_used": False,
        "covariance_used": False,
        "residual_units": "mm",
        "observation_profile": "measured displacement identity; raw physical mismatch",
        "parameter_preset": DEFAULT_PARAMETER_SET,
        "shadow_fd_step": H,
        "provenance": provenance,
        "prior": prior.as_runtime_overrides(),
        "prior_forward": prior_timing,
        "best_forward": best_timing,
        "prior_raw_rms_mm": float(np.sqrt(np.mean(prior_residual**2))),
        "prior_svd": {
            "singular_values": basis.singular_values.tolist(),
            "normalized_singular_values": basis.normalized_singular_values.tolist(),
            "right_singular_vectors": basis.right_singular_vectors.tolist(),
            "retained_basis": basis.retained_basis.tolist(),
            "discarded_basis": basis.discarded_basis.tolist(),
        },
        "starts": results,
        "best_start": best["name"],
        "parameter_bound_activity": bound_flags,
        "final_svd": {
            "singular_values": final_basis.singular_values.tolist(),
            "normalized_singular_values": final_basis.normalized_singular_values.tolist(),
            "right_singular_vectors": final_basis.right_singular_vectors.tolist(),
            "retained_basis": final_basis.retained_basis.tolist(),
        },
        "gate": gate,
        "claims": {
            "experimental_raw_m20_completed": True,
            "experimental_raw_m100_authorized": gate["m100_authorized"],
            "experimental_parameters_identified": False,
        },
    }
    np.savez_compressed(
        output / "fields.npz",
        prior_displacement=np.asarray([field.displacement for field in prior_fields]),
        best_displacement=np.asarray([field.displacement for field in best_fields]),
        target_displacement=np.asarray(target),
        prior_jacobian=prior_matrix,
        final_jacobian=final_matrix,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["gate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
