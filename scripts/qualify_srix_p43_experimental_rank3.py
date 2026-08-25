#!/usr/bin/env python3
"""Identify SRIX on real P43 M20 data with a fixed rank-three SVD basis."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET
from fem_inhouse.identification.srix_equilibrium_gap import SrixTheta4
from fem_inhouse.identification.svd_parameter_basis import (
    eta_from_reduced_coordinates,
    svd_parameter_basis,
)
from scripts.qualify_srix_femu_direct_sensitivity import _direct_jacobian
from scripts.qualify_srix_p0043_synthetic_smoke import (
    CROP,
    _factory,
    _forward,
    _git,
    _Identity,
    _load_inputs,
    _make_path,
    _vector,
)
from scripts.qualify_srix_regm_twin import _theta_from_preset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation/reference_data/p0043_experimental_srix_m20_v1"
PIXEL_NOISE_MM = 9.40e-5
H = 1.5e-3
START_Z = {
    "E1": (0.12, -0.08, 0.10),
    "E2": (-0.10, 0.10, -0.12),
    "E3": (0.08, 0.12, -0.08),
}


def _matrix(
    fields: list[Any],
    scored: tuple[int, ...],
    angles: np.ndarray,
    theta: SrixTheta4,
    factory: Any,
    library: str,
    threads: int,
) -> np.ndarray:
    matrix, _ = _direct_jacobian(
        fields=fields,
        scored=scored,
        orientations=angles,
        theta=theta,
        library=library,
        threads=threads,
        transfer=_Identity(),
        h=H,
        material_factory=factory,
    )
    return matrix / PIXEL_NOISE_MM


def _run_start(
    name: str,
    initial_z: np.ndarray,
    eta_ref: np.ndarray,
    basis: Any,
    target: list[np.ndarray],
    path: list[Any],
    scored: tuple[int, ...],
    angles: np.ndarray,
    library: str,
    threads: int,
    max_nfev: int,
) -> dict[str, Any]:
    factory = _factory(angles, library, threads)
    cache: dict[bytes, tuple[list[Any], np.ndarray]] = {}
    forwards: list[dict[str, Any]] = []
    jacobian_records: list[dict[str, Any]] = []

    def evaluate(z: np.ndarray) -> tuple[list[Any], np.ndarray]:
        eta = eta_from_reduced_coordinates(eta_ref, basis.retained_basis, z)
        key = np.asarray(eta, dtype=np.float64).tobytes()
        if key not in cache:
            theta = SrixTheta4.from_log_coordinates(eta)
            fields, timing = _forward(theta, path, angles, library, threads)
            residual = _vector(fields, scored, target) / PIXEL_NOISE_MM
            cache[key] = (fields, residual)
            forwards.append({"theta": theta.as_runtime_overrides(), **timing})
        return cache[key]

    initial_fields, initial_residual = evaluate(initial_z)

    def residual(z: np.ndarray) -> np.ndarray:
        return evaluate(z)[1]

    def jacobian(z: np.ndarray) -> np.ndarray:
        fields, _ = evaluate(z)
        eta = eta_from_reduced_coordinates(eta_ref, basis.retained_basis, z)
        started = time.perf_counter()
        matrix = _matrix(
            fields,
            scored,
            angles,
            SrixTheta4.from_log_coordinates(eta),
            factory,
            library,
            threads,
        )
        jacobian_records.append({"seconds": time.perf_counter() - started})
        return matrix @ basis.retained_basis

    started = time.perf_counter()
    fit = least_squares(
        residual,
        initial_z,
        jac=jacobian,
        bounds=(-np.log(4.0) * np.ones(3), np.log(4.0) * np.ones(3)),
        x_scale="jac",
        max_nfev=max_nfev,
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=1.0e-8,
    )
    final_eta = eta_from_reduced_coordinates(eta_ref, basis.retained_basis, fit.x)
    final_theta = SrixTheta4.from_log_coordinates(final_eta)
    final_fields, final_residual = evaluate(fit.x)
    return {
        "name": name,
        "z4_initial": 0.0,
        "z_initial": initial_z.tolist(),
        "z_final": fit.x.tolist(),
        "eta_final": final_eta.tolist(),
        "initial": SrixTheta4.from_log_coordinates(
            eta_from_reduced_coordinates(eta_ref, basis.retained_basis, initial_z)
        ).as_runtime_overrides(),
        "identified": final_theta.as_runtime_overrides(),
        "cost": {
            "initial_whitened_rms": float(np.sqrt(np.mean(initial_residual**2))),
            "final_whitened_rms": float(np.sqrt(np.mean(final_residual**2))),
            "initial_displacement_rms_mm": float(
                np.sqrt(np.mean((_vector(initial_fields, scored, target)) ** 2))
            ),
            "final_displacement_rms_mm": float(
                np.sqrt(np.mean((_vector(final_fields, scored, target)) ** 2))
            ),
        },
        "optimizer": {
            "success": bool(fit.success),
            "message": str(fit.message),
            "nfev": int(fit.nfev),
            "njev": int(fit.njev or 0),
            "seconds": time.perf_counter() - started,
            "forward_evaluations": len(forwards),
            "jacobian_evaluations": len(jacobian_records),
        },
        "forward_records": forwards,
        "jacobian_records": jacobian_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    measured_macro, angles, provenance = _load_inputs(CROP)
    path = _make_path(measured_macro, 4)
    scored = tuple(4 * index for index in range(1, 9))
    # The scored endpoints are the path steps 4,8,...,32; the path boundaries
    # are the DIC displacement targets at those macro endpoints and provide the
    # full index layout expected by the shared residual helper.
    target = [np.asarray(step.boundary, dtype=np.float64).copy() for step in path]
    prior = _theta_from_preset()
    eta_ref = prior.log_coordinates()
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )
    factory = _factory(angles, library, args.threads)

    prior_fields, prior_timing = _forward(prior, path, angles, library, args.threads)
    prior_residual = _vector(prior_fields, scored, target) / PIXEL_NOISE_MM
    prior_matrix = _matrix(prior_fields, scored, angles, prior, factory, library, args.threads)
    basis = svd_parameter_basis(prior_matrix, fixed_rank=3)
    starts = []
    for name, values in START_Z.items():
        starts.append(
            _run_start(
                name,
                np.asarray(values, dtype=np.float64),
                eta_ref,
                basis,
                target,
                path,
                scored,
                angles,
                library,
                args.threads,
                args.max_nfev,
            )
        )
    best = min(starts, key=lambda item: item["cost"]["final_whitened_rms"])
    best_theta = SrixTheta4(
        **{  # type: ignore[arg-type]
            "tau0_mpa": best["identified"]["tau0_mpa"],
            "r_mpa": best["identified"]["R_mpa"],
            "q_mpa": best["identified"]["Q_mpa"],
            "b": best["identified"]["b"],
        }
    )
    best_fields, _ = _forward(best_theta, path, angles, library, args.threads)
    final_matrix = _matrix(best_fields, scored, angles, best_theta, factory, library, args.threads)
    final_basis = svd_parameter_basis(final_matrix, fixed_rank=3)
    report = {
        "schema_version": 1,
        "method": "P43 experimental direct FEMU rank-three SVD identification M20",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "crop": list(CROP),
        "mesh": list(angles.shape[:2]),
        "path_steps": len(path),
        "scored_steps": list(scored),
        "observation_profile": "measured displacement identity plus scalar DIC whitening",
        "dic_uncertainty_mm": PIXEL_NOISE_MM,
        "parameter_preset": DEFAULT_PARAMETER_SET,
        "shadow_fd_step": H,
        "prior": prior.as_runtime_overrides(),
        "provenance": provenance,
        "prior_forward": prior_timing,
        "prior_whitened_rms": float(np.sqrt(np.mean(prior_residual**2))),
        "prior_displacement_rms_mm": float(
            np.sqrt(np.mean((_vector(prior_fields, scored, target)) ** 2))
        ),
        "prior_svd": {
            "singular_values": basis.singular_values.tolist(),
            "normalized_singular_values": basis.normalized_singular_values.tolist(),
            "right_singular_vectors": basis.right_singular_vectors.tolist(),
            "retained_basis": basis.retained_basis.tolist(),
            "discarded_basis": basis.discarded_basis.tolist(),
        },
        "starts": starts,
        "best_start": best["name"],
        "final_svd": {
            "singular_values": final_basis.singular_values.tolist(),
            "normalized_singular_values": final_basis.normalized_singular_values.tolist(),
            "right_singular_vectors": final_basis.right_singular_vectors.tolist(),
            "retained_basis": final_basis.retained_basis.tolist(),
        },
        "gate": {
            "all_optimizers_converged": all(item["optimizer"]["success"] for item in starts),
            "all_costs_decreased": all(
                item["cost"]["final_whitened_rms"] < item["cost"]["initial_whitened_rms"]
                for item in starts
            ),
            "m100_authorized": False,
        },
        "claims": {
            "experimental_m20_completed": True,
            "experimental_m100_authorized": False,
            "parameters_identified": False,
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
