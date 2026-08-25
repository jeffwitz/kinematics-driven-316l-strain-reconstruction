#!/usr/bin/env python3
"""Qualify fixed-rank SVD coordinates and profile the weak M20 direction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET
from fem_inhouse.identification.srix_equilibrium_gap import SrixTheta4
from fem_inhouse.identification.svd_parameter_basis import (
    eta_from_reduced_coordinates,
    project_eta_to_basis,
    reduced_coordinates_from_eta,
    svd_parameter_basis,
)
from scripts.qualify_srix_femu_direct_sensitivity import _direct_jacobian
from scripts.qualify_srix_p0043_synthetic_multistart import START_FACTORS, _theta_from_factors
from scripts.qualify_srix_p0043_synthetic_smoke import (
    ROOT,
    H,
    _factory,
    _forward,
    _git,
    _Identity,
    _load_inputs,
    _make_path,
    _vector,
)
from scripts.qualify_srix_regm_twin import _theta_from_preset

DEFAULT_OUTPUT = ROOT / "validation/reference_data/p0043_synthetic_svd_reparameterization_v1"
M20_FIELDS = ROOT / "validation/reference_data/p0043_synthetic_identification_v1/fields.npz"
CROP = (1610, 1630, 1075, 1095)


def _direct_matrix(
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
    return matrix


def _run_c1(
    name: str,
    raw_initial: SrixTheta4,
    truth: SrixTheta4,
    eta_ref: np.ndarray,
    basis: Any,
    target: list[np.ndarray],
    path: list[Any],
    angles: np.ndarray,
    library: str,
    threads: int,
    max_nfev: int,
) -> dict[str, Any]:
    scored = tuple(4 * index for index in range(1, 9))
    factory = _factory(angles, library, threads)
    cache: dict[bytes, tuple[list[Any], np.ndarray]] = {}
    forwards: list[dict[str, Any]] = []
    jacobians = 0
    raw_eta = raw_initial.log_coordinates()
    initial_eta = project_eta_to_basis(raw_eta, eta_ref, basis.retained_basis)
    initial_z = reduced_coordinates_from_eta(initial_eta, eta_ref, basis.retained_basis)

    def evaluate_z(z: np.ndarray) -> tuple[list[Any], np.ndarray]:
        eta = eta_from_reduced_coordinates(eta_ref, basis.retained_basis, z)
        key = np.asarray(eta, dtype=np.float64).tobytes()
        if key not in cache:
            theta = SrixTheta4.from_log_coordinates(eta)
            fields, timing = _forward(theta, path, angles, library, threads)
            residual = _vector(fields, scored, target)
            cache[key] = (fields, residual)
            forwards.append({"theta": theta.as_runtime_overrides(), **timing})
        return cache[key]

    initial_fields, initial_residual = evaluate_z(initial_z)
    scale = max(float(np.linalg.norm(initial_residual)), 1.0e-30)

    def residual(z: np.ndarray) -> np.ndarray:
        return evaluate_z(z)[1] / scale

    def jacobian(z: np.ndarray) -> np.ndarray:
        nonlocal jacobians
        fields, _ = evaluate_z(z)
        eta = eta_from_reduced_coordinates(eta_ref, basis.retained_basis, z)
        matrix = _direct_matrix(
            fields, scored, angles, SrixTheta4.from_log_coordinates(eta), factory, library, threads
        )
        jacobians += 1
        return matrix @ basis.retained_basis / scale

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
    _final_fields, final_residual = evaluate_z(fit.x)
    projection = basis.retained_basis @ (fit.x - np.zeros(3))
    return {
        "name": name,
        "raw_initial_eta": raw_eta.tolist(),
        "projected_initial_eta": initial_eta.tolist(),
        "initial_z": initial_z.tolist(),
        "final_z": fit.x.tolist(),
        "final_eta": final_eta.tolist(),
        "initial": raw_initial.as_runtime_overrides(),
        "projected_initial": SrixTheta4.from_log_coordinates(initial_eta).as_runtime_overrides(),
        "identified": final_theta.as_runtime_overrides(),
        "raw_initial_error_norm": float(np.linalg.norm(raw_eta - eta_ref)),
        "projected_initial_error_norm": float(np.linalg.norm(initial_eta - eta_ref)),
        "final_rank3_error_norm": float(np.linalg.norm(projection)),
        "discarded_final_error": float(np.dot(basis.discarded_basis[:, 0], final_eta - eta_ref)),
        "cost": {
            "initial_rms": float(np.sqrt(np.mean(_vector(initial_fields, scored, target) ** 2))),
            "identified_rms": float(np.sqrt(np.mean(final_residual**2))),
        },
        "optimizer": {
            "success": bool(fit.success),
            "message": str(fit.message),
            "nfev": int(fit.nfev),
            "njev": int(fit.njev or 0),
            "seconds": time.perf_counter() - started,
            "forward_evaluations": len(forwards),
            "jacobian_evaluations": jacobians,
        },
        "forward_records": forwards,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    history, angles, provenance = _load_inputs(CROP)
    path = _make_path(history, 4)
    scored = tuple(4 * index for index in range(1, 9))
    truth = _theta_from_preset()
    eta_ref = truth.log_coordinates()
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )
    truth_fields, truth_timing = _forward(truth, path, angles, library, args.threads)
    target = [np.asarray(field.displacement).copy() for field in truth_fields]
    reference_matrix = np.asarray(
        np.load(M20_FIELDS, allow_pickle=False)["jacobian_truth"], dtype=np.float64
    )
    basis = svd_parameter_basis(reference_matrix, fixed_rank=3)
    c1 = []
    for name, factors in START_FACTORS.items():
        c1.append(
            _run_c1(
                name,
                _theta_from_factors(truth, factors),
                truth,
                eta_ref,
                basis,
                target,
                path,
                angles,
                library,
                args.threads,
                args.max_nfev,
            )
        )
    best = min(c1, key=lambda item: item["cost"]["identified_rms"])
    best_z = np.asarray(best["final_z"], dtype=float)
    weak = basis.discarded_basis[:, 0]
    profile = []
    for z4 in (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30):
        eta = eta_from_reduced_coordinates(eta_ref, basis.retained_basis, best_z) + weak * z4
        fields, _ = _forward(
            SrixTheta4.from_log_coordinates(eta), path, angles, library, args.threads
        )
        residual = _vector(fields, scored, target)
        profile.append(
            {
                "z4": z4,
                "theta": SrixTheta4.from_log_coordinates(eta).as_runtime_overrides(),
                "rms": float(np.sqrt(np.mean(residual**2))),
            }
        )
    qb = np.array([0.0, 0.0, 1.0, -1.0]) / np.sqrt(2.0)
    qpb = np.array([0.0, 0.0, 1.0, 1.0]) / np.sqrt(2.0)
    report = {
        "schema_version": 1,
        "method": "P43 M20 fixed-rank SVD reparameterization and weak-mode profile",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "crop": list(CROP),
        "mesh": list(angles.shape[:2]),
        "path_steps": len(path),
        "scored_steps": list(scored),
        "parameter_preset": DEFAULT_PARAMETER_SET,
        "shadow_fd_step": H,
        "truth": truth.as_runtime_overrides(),
        "eta_reference": eta_ref.tolist(),
        "provenance": provenance,
        "truth_forward": truth_timing,
        "basis": {
            "fixed_rank": basis.effective_rank,
            "singular_values": basis.singular_values.tolist(),
            "normalized_singular_values": basis.normalized_singular_values.tolist(),
            "right_singular_vectors": basis.right_singular_vectors.tolist(),
            "retained_basis": basis.retained_basis.tolist(),
            "discarded_basis": basis.discarded_basis.tolist(),
            "alignment_v4_log_q_minus_log_b": float(abs(np.dot(weak, qb))),
            "alignment_v3_log_q_plus_log_b": float(abs(np.dot(basis.retained_basis[:, 2], qpb))),
        },
        "c1": c1,
        "c3": {"best_start": best["name"], "profile_reoptimized": False, "points": profile},
        "claims": {
            "c1_completed": True,
            "c3_completed": True,
            "synthetic_rank3_qualified": all(item["optimizer"]["success"] for item in c1),
            "weak_direction_profiled": True,
            "experimental_p43_authorized": False,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (output / "weak_mode_profile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("z4", "rms", "tau0_mpa", "R_mpa", "Q_mpa", "b"))
        writer.writeheader()
        for item in profile:
            writer.writerow({"z4": item["z4"], "rms": item["rms"], **item["theta"]})
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    axes[0].semilogy([item["z4"] for item in profile], [item["rms"] for item in profile], "o-")
    axes[0].set(xlabel="z4", ylabel="RMS residual")
    axes[1].plot(
        [item["z4"] for item in profile],
        [item["theta"]["Q_mpa"] for item in profile],
        "o-",
        label="Q",
    )
    axes[1].plot(
        [item["z4"] for item in profile], [item["theta"]["b"] for item in profile], "o-", label="b"
    )
    axes[1].set(xlabel="z4", ylabel="parameter", title="weak SVD direction")
    axes[1].legend()
    figure.savefig(output / "weak_mode_profile.png", dpi=180)
    plt.close(figure)
    print(
        json.dumps(
            {
                "claims": report["claims"],
                "c1_rms": [item["cost"]["identified_rms"] for item in c1],
                "c3": profile,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
