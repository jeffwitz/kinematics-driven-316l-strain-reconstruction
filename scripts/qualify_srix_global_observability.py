#!/usr/bin/env python3
"""Probe global nine-parameter SRIX displacement observability on P43 M20."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles
from scipy.stats import qmc

from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET, get_parameter_set
from fem_inhouse.identification.srix_parameter_coordinates import (
    SRIX9_NAMES,
    SrixTheta9,
)
from scripts.qualify_srix_p0043_synthetic_smoke import (
    CROP,
    _factory,
    _forward,
    _git,
    _load_inputs,
    _make_path,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation/reference_data/p0043_global_srix_observability_v1"
H = 1.0e-3


def _output(fields: list[object], scored: tuple[int, ...]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(fields[index - 1].displacement, dtype=np.float64).reshape(-1)
         for index in scored]
    )


def _bounds(eta_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = eta_ref.copy()
    high = eta_ref.copy()
    low[:3] = eta_ref[:3] + np.log(0.85)
    high[:3] = eta_ref[:3] + np.log(1.15)
    physical = np.asarray(
        [[25.0, 60.0], [8.0, 35.0], [4.0, 20.0], [1.0, 6.0],
         [10000.0, 80000.0], [500.0, 3000.0]], dtype=np.float64
    )
    for index, (lo, hi) in enumerate(physical, start=3):
        low[index] = np.log(lo)
        high[index] = np.log(hi)
    return low, high


def _sample_eta(eta_ref: np.ndarray, count: int, seed: int) -> np.ndarray:
    low, high = _bounds(eta_ref)
    if count <= 0:
        raise ValueError("sample count must be positive")
    engine = qmc.Sobol(d=9, scramble=True, seed=seed)
    points = engine.random(count)
    samples = low + points * (high - low)
    return np.vstack((eta_ref, samples))


def _jacobian(
    eta: np.ndarray,
    *,
    path: list[object],
    scored: tuple[int, ...],
    angles: np.ndarray,
    library: str,
    threads: int,
    h: float,
) -> tuple[np.ndarray, dict[str, float]]:
    timings: dict[str, float] = {}
    plus_minus: list[np.ndarray] = []
    for direction in range(9):
        started = time.perf_counter()
        columns = []
        for sign in (-1.0, 1.0):
            perturbed = eta.copy()
            perturbed[direction] += sign * h
            theta = SrixTheta9.from_log_coordinates(perturbed)
            fields, _ = _forward(theta, path, angles, library, threads)
            columns.append(_output(fields, scored))
        plus_minus.append((columns[1] - columns[0]) / (2.0 * h))
        timings[f"parameter_{direction}_seconds"] = time.perf_counter() - started
    return np.column_stack(plus_minus), timings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)

    measured_macro, angles, provenance = _load_inputs(CROP)
    path = _make_path(measured_macro, 4)
    scored = tuple(4 * index for index in range(1, 9))
    prior = SrixTheta9.from_parameter_set(get_parameter_set(DEFAULT_PARAMETER_SET))
    eta_ref = prior.log_coordinates()
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )
    samples = _sample_eta(eta_ref, args.sample_count, args.seed)
    jacobians: list[np.ndarray] = []
    local_reports: list[dict[str, object]] = []
    for sample_index, eta in enumerate(samples):
        theta = SrixTheta9.from_log_coordinates(eta)
        started = time.perf_counter()
        fields, forward_timing = _forward(theta, path, angles, library, args.threads)
        y = _output(fields, scored)
        jacobian, fd_timing = _jacobian(
            eta, path=path, scored=scored, angles=angles,
            library=library, threads=args.threads, h=H,
        )
        singular_values, _, right = np.linalg.svd(jacobian, full_matrices=False)
        jacobians.append(jacobian)
        local_reports.append({
            "sample_index": sample_index,
            "eta": eta.tolist(),
            "parameters": theta.as_runtime_overrides(),
            "output_norm_mm": float(np.linalg.norm(y)),
            "singular_values": singular_values.tolist(),
            "normalized_singular_values": (singular_values / singular_values[0]).tolist(),
            "right_singular_vectors": right.tolist(),
            "forward_timing": forward_timing,
            "fd_timing": fd_timing,
            "total_seconds": time.perf_counter() - started,
        })

    hessian = sum(jacobian.T @ jacobian for jacobian in jacobians)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    normalized = eigenvalues / max(eigenvalues[0], np.finfo(float).tiny)
    rank3_angles = [
        np.degrees(subspace_angles(report_basis, eigenvectors[:, :3]))
        .tolist()
        for report_basis in [np.asarray(item["right_singular_vectors"])[:3].T for item in local_reports]
    ]
    report = {
        "schema_version": 1,
        "method": "global nine-parameter SRIX displacement observability probe",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "crop": list(CROP),
        "path_steps": len(path),
        "scored_steps": list(scored),
        "parameter_names": list(SRIX9_NAMES),
        "parameterization": "log(C11-C12), log(C11+2C12), log(C44), log(tau0), log(R), log(Q), log(b), log(C), log(d)",
        "output_units": "mm",
        "observation_weighting": "none",
        "noise_model_used": False,
        "covariance_used": False,
        "fd_step": H,
        "sample_count_excluding_prior": args.sample_count,
        "seed": args.seed,
        "provenance": provenance,
        "bounds": {"low": _bounds(eta_ref)[0].tolist(), "high": _bounds(eta_ref)[1].tolist()},
        "local_reports": local_reports,
        "global_hessian_eigenvalues": eigenvalues.tolist(),
        "global_normalized_eigenvalues": normalized.tolist(),
        "global_right_eigenvectors": eigenvectors.tolist(),
        "local_rank3_vs_global_angles_deg": rank3_angles,
        "claims": {"parameter_identification": False, "experimental_optimization_authorized": False},
    }
    np.savez_compressed(
        output / "global_observability.npz",
        jacobians=np.asarray(jacobians),
        hessian=hessian,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        eta_samples=samples,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"samples": len(samples), "normalized_eigenvalues": normalized.tolist()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
