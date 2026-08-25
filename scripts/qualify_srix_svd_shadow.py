#!/usr/bin/env python3
"""Qualify projected SRIX shadow sensitivities on the P43 M20 smoke."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from fem_inhouse.core.plane_stress_material import evaluate_in_plane_response
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

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "validation/reference_data/p0043_global_srix_observability_v1"
DEFAULT_OUTPUT = ROOT / "validation/reference_data/srix_svd_shadow_qualification_v1"


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _solve_tangent(
    grid: StructuredGrid2D,
    kinematics: TwoSubcellDiagnostic2D,
    tangent: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    size = rhs.size

    def action(vector: np.ndarray) -> np.ndarray:
        nodal = unpack_interior(vector, grid)
        strain = kinematics.strain_samples(nodal)
        stress = np.einsum("xyqij,xyqj->xyqi", tangent, strain)
        return pack_interior(kinematics.divergence_from_sample_stress(stress))

    solution, info = gmres(
        LinearOperator((size, size), matvec=action, dtype=np.float64),
        np.asarray(rhs, dtype=np.float64), rtol=1.0e-10, atol=0.0,
        restart=50, maxiter=400, callback_type="pr_norm",
    )
    if info != 0:
        raise RuntimeError(f"projected tangent GMRES failed with info={info}")
    return np.asarray(solution, dtype=np.float64)


def _step_sizes(singular: np.ndarray, rank: int) -> np.ndarray:
    s = np.asarray(singular, dtype=np.float64)
    proposal = 1.0e-3 * s[0] / s[:rank]
    return np.clip(proposal, 5.0e-4, 5.0e-3)


def _direct_shadow(
    *,
    fields: list[Any],
    basis: np.ndarray,
    eta: np.ndarray,
    step_sizes: np.ndarray,
    angles: np.ndarray,
    scored: tuple[int, ...],
    library: str,
    threads: int,
    element_order: str = "C",
) -> tuple[np.ndarray, dict[str, Any]]:
    pixels = fields[0].displacement.shape[0] - 1
    grid = StructuredGrid2D(
        pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels
    )
    kinematics = TwoSubcellDiagnostic2D(grid)
    factory = _factory(angles, library, threads, element_order)
    shadows: list[tuple[Any, Any]] = []
    for mode, step in enumerate(step_sizes):
        plus_eta = eta + step * basis[:, mode]
        minus_eta = eta - step * basis[:, mode]
        shadows.append((
            factory(SrixTheta9.from_log_coordinates(plus_eta).as_runtime_overrides()),
            factory(SrixTheta9.from_log_coordinates(minus_eta).as_runtime_overrides()),
        ))

    scored_vectors: list[list[np.ndarray]] = [[] for _ in range(basis.shape[1])]
    gmres_solves = 0
    started = time.perf_counter()
    for state_index, accepted in enumerate(fields, start=1):
        base_strain = np.asarray(accepted.sample_strain, dtype=np.float64)
        tangent = np.asarray(accepted.algorithmic_tangent_in_plane_mpa, dtype=np.float64)
        forcings: list[np.ndarray] = []
        for mode, (plus, minus) in enumerate(shadows):
            step = float(step_sizes[mode])
            try:
                plus_trial = evaluate_in_plane_response(
                    plus, base_strain.reshape(-1, 3),
                    time_increment=accepted.time_increment,
                    response_level="tangent", consistent_tangent=True,
                )
                minus_trial = evaluate_in_plane_response(
                    minus, base_strain.reshape(-1, 3),
                    time_increment=accepted.time_increment,
                    response_level="tangent", consistent_tangent=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"shadow fixed-strain failure at increment {state_index}, "
                    f"mode {mode}, step {step}"
                ) from exc
            difference = (
                np.asarray(plus_trial.stress_in_plane_mpa)
                - np.asarray(minus_trial.stress_in_plane_mpa)
            ).reshape(
                *grid.pixel_shape, 2, 3, order=element_order
            ) / (2.0 * step)
            forcings.append(
                -pack_interior(kinematics.divergence_from_sample_stress(difference))
            )
            plus.revert()
            minus.revert()

        sensitivities: list[np.ndarray] = []
        for forcing in forcings:
            sensitivities.append(
                unpack_interior(_solve_tangent(grid, kinematics, tangent, forcing), grid)
            )
            gmres_solves += 1

        if state_index in scored:
            for mode, sensitivity in enumerate(sensitivities):
                scored_vectors[mode].append(np.asarray(sensitivity).reshape(-1))

        for mode, (plus, minus) in enumerate(shadows):
            strain_sensitivity = kinematics.strain_samples(sensitivities[mode])
            step = float(step_sizes[mode])
            for shadow, sign in ((plus, 1.0), (minus, -1.0)):
                try:
                    evaluate_in_plane_response(
                        shadow,
                        (base_strain + sign * step * strain_sensitivity).reshape(-1, 3),
                        time_increment=accepted.time_increment,
                        response_level="residual", consistent_tangent=False,
                    )
                    shadow.commit()
                except Exception as exc:
                    raise RuntimeError(
                        f"shadow history-advance failure at increment {state_index}, "
                        f"mode {mode}, sign {sign:+.0f}, step {step}"
                    ) from exc

    matrix = np.column_stack([np.concatenate(values) for values in scored_vectors])
    return matrix, {
        "elapsed_seconds": time.perf_counter() - started,
        "gmres_solves": gmres_solves,
        "shadow_count": 2 * basis.shape[1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=7)
    parser.add_argument("--step-scale", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)

    archive = np.load(SMOKE / "global_observability.npz")
    eta_samples = np.asarray(archive["eta_samples"], dtype=np.float64)
    fd_jacobians = np.asarray(archive["jacobians"], dtype=np.float64)
    eigenvectors = np.asarray(archive["eigenvectors"], dtype=np.float64)
    eigenvalues = np.asarray(archive["eigenvalues"], dtype=np.float64)
    if args.rank <= 0 or args.rank > eigenvectors.shape[1]:
        raise ValueError("rank outside the archived nine-parameter basis")
    basis = eigenvectors[:, :args.rank]
    steps = _step_sizes(np.sqrt(np.maximum(eigenvalues, 0.0)), args.rank)
    if not np.isfinite(args.step_scale) or args.step_scale <= 0.0:
        raise ValueError("step scale must be finite and positive")
    steps *= args.step_scale

    measured_macro, angles, provenance = _load_inputs(CROP)
    path = _make_path(measured_macro, 4)
    scored = tuple(4 * index for index in range(1, 9))
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )
    records: list[dict[str, Any]] = []
    matrices: list[np.ndarray] = []
    for sample_index, eta in enumerate(eta_samples):
        theta = SrixTheta9.from_log_coordinates(eta)
        fields, forward_timing = _forward(theta, path, angles, library, args.threads)
        started = time.perf_counter()
        shadow, timing = _direct_shadow(
            fields=fields, basis=basis, eta=eta, step_sizes=steps,
            angles=angles, scored=scored, library=library, threads=args.threads,
        )
        target = fd_jacobians[sample_index] @ basis
        errors = [
            float(np.linalg.norm(shadow[:, i] - target[:, i]) / np.linalg.norm(target[:, i]))
            for i in range(args.rank)
        ]
        cosines = [
            float(np.dot(shadow[:, i], target[:, i]) /
                  (np.linalg.norm(shadow[:, i]) * np.linalg.norm(target[:, i])))
            for i in range(args.rank)
        ]
        matrices.append(shadow)
        records.append({
            "sample_index": sample_index,
            "step_sizes": steps.tolist(),
            "column_relative_errors": errors,
            "column_cosines": cosines,
            "shadow_singular_values": np.linalg.svd(shadow, compute_uv=False).tolist(),
            "fd_projected_singular_values": np.linalg.svd(target, compute_uv=False).tolist(),
            "forward_timing": forward_timing,
            "shadow_timing": timing,
            "total_seconds": time.perf_counter() - started,
        })

    report = {
        "schema_version": 1,
        "method": "projected direct SRIX shadows in global SVD basis",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "rank": args.rank,
        "parameterization": "admissible nine-parameter logarithmic coordinates",
        "provenance": provenance,
        "step_policy": {
            "h_ref": 1.0e-3, "h_min": 5.0e-4, "h_max": 5.0e-3,
            "step_scale": args.step_scale,
        },
        "step_sizes": steps.tolist(),
        "records": records,
        "claims": {"svd_shadow_qualified": False, "experimental_identification_authorized": False},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output / "shadow_vs_fd.npz", basis=basis, step_sizes=steps,
        shadow=np.asarray(matrices), fd_projected=np.asarray([j @ basis for j in fd_jacobians]),
        eigenvalues=eigenvalues,
    )
    print([record["column_relative_errors"] for record in records], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
