#!/usr/bin/env python3
"""Test a sequential one-correction-per-increment SRIX surrogate on M8."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import subspace_angles
from scipy.sparse.linalg import factorized

from fem_inhouse.core.kelvin import stiffness_from_engineering
from fem_inhouse.core.plane_stress_material import evaluate_in_plane_response
from fem_inhouse.identification.dic_whitening import DICSpectralTransfer
from fem_inhouse.identification.srix_equilibrium_gap import SrixTheta4
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
    _assemble_sparse_stiffness,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_ebi import unpack_interior
from scripts.qualify_srix_regm_information_geometry import FD_STEP, _geometry, _plot
from scripts.qualify_srix_regm_transfer_noise import _Identity, _WrapFreeTransfer
from scripts.qualify_srix_regm_twin import (
    PIXEL_SIZE_MM,
    _material_factory,
    _point_elasticity,
    _theta_from_preset,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "validation/reference_data/srix_regm_twin_v1"
FEMU_GEOMETRY = ROOT / "validation/reference_data/srix_regm_information_geometry_v1"
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
DEFAULT_OUTPUT = ROOT / "validation/reference_data/srix_regm_sequential_one_newton_v1"


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _operator(grid: StructuredGrid2D, orientations: np.ndarray, transfer: Any):
    return TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
        point_elasticity=_point_elasticity(orientations),
        transfer=transfer,
        whitener=_Identity(),
    )


def _replay(
    theta: SrixTheta4,
    history: np.ndarray,
    orientations: np.ndarray,
    increments: np.ndarray,
    scored: tuple[int, ...],
    transfer: Any,
    library: str,
    threads: int,
) -> np.ndarray:
    pixels = history.shape[1] - 1
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    operator = _operator(grid, orientations, transfer)
    material = _material_factory(
        pixels=pixels, orientations=orientations, library=library, threads=threads
    )(theta.as_runtime_overrides())
    current = np.asarray(history[0], dtype=np.float64).copy()
    corrections: list[np.ndarray] = []
    for offset, increment in enumerate(increments):
        predictor = current + (history[offset + 1] - history[offset])
        trial = evaluate_in_plane_response(
            material,
            operator.kinematics.strain(predictor).reshape(-1, 3),
            time_increment=float(increment),
            response_level="tangent",
            consistent_tangent=True,
        )
        if trial.tangent_in_plane_mpa is None:
            raise RuntimeError("SRIX did not provide an algorithmic tangent")
        stress = np.asarray(trial.stress_in_plane_mpa, dtype=np.float64).reshape(
            *grid.pixel_shape, 2, 3
        )
        tangent = stiffness_from_engineering(
            np.asarray(trial.tangent_in_plane_mpa, dtype=np.float64).reshape(-1, 3, 3)
        )
        matrix = _assemble_sparse_stiffness(
            grid, operator.kinematics, tangent, operator.quadrature_weight
        )
        correction = unpack_interior(
            -factorized(matrix.tocsc())(operator.weak_equilibrium_residual(stress)), grid
        )
        material.revert()
        accepted = predictor + correction
        evaluate_in_plane_response(
            material,
            operator.kinematics.strain(accepted).reshape(-1, 3),
            time_increment=float(increment),
            response_level="residual",
            consistent_tangent=False,
        )
        material.commit()
        current = accepted
        if offset + 1 in scored:
            corrections.append(np.asarray(transfer.apply(correction), dtype=np.float64).reshape(-1))
    return np.concatenate(corrections)


def _jacobian(
    eta: np.ndarray,
    history: np.ndarray,
    orientations: np.ndarray,
    increments: np.ndarray,
    scored: tuple[int, ...],
    transfer: Any,
    library: str,
    threads: int,
) -> np.ndarray:
    columns = []
    for index in range(4):
        plus = eta.copy()
        minus = eta.copy()
        plus[index] += FD_STEP
        minus[index] -= FD_STEP
        columns.append(
            (
                _replay(
                    SrixTheta4.from_log_coordinates(plus), history, orientations,
                    increments, scored, transfer, library, threads
                )
                - _replay(
                    SrixTheta4.from_log_coordinates(minus), history, orientations,
                    increments, scored, transfer, library, threads
                )
            )
            / (2.0 * FD_STEP)
        )
    return np.column_stack(columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--library", default="build/mfront/src/libBehaviour.so")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    fields = np.load(SOURCE / "fields.npz", mmap_mode="r")
    history = np.asarray(fields["displacement_history"], dtype=np.float64)
    orientations = np.asarray(fields["orientations_deg"], dtype=np.float64)
    report = json.loads((SOURCE / "report.json").read_text())
    increments = np.asarray(report["time_increments"], dtype=np.float64)
    scored = tuple(int(value) for value in report["states_scored"])
    transfer = _WrapFreeTransfer(DICSpectralTransfer.from_sinusoidal_csv(TRANSFER))
    started = time.perf_counter()
    matrix = _jacobian(
        _theta_from_preset().log_coordinates(), history, orientations, increments,
        scored, transfer, args.library, args.threads
    )
    elapsed = time.perf_counter() - started
    geometry = _geometry(matrix)
    geometry["cumulative"] = []
    femu = json.loads((FEMU_GEOMETRY / "report.json").read_text())["geometries"]["FEMU_observed"]
    angles = {}
    for count in (1, 2, 3):
        left = np.asarray(geometry["right_singular_vectors"])[:, :count]
        right = np.asarray(femu["right_singular_vectors"])[:, :count]
        angles[str(count)] = np.degrees(subspace_angles(left, right)).tolist()
    result = {
        "schema_version": 1,
        "method": "sequential one algorithmic-tangent correction per increment",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "source_twin": str(SOURCE.relative_to(ROOT)),
        "reference_femu_geometry": str((FEMU_GEOMETRY / "report.json").relative_to(ROOT)),
        "fd_step_log": FD_STEP,
        "states_scored": list(scored),
        "timing_seconds": elapsed,
        "geometry": geometry,
        "femu_geometry": femu,
        "principal_angles_to_femu_degrees": angles,
        "claims": {"new_global_mechanics": False, "p43_authorized": False},
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(output / "jacobian.npz", SREGM_one_newton=matrix)
    _plot({"SREGM_1Newton": geometry, "FEMU_observed": femu}, output)
    print(geometry["normalized_singular_values"], geometry["condition_number"], flush=True)


if __name__ == "__main__":
    main()
