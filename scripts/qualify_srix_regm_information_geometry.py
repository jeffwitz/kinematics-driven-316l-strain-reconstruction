#!/usr/bin/env python3
"""Compare REGM and complete-FEMU parameter information on the M8 twin."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import subspace_angles

from fem_inhouse.identification.dic_whitening import DICSpectralTransfer
from fem_inhouse.identification.srix_equilibrium_gap import (
    SrixEquilibriumGapProblem,
    SrixTheta4,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from scripts.qualify_srix_regm_transfer_noise import _Identity, _WrapFreeTransfer
from scripts.qualify_srix_regm_twin import (
    PIXEL_SIZE_MM,
    _generate_twin,
    _material_factory,
    _point_elasticity,
    _theta_from_preset,
)
from scripts.qualify_srix_regm_twin import _operator as exact_operator

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "validation/reference_data/srix_regm_twin_v1"
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
DEFAULT_OUTPUT = ROOT / "validation/reference_data/srix_regm_information_geometry_v1"
PARAMETER_NAMES = ("log_tau0", "log_R", "log_Q", "log_b")
FD_STEP = 3.0e-3
RELATIVE_THRESHOLD = 1.0e-6


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _observed_operator(grid: StructuredGrid2D, orientations: np.ndarray, transfer: Any):
    from fem_inhouse.identification.tensor_plastic_observability import (
        TensorPlasticObservabilityOperator,
    )

    return TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
        point_elasticity=_point_elasticity(orientations),
        transfer=transfer,
        whitener=_Identity(),
    )


def _problem(
    history: np.ndarray,
    orientations: np.ndarray,
    transfer: Any,
    increments: np.ndarray,
    scored: tuple[int, ...],
    library: str,
    threads: int,
) -> SrixEquilibriumGapProblem:
    pixels = history.shape[1] - 1
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    operator = (
        exact_operator(grid, orientations)
        if isinstance(transfer, _Identity)
        else _observed_operator(grid, orientations, transfer)
    )
    return SrixEquilibriumGapProblem(
        operator=operator,
        displacement_history=history,
        state_indices=tuple(range(1, len(history))),
        scored_states=set(scored),
        material_factory=_material_factory(
            pixels=pixels, orientations=orientations, library=library, threads=threads
        ),
        time_increments=increments,
        debug=False,
    )


def _femu_residual(
    history: np.ndarray,
    generated_states: tuple[int, ...],
    target_observed: np.ndarray,
    transfer: Any,
) -> np.ndarray:
    values = []
    for generated, target in zip(generated_states, target_observed, strict=True):
        values.append((transfer.apply(history[generated]) - target).reshape(-1))
    return np.concatenate(values)


def _jacobian_fd(
    evaluator: Any,
    eta: np.ndarray,
    step: float,
) -> np.ndarray:
    columns = []
    for index in range(4):
        plus = eta.copy()
        minus = eta.copy()
        plus[index] += step
        minus[index] -= step
        columns.append((evaluator(plus) - evaluator(minus)) / (2.0 * step))
    return np.column_stack(columns)


def _geometry(matrix: np.ndarray) -> dict[str, Any]:
    _, singular, right_transposed = np.linalg.svd(matrix, full_matrices=False)
    normalized = singular / max(float(singular[0]), np.finfo(float).tiny)
    rank = int(np.count_nonzero(normalized > RELATIVE_THRESHOLD))
    fisher = matrix.T @ matrix
    covariance = np.linalg.pinv(fisher, rcond=RELATIVE_THRESHOLD)
    diagonal = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(diagonal, diagonal)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > np.finfo(float).tiny,
    )
    return {
        "singular_values": singular.tolist(),
        "normalized_singular_values": normalized.tolist(),
        "right_singular_vectors": right_transposed.T.tolist(),
        "numerical_rank": rank,
        "condition_number": float(singular[0] / singular[rank - 1]) if rank else float("inf"),
        "fisher": fisher.tolist(),
        "covariance_pseudoinverse": covariance.tolist(),
        "correlation": correlation.tolist(),
    }


def _cumulative_geometry(matrix: np.ndarray, block_size: int) -> list[dict[str, Any]]:
    rows = []
    for count in range(1, matrix.shape[0] // block_size + 1):
        rows.append(_geometry(matrix[: count * block_size]))
    return rows


def _plot(geometries: dict[str, dict[str, Any]], output: Path) -> None:
    labels = tuple(geometries)
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for label in labels:
        values = np.asarray(geometries[label]["normalized_singular_values"])
        axes[0, 0].semilogy(np.arange(1, len(values) + 1), values, marker="o", label=label)
    axes[0, 0].set(xlabel="singular direction", ylabel="sigma / sigma1")
    axes[0, 0].legend()
    for index, label in enumerate(labels):
        vectors = np.asarray(geometries[label]["right_singular_vectors"])
        image = axes[0, index + 1].imshow(
            vectors, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1
        )
        axes[0, index + 1].set_title(label)
        axes[0, index + 1].set_xticks(range(4), PARAMETER_NAMES, rotation=45, ha="right")
        axes[0, index + 1].set_ylabel("right singular vector")
    figure.colorbar(image, ax=axes[0, 1:4], shrink=0.8)
    for label in labels:
        cumulative = geometries[label]["cumulative"]
        minimum = [row["normalized_singular_values"][-1] for row in cumulative]
        axes[1, 0].semilogy(
            np.arange(1, len(minimum) + 1), minimum, marker="o", label=label
        )
    axes[1, 0].set(xlabel="number of scored states", ylabel="smallest normalized singular value")
    axes[1, 0].legend()
    for index, label in enumerate(labels):
        correlation = np.asarray(geometries[label]["correlation"])
        axes[1, index + 1].imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
        axes[1, index + 1].set_title(f"{label} correlation")
        axes[1, index + 1].set_xticks(range(4), PARAMETER_NAMES, rotation=45, ha="right")
        axes[1, index + 1].set_yticks(range(4), PARAMETER_NAMES)
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"srix_regm_information_geometry.{suffix}", dpi=180)
    plt.close(figure)


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
    raw = np.asarray(fields["displacement_history"], dtype=np.float64)
    orientations = np.asarray(fields["orientations_deg"], dtype=np.float64)
    source_report = json.loads((SOURCE / "report.json").read_text())
    increments = np.asarray(source_report["time_increments"], dtype=np.float64)
    scored = tuple(int(value) for value in source_report["states_scored"])
    truth = _theta_from_preset()
    eta = truth.log_coordinates()
    transfer = _WrapFreeTransfer(DICSpectralTransfer.from_sinusoidal_csv(TRANSFER))
    observed = np.asarray([transfer.apply(state) for state in raw])
    target_observed = np.asarray([transfer.apply(raw[index]) for index in scored])

    exact_problem = _problem(
        raw, orientations, _Identity(), increments, scored, args.library, args.threads
    )
    observed_problem = _problem(
        observed, orientations, transfer, increments, scored, args.library, args.threads
    )
    started = time.perf_counter()
    exact_jacobian = exact_problem.jacobian_fd(eta, relative_step=FD_STEP)
    observed_jacobian = observed_problem.jacobian_fd(eta, relative_step=FD_STEP)
    replay_seconds = time.perf_counter() - started

    def forward_residual(coordinates: np.ndarray) -> np.ndarray:
        theta = SrixTheta4.from_log_coordinates(coordinates)
        history, _, generated_scored, _, _ = _generate_twin(
            pixels=raw.shape[1] - 1,
            library=args.library,
            threads=args.threads,
            theta=theta,
        )
        if len(generated_scored) != len(scored):
            raise RuntimeError("perturbed forward solve changed the registered endpoint count")
        return _femu_residual(history, generated_scored, target_observed, transfer)

    started = time.perf_counter()
    forward_truth = forward_residual(eta)
    femu_jacobian = _jacobian_fd(forward_residual, eta, FD_STEP)
    femu_seconds = time.perf_counter() - started

    block_size = int(target_observed[0].size)
    matrices = {
        "REGM_exact": exact_jacobian,
        "REGM_observed": observed_jacobian,
        "FEMU_observed": femu_jacobian,
    }
    geometries: dict[str, dict[str, Any]] = {}
    for label, matrix in matrices.items():
        geometry = _geometry(matrix)
        geometry["cumulative"] = _cumulative_geometry(matrix, block_size)
        geometries[label] = geometry
    angles = {}
    pairs = (
        ("REGM_exact", "REGM_observed"),
        ("REGM_exact", "FEMU_observed"),
        ("REGM_observed", "FEMU_observed"),
    )
    for left, right in pairs:
        left_rank = geometries[left]["numerical_rank"]
        right_rank = geometries[right]["numerical_rank"]
        maximum = min(left_rank, right_rank, 3)
        angles[f"{left}__{right}"] = {}
        for count in range(1, maximum + 1):
            left_vectors = np.asarray(geometries[left]["right_singular_vectors"])[:, :count]
            right_vectors = np.asarray(geometries[right]["right_singular_vectors"])[:, :count]
            angles[f"{left}__{right}"][str(count)] = np.degrees(
                subspace_angles(left_vectors, right_vectors)
            ).tolist()

    report = {
        "schema_version": 1,
        "method": "local information geometry comparison of REGM and observed FEMU",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "source_twin": str(SOURCE.relative_to(ROOT)),
        "transfer_csv": str(TRANSFER.relative_to(ROOT)),
        "parameter_names": PARAMETER_NAMES,
        "fd_step_log": FD_STEP,
        "relative_threshold": RELATIVE_THRESHOLD,
        "states_scored": list(scored),
        "timings": {
            "regm_jacobians_seconds": replay_seconds,
            "femu_jacobian_seconds": femu_seconds,
        },
        "forward_truth_residual_rms": float(np.sqrt(np.mean(forward_truth**2))),
        "geometries": geometries,
        "subspace_principal_angles_degrees": angles,
        "jacobians": {label: matrix.tolist() for label, matrix in matrices.items()},
        "claims": {"new_mechanics_launched": True, "p43_authorized": False},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(output / "jacobians.npz", **matrices)
    _plot(geometries, output)
    print(
        f"REGM Jacobians: {replay_seconds:.2f} s; FEMU Jacobian: {femu_seconds:.2f} s",
        flush=True,
    )
    for label, geometry in geometries.items():
        print(
            label,
            geometry["normalized_singular_values"],
            "rank",
            geometry["numerical_rank"],
            flush=True,
        )


if __name__ == "__main__":
    main()
