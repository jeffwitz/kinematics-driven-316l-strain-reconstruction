#!/usr/bin/env python3
"""Qualify SRIX-REGM after the measured DIC transfer and displacement noise."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from fem_inhouse.identification.dic_whitening import DICSpectralTransfer, DICSpectralWhitener
from fem_inhouse.identification.srix_equilibrium_gap import (
    SrixEquilibriumGapProblem,
    SrixTheta4,
)
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.measurement import image_flow_to_canonical
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from scripts.qualify_srix_regm_twin import (
    PIXEL_SIZE_MM,
    _material_factory,
    _point_elasticity,
    _theta_from_preset,
)

ROOT = Path(__file__).resolve().parents[1]
TWIN = ROOT / "validation/reference_data/srix_regm_twin_v1"
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
NOISE = (
    ROOT
    / "validation/reference_data/dic_uncertainty_propagation_p0043_v1"
    / "centred_repeat_flow_pixels.npy"
)


class _Identity:
    def apply(self, values: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(values, dtype=np.float64)


class _WrapFreeTransfer:
    def __init__(self, transfer: DICSpectralTransfer) -> None:
        self.transfer = transfer

    def apply(self, values: ArrayLike) -> NDArray[np.float64]:
        return self.transfer.apply_without_wrap(values)

    def adjoint(self, values: ArrayLike) -> NDArray[np.float64]:
        return self.transfer.adjoint_without_wrap(values)


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _operator(
    grid: StructuredGrid2D,
    orientations: NDArray[np.float64],
    transfer: Any,
    whitener: Any,
) -> TensorPlasticObservabilityOperator:
    return TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
        point_elasticity=_point_elasticity(orientations),
        transfer=transfer,
        whitener=whitener,
    )


def _sample_noise(
    noise: NDArray[np.float64], state_count: int, node_shape: tuple[int, int]
) -> NDArray[np.float64]:
    rng = np.random.default_rng(20260823)
    result = np.zeros((state_count, *node_shape, 2), dtype=np.float64)
    for state in range(1, state_count):
        x0 = int(rng.integers(0, noise.shape[0] - node_shape[0] + 1))
        y0 = int(rng.integers(0, noise.shape[1] - node_shape[1] + 1))
        result[state] = noise[x0 : x0 + node_shape[0], y0 : y0 + node_shape[1]]
    return result


def _record(evaluation: Any) -> dict[str, Any]:
    return {
        "theta": evaluation.theta.as_runtime_overrides(),
        "cost": evaluation.cost,
        "residual_rms": evaluation.residual_rms,
        "timing": asdict(evaluation.timing),
        "backend_timing": dict(evaluation.backend_timing),
    }


def _run_level(
    *,
    name: str,
    history: NDArray[np.float64],
    time_increments: NDArray[np.float64],
    scored_states: tuple[int, ...],
    orientations: NDArray[np.float64],
    transfer: Any,
    whitener: Any,
    library: str,
    threads: int,
    maximum_evaluations: int,
) -> dict[str, Any]:
    pixels = history.shape[1] - 1
    grid = StructuredGrid2D(
        pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels
    )
    problem = SrixEquilibriumGapProblem(
        operator=_operator(grid, orientations, transfer, whitener),
        displacement_history=history,
        state_indices=tuple(range(1, len(history))),
        scored_states=set(scored_states),
        material_factory=_material_factory(
            pixels=pixels, orientations=orientations, library=library, threads=threads
        ),
        time_increments=time_increments,
        debug=False,
    )
    truth_theta = _theta_from_preset()
    initial_theta = SrixTheta4(
        tau0_mpa=1.25 * truth_theta.tau0_mpa,
        r_mpa=0.80 * truth_theta.r_mpa,
        q_mpa=1.30 * truth_theta.q_mpa,
        b=0.75 * truth_theta.b,
    )
    truth = problem.evaluate(truth_theta)
    initial = problem.evaluate(initial_theta)
    jacobian = problem.jacobian_fd(truth_theta.log_coordinates(), relative_step=3.0e-3)
    svd = problem.sensitivity_svd(jacobian, relative_threshold=1.0e-6)
    scale = max(initial.residual_rms, np.finfo(float).tiny)

    def residual(eta: NDArray[np.float64]) -> NDArray[np.float64]:
        return problem.residual_vector(SrixTheta4.from_log_coordinates(eta)) / scale

    def derivative(eta: NDArray[np.float64]) -> NDArray[np.float64]:
        return problem.jacobian_fd(eta, relative_step=3.0e-3) / scale

    true_eta = truth_theta.log_coordinates()
    started = time.perf_counter()
    fit = least_squares(
        residual,
        initial_theta.log_coordinates(),
        jac=derivative,
        bounds=(true_eta - np.log(4.0), true_eta + np.log(4.0)),
        max_nfev=maximum_evaluations,
        x_scale="jac",
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=None,
    )
    fit_seconds = time.perf_counter() - started
    identified_theta = SrixTheta4.from_log_coordinates(fit.x)
    identified = problem.evaluate(identified_theta)
    retained = svd.right_singular_vectors[:, : svd.numerical_rank]
    projected_error = (
        0.0
        if svd.numerical_rank == 0
        else float(np.linalg.norm(retained.T @ (fit.x - true_eta)) / np.sqrt(svd.numerical_rank))
    )
    print(
        f"{name}: truth={truth.residual_rms:.4e} initial={initial.residual_rms:.4e} "
        f"identified={identified.residual_rms:.4e} rank={svd.numerical_rank}",
        flush=True,
    )
    return {
        "name": name,
        "evaluations": {
            "truth": _record(truth),
            "initial": _record(initial),
            "identified": _record(identified),
        },
        "parameters_identified": identified_theta.as_runtime_overrides(),
        "identifiable_log_error_rms": projected_error,
        "sensitivity": {
            "singular_values": svd.singular_values.tolist(),
            "normalized_singular_values": svd.normalized_singular_values.tolist(),
            "right_singular_vectors": svd.right_singular_vectors.tolist(),
            "numerical_rank": svd.numerical_rank,
            "condition_number": svd.condition_number,
        },
        "optimizer": {
            "success": bool(fit.success),
            "message": str(fit.message),
            "nfev": int(fit.nfev),
            "njev": int(fit.njev or 0),
            "seconds": fit_seconds,
            "residual_scale": scale,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--maximum-evaluations", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation/reference_data/srix_regm_transfer_noise_v1",
    )
    arguments = parser.parse_args()
    output = arguments.output if arguments.output.is_absolute() else ROOT / arguments.output
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    git_sha = _git("rev-parse HEAD")
    git_dirty = bool(_git("status --porcelain"))
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )
    fields = np.load(TWIN / "fields.npz", allow_pickle=False)
    raw_history = np.asarray(fields["displacement_history"], dtype=np.float64)
    orientations = np.asarray(fields["orientations_deg"], dtype=np.float64)
    twin_report = json.loads((TWIN / "report.json").read_text(encoding="utf-8"))
    time_increments = np.asarray(twin_report["time_increments"], dtype=np.float64)
    scored_states = tuple(map(int, twin_report["states_scored"]))
    spatial = DICSpectralTransfer.from_sinusoidal_csv(TRANSFER)
    transfer = _WrapFreeTransfer(spatial)
    transferred = np.asarray([transfer.apply(state) for state in raw_history])

    noise_pixels = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    noise_mm = image_flow_to_canonical(
        np.asarray(noise_pixels[:512, :512]), pixel_size_mm=PIXEL_SIZE_MM
    )
    support = np.ones((*raw_history.shape[1:3], 2), dtype=np.float64)
    support[[0, -1], :, :] = 0.0
    support[:, [0, -1], :] = 0.0
    whitener = DICSpectralWhitener.from_stationary_noise_field(
        noise_mm,
        target_shape=raw_history.shape[1:3],
        sample_count=256,
        seed=42,
        remove_spatial_mean=False,
        support_mask=support,
    )
    noisy = transferred + _sample_noise(noise_mm, len(transferred), raw_history.shape[1:3])
    levels = [
        _run_level(
            name="T1_transfer",
            history=transferred,
            time_increments=time_increments,
            scored_states=scored_states,
            orientations=orientations,
            transfer=transfer,
            whitener=_Identity(),
            library=library,
            threads=arguments.threads,
            maximum_evaluations=arguments.maximum_evaluations,
        ),
        _run_level(
            name="T2_transfer_noise",
            history=noisy,
            time_increments=time_increments,
            scored_states=scored_states,
            orientations=orientations,
            transfer=transfer,
            whitener=whitener,
            library=library,
            threads=arguments.threads,
            maximum_evaluations=arguments.maximum_evaluations,
        ),
    ]
    report = {
        "schema_version": 1,
        "method": "SRIX-REGM transfer/noise twin",
        "git_sha": git_sha,
        "dirty": git_dirty,
        "machine": platform.node(),
        "source_twin": str(TWIN.relative_to(ROOT)),
        "transfer_csv": str(TRANSFER.relative_to(ROOT)),
        "noise_source": str(NOISE.relative_to(ROOT)),
        "transfer_mode": "affine_preserving_without_wrap",
        "noise_seed": 20260823,
        "whitener_seed": 42,
        "levels": levels,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    exact = twin_report["sensitivity"]["singular_values"]
    series = [("T0 exact", exact)]
    series.extend(
        (row["name"], row["sensitivity"]["singular_values"]) for row in levels
    )
    for label, values in series:
        normalized = np.asarray(values) / float(values[0])
        axes[0].semilogy(np.arange(1, 5), normalized, marker="o", label=label)
    axes[0].set(xlabel="singular direction", ylabel="normalized singular value")
    axes[0].legend()
    names = ("tau0", "R", "Q", "b")
    x = np.arange(4)
    width = 0.25
    axes[1].bar(x - width, _theta_from_preset().as_array(), width, label="truth")
    for offset, row in enumerate(levels):
        theta = SrixTheta4(**{
            "tau0_mpa": row["parameters_identified"]["tau0_mpa"],
            "r_mpa": row["parameters_identified"]["R_mpa"],
            "q_mpa": row["parameters_identified"]["Q_mpa"],
            "b": row["parameters_identified"]["b"],
        })
        axes[1].bar(x + offset * width, theta.as_array(), width, label=row["name"])
    axes[1].set_xticks(x, names)
    axes[1].set_yscale("log")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "srix_regm_transfer_noise.png", dpi=180)
    figure.savefig(output / "srix_regm_transfer_noise.pdf")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
