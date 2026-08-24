#!/usr/bin/env python3
"""Smoke-test direct SRIX identification on a real-data P43 synthetic crop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch
from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET
from fem_inhouse.identification.srix_equilibrium_gap import SrixTheta4
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_two_state import (
    TwoStateIncrementFields,
    solve_two_state_dirichlet_plane_stress,
)
from fem_inhouse.spectral2d.step_control import LoadPathStep
from scripts.qualify_srix_femu_direct_sensitivity import (
    _direct_jacobian,
    _geometry,
    _oracle_config,
)
from scripts.qualify_srix_regm_twin import PIXEL_SIZE_MM, _theta_from_preset

ROOT = Path(__file__).resolve().parents[1]
HISTORY = (
    ROOT
    / "validation/reference_data/dic_multistep_history_p0043_repaired_v1/repaired_history_mm.npy"
)
HISTORY_REPORT = HISTORY.with_name("report.json")
EBSD = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")
DEFAULT_OUTPUT = ROOT / "validation/reference_data/p0043_synthetic_identification_v1"
CROP = (1610, 1630, 1075, 1095)
SUBDIVISIONS = 4
H = 1.5e-3


class _Identity:
    def apply(self, values: Any) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values: Any) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)


def _git(command: str) -> str:
    return subprocess.run(
        ["git", *command.split()], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_inputs(
    crop: tuple[int, int, int, int] = CROP,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    report = json.loads(HISTORY_REPORT.read_text(encoding="utf-8"))
    x0, x1, y0, y1 = crop
    bx0, _, by0, _ = map(int, report["solve_bounds"])
    source = np.load(HISTORY, mmap_mode="r", allow_pickle=False)
    local = np.asarray(
        source[:, x0 - bx0 : x1 - bx0 + 1, y0 - by0 : y1 - by0 + 1, :],
        dtype=np.float64,
    )
    local -= local[0]
    with h5py.File(EBSD, "r") as handle:
        angles = np.stack(
            [
                np.asarray(handle[f"orientation/{name}"][x0:x1, y0:y1], dtype=np.float64)
                for name in ("phi1", "Phi", "phi2")
            ],
            axis=-1,
        )
    if angles.shape != (x1 - x0, y1 - y0, 3) or not np.isfinite(angles).all():
        raise ValueError("invalid co-registered P43 EBSD crop")
    macro = local[::5]
    if macro.shape[0] != 9:
        raise ValueError("expected repaired P43 states 0,5,...,40")
    return macro, angles, {
        "crop_absolute": list(crop),
        "solve_bounds": report["solve_bounds"],
        "history_sha256": hashlib.sha256(np.ascontiguousarray(macro).tobytes()).hexdigest(),
        "ebsd_sha256": hashlib.sha256(np.ascontiguousarray(angles).tobytes()).hexdigest(),
    }


def _make_path(history: np.ndarray, subdivisions: int) -> list[LoadPathStep]:
    path: list[LoadPathStep] = []
    segments = history.shape[0] - 1
    index = 0
    for segment in range(segments):
        for substep in range(1, subdivisions + 1):
            index += 1
            fraction = substep / subdivisions
            boundary = (1.0 - fraction) * history[segment] + fraction * history[segment + 1]
            start = (segment + (substep - 1) / subdivisions) / segments
            end = (segment + fraction) / segments
            path.append(
                LoadPathStep(
                    index=index,
                    start_fraction=start,
                    end_fraction=end,
                    boundary=np.asarray(boundary, dtype=np.float64).copy(),
                    time_increment=1.0 / (segments * subdivisions),
                )
            )
    return path


def _factory(angles: np.ndarray, library: str, threads: int):
    point_count = 2 * angles.shape[0] * angles.shape[1]

    def create(overrides: dict[str, float]):
        return create_plane_stress_material_batch(
            "mfront-3d-condensed-plane-stress",
            np.ones(point_count),
            np.ones(point_count),
            0.245,
            young_modulus_mpa=205_000.0,
            poisson_ratio=0.30,
            hardening_mode="ludwik",
            plastic_strain_max=0.2,
            plastic_table_points=1_000,
            first_positive_plastic_strain=1.0e-6,
            mfront_library=library,
            mfront_threads=threads,
            mfront_behaviour_id="fcc_forest_rubin_srix",
            local_plane_stress_options={
                "local_condition_check_mode": "on_failure",
                "local_transverse_predictor": "tangent",
            },
            constitutive_options={
                "parameter_set": DEFAULT_PARAMETER_SET,
                "parameters": overrides,
                "crystal_orientation": {
                    "mode": "ebsd",
                    "euler_bunge_deg": angles,
                },
            },
        )

    return create


def _copy_field(value: TwoStateIncrementFields) -> TwoStateIncrementFields:
    return TwoStateIncrementFields(
        increment=value.increment,
        start_fraction=value.start_fraction,
        end_fraction=value.end_fraction,
        time_increment=value.time_increment,
        boundary=np.asarray(value.boundary).copy(),
        displacement=np.asarray(value.displacement).copy(),
        sample_strain=np.asarray(value.sample_strain).copy(),
        stress_in_plane_mpa=np.asarray(value.stress_in_plane_mpa).copy(),
        algorithmic_tangent_in_plane_mpa=np.asarray(value.algorithmic_tangent_in_plane_mpa).copy(),
        plastic_strain_tensor=None,
    )


def _forward(
    theta: SrixTheta4,
    path: list[LoadPathStep],
    angles: np.ndarray,
    library: str,
    threads: int,
) -> tuple[list[TwoStateIncrementFields], dict[str, Any]]:
    pixels = angles.shape[0]
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    material = _factory(angles, library, threads)(theta.as_runtime_overrides())
    history = np.stack([np.zeros_like(path[0].boundary), *[step.boundary for step in path]])
    fields: list[TwoStateIncrementFields] = []

    def observe(value: TwoStateIncrementFields) -> None:
        fields.append(_copy_field(value))

    started = time.perf_counter()
    result = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=material,
        boundary_displacement_history=history,
        config=_oracle_config(),
        load_path_override=path,
        increment_observer=observe,
    )
    if len(fields) != len(path):
        raise RuntimeError("P43 synthetic forward did not preserve its fixed path")
    return fields, {
        "seconds": time.perf_counter() - started,
        "steps": len(fields),
        "verification_residual": result.diagnostics.verification_residual,
        "gmres_iterations": int(result.diagnostics.timings["gmres_iterations"]),
    }


def _vector(
    fields: list[TwoStateIncrementFields],
    scored: tuple[int, ...],
    target: list[np.ndarray],
) -> np.ndarray:
    return np.concatenate(
        [
            (np.asarray(fields[index - 1].displacement) - target[index - 1]).reshape(-1)
            for index in scored
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--maximum-evaluations", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    library = os.environ.get(
        "MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so")
    )
    history, angles, provenance = _load_inputs()
    path = _make_path(history, SUBDIVISIONS)
    scored = tuple(4 * index for index in range(1, 9))
    truth_theta = _theta_from_preset()
    initial_theta = SrixTheta4(
        tau0_mpa=1.05 * truth_theta.tau0_mpa,
        r_mpa=0.95 * truth_theta.r_mpa,
        q_mpa=1.08 * truth_theta.q_mpa,
        b=0.92 * truth_theta.b,
    )
    truth_fields, _truth_timing = _forward(truth_theta, path, angles, library, args.threads)
    target = [np.asarray(field.displacement).copy() for field in truth_fields]
    identity = _Identity()
    factory = _factory(angles, library, args.threads)
    cache: dict[bytes, tuple[list[TwoStateIncrementFields], np.ndarray]] = {}
    forward_records: list[dict[str, Any]] = []

    def evaluate(eta: np.ndarray) -> tuple[list[TwoStateIncrementFields], np.ndarray]:
        key = np.asarray(eta, dtype=np.float64).tobytes()
        if key not in cache:
            theta = SrixTheta4.from_log_coordinates(eta)
            fields, timing = _forward(theta, path, angles, library, args.threads)
            residual = _vector(fields, scored, target)
            cache[key] = (fields, residual)
            forward_records.append({"theta": theta.as_runtime_overrides(), **timing})
        return cache[key]

    _initial_fields, initial_residual = evaluate(initial_theta.log_coordinates())
    scale = max(float(np.linalg.norm(initial_residual)), 1.0e-30)
    jacobian_records: list[dict[str, Any]] = []

    def residual(eta: np.ndarray) -> np.ndarray:
        return evaluate(eta)[1] / scale

    def jacobian(eta: np.ndarray) -> np.ndarray:
        fields, _ = evaluate(eta)
        started = time.perf_counter()
        matrix, timing = _direct_jacobian(
            fields=fields,
            scored=scored,
            orientations=angles,
            theta=SrixTheta4.from_log_coordinates(eta),
            library=library,
            threads=args.threads,
            transfer=identity,
            h=H,
            material_factory=factory,
        )
        jacobian_records.append({"seconds": time.perf_counter() - started, **timing})
        return matrix / scale

    started = time.perf_counter()
    fit = least_squares(
        residual,
        initial_theta.log_coordinates(),
        jac=jacobian,
        bounds=(
            truth_theta.log_coordinates() - np.log(4.0),
            truth_theta.log_coordinates() + np.log(4.0),
        ),
        max_nfev=args.maximum_evaluations,
        x_scale="jac",
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=1.0e-8,
    )
    identified_theta = SrixTheta4.from_log_coordinates(fit.x)
    identified_fields, identified_residual = evaluate(fit.x)
    jacobian_truth = jacobian(truth_theta.log_coordinates())
    geometry = _geometry(jacobian_truth)
    report = {
        "schema_version": 1,
        "method": "P43 M20 synthetic direct FEMU identification smoke test",
        "git_sha": _git("rev-parse HEAD"),
        "dirty": bool(_git("status --porcelain")),
        "machine": platform.node(),
        "crop": list(CROP),
        "mesh": [angles.shape[0], angles.shape[1]],
        "history_states": 9,
        "path_subdivisions_per_macro_segment": SUBDIVISIONS,
        "path_steps": len(path),
        "scored_steps": list(scored),
        "observation_profile": "identity_synthetic_smoke",
        "parameter_preset": DEFAULT_PARAMETER_SET,
        "shadow_fd_step": H,
        "provenance": provenance,
        "truth": truth_theta.as_runtime_overrides(),
        "initial": initial_theta.as_runtime_overrides(),
        "identified": identified_theta.as_runtime_overrides(),
        "log_error_initial": (
            initial_theta.log_coordinates() - truth_theta.log_coordinates()
        ).tolist(),
        "log_error_identified": (fit.x - truth_theta.log_coordinates()).tolist(),
        "cost": {
            "initial_rms": float(np.sqrt(np.mean(initial_residual**2))),
            "identified_rms": float(np.sqrt(np.mean(identified_residual**2))),
            "truth_rms": 0.0,
        },
        "optimizer": {
            "name": "scipy.optimize.least_squares",
            "success": bool(fit.success),
            "message": str(fit.message),
            "nfev": int(fit.nfev),
            "njev": int(fit.njev or 0),
            "seconds": time.perf_counter() - started,
            "forward_evaluations": len(forward_records),
            "jacobian_evaluations": len(jacobian_records),
        },
        "forward_records": forward_records,
        "jacobian_records": jacobian_records,
        "sensitivity": geometry,
        "claims": {
            "synthetic_smoke_completed": True,
            "four_parameter_recovery_claimed": False,
            "experimental_p43_authorized": False,
        },
    }
    np.savez_compressed(
        output / "fields.npz",
        truth_displacement=np.asarray([field.displacement for field in truth_fields]),
        identified_displacement=np.asarray([field.displacement for field in identified_fields]),
        target_residual=identified_residual,
        jacobian_truth=jacobian_truth,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    axes[0].semilogy(
        [initial_residual @ initial_residual, identified_residual @ identified_residual],
        "o-",
    )
    axes[0].set(xlabel="initial / identified", ylabel="displacement residual squared")
    x = np.arange(4)
    axes[1].bar(x - 0.2, truth_theta.as_array(), width=0.2, label="truth")
    axes[1].bar(x, initial_theta.as_array(), width=0.2, label="initial")
    axes[1].bar(x + 0.2, identified_theta.as_array(), width=0.2, label="identified")
    axes[1].set_xticks(x, ("tau0", "R", "Q", "b"))
    axes[1].set_yscale("log")
    axes[1].legend()
    figure.savefig(output / "synthetic_identification_smoke.png", dpi=180)
    plt.close(figure)
    print(
        json.dumps(
            {
                "cost": report["cost"],
                "identified": report["identified"],
                "optimizer": report["optimizer"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
