#!/usr/bin/env python3
"""Run the P43 M20 forward with the vectorised NumPy SRIX backend (order F)."""

from __future__ import annotations

import json
import platform
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch
from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET
from fem_inhouse.identification.srix_parameter_coordinates import SrixTheta9
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_two_state import (
    TwoStateIncrementFields,
    solve_two_state_dirichlet_plane_stress,
)
from fem_inhouse.spectral2d.step_control import LoadPathStep
from scripts.plot_p0043_raw_svd7_evm_maps import _evm
from scripts.qualify_srix_femu_direct_sensitivity import _oracle_config
from scripts.qualify_srix_p0043_synthetic_smoke import CROP, _load_inputs, _make_path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPORT = ROOT / (
    "validation/reference_data/p0043_experimental_raw_svd7_f_provisional_v1/report.json"
)
MFRONT_FIELDS = ROOT / "validation/reference_data/p0043_m20_c_f_forward_identified_v1/fields_F.npz"
OUTPUT = ROOT / "validation/reference_data/p0043_m20_numpy_srix_forward_f_v1"
PIXEL_SIZE_MM = 0.00184


def _centered_crop(pixels: int) -> tuple[int, int, int, int]:
    """Return a square crop centered on the qualified P43 M20 crop."""
    if pixels < 1:
        raise ValueError("pixels must be positive")
    cx = (CROP[0] + CROP[1]) // 2
    cy = (CROP[2] + CROP[3]) // 2
    half = pixels // 2
    return (cx - half, cx - half + pixels, cy - half, cy - half + pixels)


def _factory(
    angles: np.ndarray,
    theta: SrixTheta9,
    local_iterations: int,
    predictor: str = "committed",
    parallel_backend: str = "serial",
    dask_workers: int = 1,
    batch_size: int | None = None,
    local_linear_solver: str = "numpy",
    plane_stress_solver: str = "nested",
    coupled_block_solver: str = "numpy",
):
    count = 2 * angles.shape[0] * angles.shape[1]
    return create_plane_stress_material_batch(
        "numpy-srix-plane-stress",
        np.ones(count),
        np.ones(count),
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1_000,
        first_positive_plastic_strain=1.0e-6,
        mfront_library="",
        mfront_threads=1,
        local_plane_stress_options={
            "material_newton_max_iterations": local_iterations,
            "plane_stress_max_iterations": 15,
            "local_tolerance_mpa": 1.0e-8,
            "local_transverse_predictor": predictor,
            "plane_stress_solver": plane_stress_solver,
        },
        constitutive_options={
            "parameter_set": DEFAULT_PARAMETER_SET,
            "parameters": theta.as_runtime_overrides(),
            "parallel_backend": parallel_backend,
            "dask_workers": dask_workers,
            "batch_size": batch_size,
            "local_linear_solver": local_linear_solver,
            "coupled_block_solver": coupled_block_solver,
            "crystal_orientation": {
                "mode": "ebsd",
                "euler_bunge_deg": angles,
                "element_order": "F",
            },
        },
    )


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
        plastic_strain_tensor=None
        if value.plastic_strain_tensor is None
        else np.asarray(value.plastic_strain_tensor).copy(),
        elastic_strain_tensor=None
        if value.elastic_strain_tensor is None
        else np.asarray(value.elastic_strain_tensor).copy(),
        observables={name: np.asarray(data).copy() for name, data in value.observables.items()},
    )


def _forward(
    theta: SrixTheta9,
    path: list[LoadPathStep],
    angles: np.ndarray,
    local_iterations: int,
    predictor: str = "committed",
    parallel_backend: str = "serial",
    dask_workers: int = 1,
    batch_size: int | None = None,
    local_linear_solver: str = "numpy",
    plane_stress_solver: str = "nested",
    coupled_block_solver: str = "numpy",
) -> tuple[list[TwoStateIncrementFields], dict[str, Any]]:
    pixels = angles.shape[0]
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    material = _factory(
        angles,
        theta,
        local_iterations,
        predictor,
        parallel_backend,
        dask_workers,
        batch_size,
        local_linear_solver,
        plane_stress_solver,
        coupled_block_solver,
    )
    history = np.stack([np.zeros_like(path[0].boundary), *[step.boundary for step in path]])
    fields: list[TwoStateIncrementFields] = []

    started = time.perf_counter()
    result = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=material,
        boundary_displacement_history=history,
        config=_oracle_config(),
        load_path_override=path,
        increment_observer=lambda value: fields.append(_copy_field(value)),
    )
    elapsed = time.perf_counter() - started
    if len(fields) != len(path):
        raise RuntimeError(f"NumPy forward accepted {len(fields)} of {len(path)} increments")
    timing_result = {
        "seconds": elapsed,
        "steps": len(fields),
        "verification_residual": result.diagnostics.verification_residual,
        "gmres_iterations": int(result.diagnostics.timings["gmres_iterations"]),
        "backend": "numpy-srix-condensed-plane-stress",
        "local_transverse_predictor": predictor,
        "parallel_backend": parallel_backend,
        "dask_workers": dask_workers,
        "batch_size": batch_size,
        "local_linear_solver": local_linear_solver,
        "plane_stress_solver": plane_stress_solver,
        "coupled_block_solver": coupled_block_solver,
    }
    timing_result["material"] = material.timing_statistics
    timing_result["global_newton_iterations_per_increment"] = list(
        result.diagnostics.iterations_per_increment
    )
    timing_result["global_newton_iterations_total"] = int(
        sum(result.diagnostics.iterations_per_increment)
    )
    timing_result["global_relative_residual_history"] = list(
        result.diagnostics.relative_residual_history
    )
    timing_result["global_absolute_residual_history"] = list(
        result.diagnostics.absolute_residual_history
    )
    timing_result["linear_solves"] = [asdict(item) for item in result.diagnostics.linear_solves]
    return fields, timing_result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pixels", type=int, default=20)
    parser.add_argument("--subdivisions", type=int, default=4)
    parser.add_argument("--local-iterations", type=int, default=30)
    parser.add_argument("--predictor", choices=("committed", "tangent"), default="committed")
    parser.add_argument("--parallel-backend", choices=("serial", "dask-threads"), default="serial")
    parser.add_argument("--dask-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--local-linear-solver", choices=("numpy", "numba-lu12"), default="numpy")
    parser.add_argument("--plane-stress-solver", choices=("nested", "coupled"), default="nested")
    parser.add_argument("--coupled-block-solver", choices=("numpy", "numba-fused"), default="numpy")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    crop = _centered_crop(args.pixels)
    measured, angles, provenance = _load_inputs(crop)
    path = _make_path(measured, args.subdivisions)
    scored = tuple(args.subdivisions * i for i in range(1, 9))
    target = np.asarray([path[i - 1].boundary for i in scored])
    report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    theta = SrixTheta9.from_log_coordinates(np.asarray(report["final_eta"], dtype=float))
    started = time.perf_counter()
    try:
        fields, timing = _forward(
            theta,
            path,
            angles,
            args.local_iterations,
            args.predictor,
            args.parallel_backend,
            args.dask_workers,
            args.batch_size,
            args.local_linear_solver,
            args.plane_stress_solver,
            args.coupled_block_solver,
        )
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "backend": "numpy-srix-condensed-plane-stress",
            "element_order": "F",
            "crop": list(crop),
            "path_steps": len(path),
            "local_iterations": args.local_iterations,
            "local_transverse_predictor": args.predictor,
            "parallel_backend": args.parallel_backend,
            "dask_workers": args.dask_workers,
            "batch_size": args.batch_size,
            "local_linear_solver": args.local_linear_solver,
            "plane_stress_solver": args.plane_stress_solver,
            "coupled_block_solver": args.coupled_block_solver,
            "elapsed_seconds_wall": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        (output / "failure_report.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(failure, sort_keys=True), flush=True)
        return 1
    selected = np.asarray([field.displacement for field in fields])[np.asarray(scored) - 1]
    residual = selected - target
    evm = np.asarray([_evm(field) for field in selected])
    dic_evm = np.asarray([_evm(field) for field in target])
    np.savez_compressed(
        output / "fields_numpy.npz",
        displacement=np.asarray([field.displacement for field in fields]),
        sample_strain=np.asarray([field.sample_strain for field in fields]),
        stress_in_plane_mpa=np.asarray([field.stress_in_plane_mpa for field in fields]),
        scored_displacement=selected,
        dic_displacement=target,
        evm=evm,
        dic_evm=dic_evm,
    )
    mfront = np.load(MFRONT_FIELDS)
    mfront_selected = np.asarray(mfront["scored_displacement"])
    if mfront_selected.shape == selected.shape:
        delta = selected - mfront_selected
        mfront_comparison: dict[str, Any] = {
            "source": str(MFRONT_FIELDS.relative_to(ROOT)),
            "displacement_max_abs_mm": float(np.max(np.abs(delta))),
            "displacement_relative_l2": float(
                np.linalg.norm(delta) / max(np.linalg.norm(mfront_selected), 1.0e-30)
            ),
            "displacement_rmse_mm": float(np.sqrt(np.mean(delta**2))),
        }
    else:
        mfront_comparison = {
            "source": str(MFRONT_FIELDS.relative_to(ROOT)),
            "status": "not_comparable_shape",
            "numpy_shape": list(selected.shape),
            "mfront_shape": list(mfront_selected.shape),
        }
    result = {
        "schema_version": 1,
        "method": "P43 M20 NumPy SRIX forward, EBSD order F",
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "machine": platform.node(),
        "crop": list(crop),
        "path_steps": len(path),
        "scored_steps": list(scored),
        "parameters": theta.as_runtime_overrides(),
        "provenance": provenance,
        "timing": timing,
        "raw_rms_mm": float(np.sqrt(np.mean(residual**2))),
        "mfront_comparison": mfront_comparison,
        "evm_rms_percent": (100.0 * np.sqrt(np.mean(evm**2, axis=(1, 2)))).tolist(),
        "dic_evm_rms_percent": (100.0 * np.sqrt(np.mean(dic_evm**2, axis=(1, 2)))).tolist(),
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
