#!/usr/bin/env python3
"""Benchmark legacy and lightweight micromorphic constitutive evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.core.mfront import MFrontNativePlaneStressBatch
from fem_inhouse.core.nonlocal_plasticity import (
    NonlocalFixedPointWorkspace,
    _element_average,
    _gauss_values,
    _mixed_relative_maximum_norm,
    evaluate_nonlocal_fixed_point,
)
from fem_inhouse.partitioning import PartitionLayout
from fem_inhouse.postprocessing.helmholtz import helmholtz_filter_element_field
from fem_inhouse.postprocessing.kinematics import cell_average, strain_from_displacement

FloatArray = NDArray[np.float64]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--partition-id", type=int, required=True)
    parser.add_argument("--parts-x", type=int, default=10)
    parser.add_argument("--parts-y", type=int, default=10)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--mode", choices=("legacy", "lightweight"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--spacing-mm", type=float, default=0.00184)
    parser.add_argument("--length-scale-mm", type=float, default=0.05888)
    parser.add_argument("--coupling-modulus-mpa", type=float, required=True)
    parser.add_argument("--relaxation", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--maximum-iterations", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _prepare_partition(args: argparse.Namespace) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    tuple[int, int],
    dict[str, Any],
]:
    displacement_x = np.load(
        args.input / "displacement_x_mm.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    displacement_y = np.load(
        args.input / "displacement_y_mm.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    yield_stress = np.load(
        args.input / "yield_stress_mpa.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    hardening = np.load(
        args.input / "hardening_coefficient_mpa.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    global_shape = (int(yield_stress.shape[0]), int(yield_stress.shape[1]))
    layout = PartitionLayout(
        global_shape,
        (args.parts_x, args.parts_y),
        padding=args.padding,
    )
    partition = layout.get(args.partition_id)
    ex = partition.solve_element_slice_global[0]
    ey = partition.solve_element_slice_global[1]
    nx_slice = slice(ex.start, ex.stop + 1)
    ny_slice = slice(ey.start, ey.stop + 1)
    kinematics = strain_from_displacement(
        np.asarray(displacement_x[nx_slice, ny_slice]),
        np.asarray(displacement_y[nx_slice, ny_slice]),
        spacing_x=args.spacing_mm,
        spacing_y=args.spacing_mm,
    )
    element_strain = np.stack(
        (
            cell_average(kinematics.epsilon_xx),
            cell_average(kinematics.epsilon_yy),
            cell_average(kinematics.gamma_xy),
        ),
        axis=-1,
    )
    element_shape = (int(element_strain.shape[0]), int(element_strain.shape[1]))
    point_strain = np.repeat(
        element_strain.reshape(-1, 3, order="F"),
        4,
        axis=0,
    )
    point_yield = np.repeat(
        np.asarray(yield_stress[ex, ey]).ravel(order="F"),
        4,
    )
    point_hardening = np.repeat(
        np.asarray(hardening[ex, ey]).ravel(order="F"),
        4,
    )
    metadata = {
        "partition": partition.as_dict(),
        "element_shape": list(element_shape),
        "point_count": int(point_strain.shape[0]),
    }
    return point_strain, point_yield, point_hardening, element_shape, metadata


def _legacy_fixed_point(
    batch: MFrontNativePlaneStressBatch,
    strain: FloatArray,
    *,
    element_shape: tuple[int, int],
    length_scale_mm: float,
    spacing_mm: float,
    coupling_modulus_mpa: float,
    relaxation: float,
    tolerance: float,
    maximum_iterations: int,
) -> dict[str, Any]:
    chi = np.zeros(element_shape, dtype=np.float64)
    relative_change = float("inf")
    iterations = 0
    for iteration in range(1, maximum_iterations + 1):
        iterations = iteration
        batch.set_nonlocal_equivalent_plastic_strain(_gauss_values(chi, 4))
        trial = batch.evaluate(strain, time_increment=1.0, consistent_tangent=True)
        local_peeq = _element_average(
            trial.observables["equivalent_plastic_strain"],
            element_shape=element_shape,
            gauss_points_per_element=4,
            name="equivalent_plastic_strain",
        )
        filtered = helmholtz_filter_element_field(
            local_peeq,
            length_scale_mm=length_scale_mm,
            spacing_x_mm=spacing_mm,
            spacing_y_mm=spacing_mm,
        ).filtered_element_field
        next_chi = (1.0 - relaxation) * chi + relaxation * filtered
        relative_change = _mixed_relative_maximum_norm(
            next_chi - chi,
            next_chi,
            filtered,
        )
        chi = next_chi
        if relative_change <= relaxation * tolerance:
            break
    else:
        raise RuntimeError(
            f"legacy fixed point did not converge: residual={relative_change:.3e}"
        )
    batch.set_nonlocal_equivalent_plastic_strain(_gauss_values(chi, 4))
    final = batch.evaluate(strain, time_increment=1.0, consistent_tangent=True)
    return {
        "trial": final,
        "chi": chi,
        "iterations": iterations,
        "relative_residual": relative_change,
    }


def _array_sha256(array: FloatArray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    strain, yield_stress, hardening, element_shape, metadata = _prepare_partition(args)
    batch = MFrontNativePlaneStressBatch(
        args.library,
        yield_stress,
        hardening,
        np.full(yield_stress.shape, 0.245),
        thread_count=args.threads,
        behaviour_name="PixelMicromorphicLudwikJ2Plasticity",
        micromorphic_coupling_modulus_mpa=args.coupling_modulus_mpa,
    )
    started = time.perf_counter()
    if args.mode == "legacy":
        legacy_evaluation = _legacy_fixed_point(
            batch,
            strain,
            element_shape=element_shape,
            length_scale_mm=args.length_scale_mm,
            spacing_mm=args.spacing_mm,
            coupling_modulus_mpa=args.coupling_modulus_mpa,
            relaxation=args.relaxation,
            tolerance=args.tolerance,
            maximum_iterations=args.maximum_iterations,
        )
        trial = legacy_evaluation["trial"]
        chi = legacy_evaluation["chi"]
        iterations = legacy_evaluation["iterations"]
        relative_residual = legacy_evaluation["relative_residual"]
    else:
        lightweight_evaluation = evaluate_nonlocal_fixed_point(
            batch,
            strain,
            time_increment=1.0,
            element_shape=element_shape,
            gauss_points_per_element=4,
            initial_nonlocal_peeq=np.zeros(element_shape),
            length_scale_mm=args.length_scale_mm,
            spacing_x_mm=args.spacing_mm,
            spacing_y_mm=args.spacing_mm,
            coupling_modulus_mpa=args.coupling_modulus_mpa,
            relaxation=args.relaxation,
            relative_tolerance=args.tolerance,
            maximum_iterations=args.maximum_iterations,
            maximum_helmholtz_residual=1e-10,
            workspace=NonlocalFixedPointWorkspace.create(element_shape, 4),
        )
        trial = lightweight_evaluation.constitutive_trial
        chi = lightweight_evaluation.nonlocal_peeq
        iterations = lightweight_evaluation.iterations
        relative_residual = lightweight_evaluation.relative_residual
        trial = batch.complete_trial(trial)
    elapsed = time.perf_counter() - started
    peeq = np.asarray(trial.observables["equivalent_plastic_strain"])
    tangent = np.asarray(trial.tangent_in_plane_mpa)
    stress = np.asarray(trial.stress_in_plane_mpa)
    np.savez(
        args.output / "fields.npz",
        stress_in_plane_mpa=stress,
        tangent_in_plane_mpa=tangent,
        equivalent_plastic_strain=peeq,
        nonlocal_equivalent_plastic_strain=chi,
    )
    timing = batch.timing_statistics
    report = {
        **metadata,
        "mode": args.mode,
        "elapsed_seconds": elapsed,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "iterations": int(iterations),
        "relative_residual": float(relative_residual),
        "length_scale_mm": args.length_scale_mm,
        "coupling_modulus_mpa": args.coupling_modulus_mpa,
        "relaxation": args.relaxation,
        "tolerance": args.tolerance,
        "threads": args.threads,
        "timing": {
            key: getattr(timing, key)
            for key in timing.__dataclass_fields__
        },
        "hashes": {
            "stress_in_plane_mpa": _array_sha256(stress),
            "tangent_in_plane_mpa": _array_sha256(tangent),
            "equivalent_plastic_strain": _array_sha256(peeq),
            "nonlocal_equivalent_plastic_strain": _array_sha256(chi),
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
