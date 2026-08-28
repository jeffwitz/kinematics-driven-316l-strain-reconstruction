#!/usr/bin/env python3
"""Benchmark vectorised SRIX state plus fused block against fused state/block."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fem_inhouse.core.small_linear_solvers import (
    solve_coupled_block_numba,
    solve_coupled_state_block_numba,
)
from fem_inhouse.core.srix_numpy import SrixNumpy3DMaterialPointBatch
from scripts.run_p0043_m20_numpy_srix_forward import _centered_crop, _load_inputs

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation/reference_data/p0043_coupled_state_crossover_v1"


def _state_block(wrapper, values, slips, p_base, a_base, elastic_base):
    bridge = wrapper._bridge
    transform = bridge._kelvin_rotation
    de = values.copy()
    de[:, :3] -= np.mean(de[:, :3], axis=1)[:, None]
    deq = np.sqrt(np.maximum(2.0 * np.sum(de * de, axis=1) / 3.0, 0.0))
    slope = deq / bridge.parameters.overstress_modulus_mpa
    abs_dg = np.abs(slips)
    sign_dg = np.where(slips > 0.0, 1.0, np.where(slips < 0.0, -1.0, 0.0))
    exp_bp = np.exp(-bridge.parameters.b * (p_base + abs_dg))
    resistance = bridge.parameters.tau0_mpa + bridge.parameters.q_mpa * (
        (1.0 - exp_bp) @ bridge._interaction.T
    )
    tau_trial = (elastic_base + values) @ bridge._mce.T
    tau = tau_trial - slips @ bridge._plastic_modulus.T
    da = (slips - bridge.parameters.d * a_base * abs_dg) / (
        1.0 + bridge.parameters.d * abs_dg
    )
    drive = tau - bridge.parameters.c_mpa * (a_base + da)
    sgn = np.where(drive > 0.0, 1.0, -1.0)
    overstress = np.maximum(np.abs(drive) - resistance, 0.0)
    residual = slips - slope[:, None] * overstress * sgn
    stress_material = (
        elastic_base + values - slips @ bridge._schmid_material
    ) @ bridge._ce_material.T
    stress_global = np.einsum(
        "nij,nj->ni", np.swapaxes(transform, 1, 2), stress_material
    )
    stress_b = stress_global[:, np.array([2, 4, 5])]
    den = 1.0 + bridge.parameters.d * abs_dg
    num = slips - bridge.parameters.d * a_base * abs_dg
    dnum = 1.0 - bridge.parameters.d * a_base * sign_dg
    dden = bridge.parameters.d * sign_dg
    dda = (dnum * den - num * dden) / (den * den)
    delta_g, delta_b, success = solve_coupled_block_numba(
        slope,
        (overstress > 0.0).astype(float),
        sgn,
        exp_bp,
        sign_dg,
        dda,
        residual,
        stress_b,
        de,
        deq,
        overstress,
        bridge._mce,
        transform[:, :, np.array([2, 4, 5])],
        bridge._plastic_modulus,
        bridge._interaction,
        wrapper._coupled_dmat,
        wrapper._coupled_c_base,
        bridge.parameters.q_mpa,
        bridge.parameters.b,
        bridge.parameters.c_mpa,
        bridge.parameters.overstress_modulus_mpa,
    )
    if not np.all(success):
        raise RuntimeError("reference block solve failed")
    return residual, stress_b, delta_g, delta_b


def _make_inputs(rotations, count, rng):
    tiled = np.resize(rotations, (count, 3, 3))
    bridge = SrixNumpy3DMaterialPointBatch(
        point_count=count,
        rotation_global_to_material=tiled,
        maximum_local_iterations=30,
    )
    from fem_inhouse.core.srix_numpy import SrixNumpyCondensedPlaneStressBatch

    wrapper = SrixNumpyCondensedPlaneStressBatch(
        bridge, plane_stress_solver="coupled", coupled_block_solver="numba-fused"
    )
    values = rng.normal(0.0, 2.0e-4, (count, 6))
    values[:, 3:] *= 0.5
    slips = rng.normal(0.0, 2.0e-5, (count, 12))
    p_base = np.abs(rng.normal(0.0, 2.0e-4, (count, 12)))
    a_base = rng.normal(0.0, 1.0e-4, (count, 12))
    elastic_base = rng.normal(0.0, 2.0e-4, (count, 6))
    return wrapper, values, slips, p_base, a_base, elastic_base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[800, 2_000, 5_000, 10_000, 20_000])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    _, angles, _ = _load_inputs(_centered_crop(20))
    rotations = SrixNumpy3DMaterialPointBatch(
        point_count=angles.size // 2,
        rotation_global_to_material=np.broadcast_to(np.eye(3), (angles.size // 2, 3, 3)),
    )._rotation
    rng = np.random.default_rng(20260828)
    sizes = args.sizes
    # Compile all kernels outside the measured loops.
    warm = _make_inputs(rotations, sizes[0], rng)
    wrapper, values, slips, p_base, a_base, elastic_base = warm
    _state_block(wrapper, values, slips, p_base, a_base, elastic_base)
    solve_coupled_state_block_numba(
        values,
        slips,
        p_base,
        a_base,
        elastic_base,
        wrapper._bridge._kelvin_rotation,
        wrapper._bridge._ce_material,
        wrapper._bridge._schmid_material,
        wrapper._bridge._mce,
        wrapper._bridge._plastic_modulus,
        wrapper._bridge._interaction,
        wrapper._coupled_dmat,
        wrapper._coupled_c_base,
        wrapper._bridge._kelvin_rotation[:, :, np.array([2, 4, 5])],
        np.zeros(sizes[0], dtype=bool),
        wrapper._bridge.parameters.tau0_mpa,
        wrapper._bridge.parameters.q_mpa,
        wrapper._bridge.parameters.b,
        wrapper._bridge.parameters.c_mpa,
        wrapper._bridge.parameters.d,
        wrapper._bridge.parameters.overstress_modulus_mpa,
        wrapper._bridge._tolerance,
        wrapper._tol,
    )
    rows = []
    for count in sizes:
        wrapper, values, slips, p_base, a_base, elastic_base = _make_inputs(rotations, count, rng)
        timings = {"vectorised_state_block": [], "fused_state_block": []}
        for _ in range(3):
            started = time.perf_counter()
            reference = _state_block(wrapper, values, slips, p_base, a_base, elastic_base)
            timings["vectorised_state_block"].append(time.perf_counter() - started)
            started = time.perf_counter()
            fused = solve_coupled_state_block_numba(
                values,
                slips,
                p_base,
                a_base,
                elastic_base,
                wrapper._bridge._kelvin_rotation,
                wrapper._bridge._ce_material,
                wrapper._bridge._schmid_material,
                wrapper._bridge._mce,
                wrapper._bridge._plastic_modulus,
                wrapper._bridge._interaction,
                wrapper._coupled_dmat,
                wrapper._coupled_c_base,
                wrapper._bridge._kelvin_rotation[:, :, np.array([2, 4, 5])],
                np.zeros(count, dtype=bool),
                wrapper._bridge.parameters.tau0_mpa,
                wrapper._bridge.parameters.q_mpa,
                wrapper._bridge.parameters.b,
                wrapper._bridge.parameters.c_mpa,
                wrapper._bridge.parameters.d,
                wrapper._bridge.parameters.overstress_modulus_mpa,
                wrapper._bridge._tolerance,
                wrapper._tol,
            )
            timings["fused_state_block"].append(time.perf_counter() - started)
        rows.append(
            {
                "points": count,
                "vectorised_state_block_median_s": float(
                    np.median(timings["vectorised_state_block"])
                ),
                "fused_state_block_median_s": float(
                    np.median(timings["fused_state_block"])
                ),
                "ratio_fused_over_vectorised": float(
                    np.median(timings["fused_state_block"])
                    / np.median(timings["vectorised_state_block"])
                ),
                "max_residual_delta": float(np.max(np.abs(reference[0] - fused[0]))),
                "max_stress_b_delta": float(np.max(np.abs(reference[1] - fused[1]))),
                "max_delta_g_delta": float(np.max(np.abs(reference[2] - fused[11]))),
                "max_delta_b_delta": float(np.max(np.abs(reference[3] - fused[12]))),
            }
        )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps({"schema_version": 1, "status": "completed", "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
