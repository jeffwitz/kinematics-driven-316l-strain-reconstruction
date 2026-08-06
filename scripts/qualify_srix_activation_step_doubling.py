"""Measure step-doubling error around the first SRIX plastic activation."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from fem_inhouse.core.crystal_parameter_pairs import resolve_paired_crystal_parameters
from fem_inhouse.core.mfront_crystal_structure import read_crystal_structure_fingerprint
from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch
from fem_inhouse.spectral2d import (
    EBISpectralSolverConfig,
    StepDoublingErrorConfig,
    estimate_step_error,
)
from fem_inhouse.spectral2d.newton_two_state import (
    TraditionalTwoStateTriangleBatch,
    _step_doubling_observables,
    solve_two_state_dirichlet_plane_stress,
)
from fem_inhouse.spectral2d.transform_factory import create_full_dirichlet_dsti_plan
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig

try:
    from scripts.benchmark_tri2_j2_krylov import _load_case
except ModuleNotFoundError:
    from benchmark_tri2_j2_krylov import _load_case  # type: ignore[import-not-found,no-redef]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-nodes", nargs=4, type=int, required=True)
    parser.add_argument("--activation-fraction", type=float, default=0.0625)
    parser.add_argument(
        "--step-sizes",
        nargs="+",
        type=float,
        default=[1.0 / 128, 1.0 / 256, 1.0 / 512, 1.0 / 1024],
    )
    parser.add_argument("--paired-parameter-set", required=True)
    parser.add_argument("--mfront-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _config() -> EBISpectralSolverConfig:
    return EBISpectralSolverConfig(
        relative_equilibrium_tolerance=1.0e-8,
        linear_tolerance_mode="eisenstat_walker",
        verify_final_state=False,
        adaptive_stepping_enabled=False,
        step_doubling=StepDoublingErrorConfig(enabled=False),
        krylov_method="lgmres",
        krylov_recycling=True,
        transform=SpectralTransformConfig(
            backend="fftw",
            workers=1,
            fftw_planner_effort="measure",
            fftw_planning_time_limit_s=2.0,
            fftw_use_wisdom=False,
        ),
    )


def main() -> int:
    arguments = _parser().parse_args()
    crop = tuple(arguments.crop_nodes)
    mesh = crop[1] - crop[0]
    if mesh != crop[3] - crop[2]:
        raise SystemExit("crop must be square")
    if not 0.0 < arguments.activation_fraction < 1.0:
        raise SystemExit("activation fraction must be in (0, 1)")

    grid, _, yield_stress, coefficient, boundary = _load_case(mesh, crop)
    _overrides, manifest = resolve_paired_crystal_parameters(
        paired_parameter_set=arguments.paired_parameter_set,
        law="forest_rubin_srix",
    )
    source = Path(__file__).resolve().parents[1] / "mfront/Fcc316LForestRubinSrix.mfront"
    fingerprint = read_crystal_structure_fingerprint(source)
    material = create_plane_stress_material_batch(
        "mfront-3d-condensed-plane-stress",
        np.repeat(yield_stress, 2),
        np.repeat(coefficient, 2),
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1_000,
        first_positive_plastic_strain=1.0e-6,
        mfront_library=os.environ.get(
            "MFRONT_BEHAVIOUR_LIBRARY", "build/mfront/src/libBehaviour.so"
        ),
        mfront_threads=arguments.mfront_threads,
        mfront_behaviour_id="fcc_forest_rubin_srix",
        local_plane_stress_options={"local_condition_check_mode": "on_failure"},
        constitutive_options={
            "crystal_orientation": {
                "mode": "homogeneous",
                "euler_bunge_deg": [35.0, 20.0, 15.0],
            },
            "paired_parameter_set": arguments.paired_parameter_set,
        },
    )
    plan = create_full_dirichlet_dsti_plan(grid, _config().transform)
    elements = TraditionalTwoStateTriangleBatch(material, grid.pixel_shape)
    zero = np.zeros_like(boundary)
    activation_boundary = arguments.activation_fraction * boundary
    started = time.perf_counter()
    preload = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=material,
        boundary_displacement_history=np.stack((zero, activation_boundary)),
        config=_config(),
        transform_plan=plan,
        time_increment_override=arguments.activation_fraction,
    )
    start_snapshot = elements.snapshot_state()
    base_config = replace(
        _config(),
        reference_parameter_mode="explicit",
        reference_lambda_0=preload.diagnostics.reference_lambda_0,
        reference_mu_0=preload.diagnostics.reference_mu_0,
    )

    def attempt(start: float, end: float, snapshot: object):
        elements.restore_state(snapshot)
        result = solve_two_state_dirichlet_plane_stress(
            grid=grid,
            material=material,
            boundary_displacement_history=np.stack((zero, end * boundary)),
            config=base_config,
            transform_plan=plan,
            time_increment_override=end - start,
        )
        return elements.snapshot_state(), _step_doubling_observables(result), result

    records: list[dict[str, object]] = []
    for step in arguments.step_sizes:
        _coarse_state, coarse_obs, coarse_result = attempt(
            arguments.activation_fraction,
            arguments.activation_fraction + step,
            start_snapshot,
        )
        midpoint = arguments.activation_fraction + 0.5 * step
        first_state, _, first_result = attempt(
            arguments.activation_fraction, midpoint, start_snapshot
        )
        _, fine_obs, fine_result = attempt(
            midpoint, arguments.activation_fraction + step, first_state
        )
        error = estimate_step_error(
            fine_obs,
            coarse_obs,
            StepDoublingErrorConfig(
                stress_relative_tolerance=5.0e-3,
                reaction_relative_tolerance=5.0e-3,
                signed_slip_relative_tolerance=5.0e-3,
                accumulated_slip_relative_tolerance=5.0e-3,
                displacement_relative_tolerance=5.0e-5,
            ),
        )
        records.append(
            {
                "step_size": step,
                "start_fraction": arguments.activation_fraction,
                "end_fraction": arguments.activation_fraction + step,
                "coarse_newton_iterations": sum(
                    coarse_result.diagnostics.iterations_per_increment
                ),
                "first_half_newton_iterations": sum(
                    first_result.diagnostics.iterations_per_increment
                ),
                "second_half_newton_iterations": sum(
                    fine_result.diagnostics.iterations_per_increment
                ),
                "maximum_error_ratio": error.maximum_ratio,
                "controlling_quantity": error.controlling_quantity,
                "controlling_system": error.controlling_system,
                "stress_ratio": error.stress.ratio,
                "stress_relative_l2": error.stress.relative_l2,
                "signed_slip_ratio_per_system": error.signed_slip_ratio_per_system.tolist(),
                "equivalent_plastic_slip_ratio_per_system": (
                    error.equivalent_plastic_slip_ratio_per_system.tolist()
                ),
                "signed_slip_active_set_mismatch": (
                    error.signed_slip_details.active_set_mismatch.tolist()
                ),
                "equivalent_plastic_slip_active_set_mismatch": (
                    error.equivalent_plastic_slip_details.active_set_mismatch.tolist()
                ),
                "fine_maximum_slip_per_system": (
                    error.signed_slip_details.fine_linf_amplitudes.tolist()
                ),
                "coarse_maximum_slip_per_system": (
                    error.signed_slip_details.coarse_linf_amplitudes.tolist()
                ),
                "absolute_difference_l2_per_system": (
                    error.signed_slip_details.difference_l2_norms.tolist()
                ),
            }
        )

    report = {
        "status": "complete",
        "mesh": [mesh, mesh],
        "crop_nodes": list(crop),
        "activation_fraction": arguments.activation_fraction,
        "step_sizes": arguments.step_sizes,
        "paired_parameter_set": arguments.paired_parameter_set,
        "mfront_threads": arguments.mfront_threads,
        "source_sha256": fingerprint.source_sha256,
        "backbone_manifest": manifest,
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
