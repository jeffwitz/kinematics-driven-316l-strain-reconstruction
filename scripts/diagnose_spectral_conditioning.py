"""Spectral analysis of the assembled B^T C B at the deep plastic states.

The five-test falsification programme acquitted every local hypothesis and
showed the Newton count follows the TANGENT, not the stress: with the
reference tangent the GPS residual converges like the reference (47 vs 46),
with the GPS tangent even the reference residual converges like the GPS
(54 vs 52). The remaining suspect is the spectrum of the matrix the global
Newton actually iterates: J = B^T C B assembled from the material tangent
through the spectral kinematics, at the REAL states of the two trajectories
(which differ at the deep increments).

This script assembles the FULL matrix J for each backend at the deep
increments (5-8), from the recorded trial of the solver's own first call of
each increment, and reports:

- the raw spectrum: max|lambda| / min|lambda| (conditioning) and the full
  eigenvalue list, at each backend's own committed state;
- the PRECONDITIONED spectrum of J P, P being the same spectral green
  preconditioner the solver applies (identical for both backends), which is
  the operator GMRES actually iterates: clustering around 1 is what a fast
  linear solve needs.

Diagnosis: the GPS matrix is a worse iterator at the deep states (linear
x0.2 vs x0.05 measured); the spectrum should show it -- either a wider
conditioning of J, or a less clustered preconditioned spectrum J P.

Usage:

    .venv/bin/python scripts/diagnose_spectral_conditioning.py \
        --output validation/_generated/performance/spectral_conditioning.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CROP_20X20 = (1610, 1630, 1075, 1095)
EBSD_ORIENTATION_H5 = "/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5"
PAIRED_PARAMETER_SET = "316l_guilhem2013_nasri2018_meric_srix_rate_1e-3"
ANALYSED_INCREMENTS = (5, 6, 7, 8)


class RecordingMaterial:
    """Record every solver call with its tangent and the committed snapshots."""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.calls: list[dict[str, object]] = []
        self.committed_snapshots: list[object] = []
        self._last_substeps = 0

    def _record(
        self,
        strain: np.ndarray,
        trial: object,
        time_increment: float,
    ) -> None:
        tangent = getattr(trial, "tangent_in_plane_mpa", None)
        substeps = int(getattr(self._inner, "_substep_uses", 0))
        self.calls.append(
            {
                "strain": np.asarray(strain, dtype=float).copy(),
                "tangent": (
                    None
                    if tangent is None
                    else np.asarray(tangent, dtype=float).copy()
                ),
                "time_increment": float(time_increment),
                "substeps": substeps - self._last_substeps,
                "committed_before": len(self.committed_snapshots),
            }
        )
        self._last_substeps = substeps

    def evaluate(
        self,
        in_plane_strain: np.ndarray,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> object:
        strain = np.asarray(in_plane_strain, dtype=float)
        trial = self._inner.evaluate(
            strain, time_increment=time_increment, consistent_tangent=consistent_tangent
        )
        self._record(strain, trial, time_increment)
        return trial

    def evaluate_in_plane(
        self,
        in_plane_strain: np.ndarray,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ) -> object:
        strain = np.asarray(in_plane_strain, dtype=float)
        trial = self._inner.evaluate_in_plane(
            strain, time_increment=time_increment, consistent_tangent=consistent_tangent
        )
        self._record(strain, trial, time_increment)
        return trial

    def evaluate_in_plane_response(
        self,
        in_plane_strain: np.ndarray,
        *,
        time_increment: float,
        response_level: str = "tangent",
        consistent_tangent: bool = True,
    ) -> object:
        strain = np.asarray(in_plane_strain, dtype=float)
        from fem_inhouse.core.plane_stress_material import evaluate_in_plane_response

        trial = evaluate_in_plane_response(
            self._inner,
            strain,
            time_increment=time_increment,
            response_level=response_level,
            consistent_tangent=consistent_tangent,
        )
        self._record(strain, trial, time_increment)
        return trial

    def commit(self) -> None:
        self._inner.commit()
        self.committed_snapshots.append(self._inner.snapshot_state())

    def revert(self) -> None:
        self._inner.revert()

    @property
    def point_count(self) -> int:
        return self._inner.point_count

    @property
    def timing_statistics(self) -> object:
        return getattr(self._inner, "timing_statistics", None)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


def _run_backend(
    backend: str,
    arguments: argparse.Namespace,
) -> dict[str, object]:
    from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch
    from fem_inhouse.spectral2d.green import B0Green2D
    from fem_inhouse.spectral2d.newton_two_state import (
        EBISpectralSolverConfig,
        TwoStateJacobianWorkspace,
        TwoSubcellDiagnostic2D,
        solve_two_state_dirichlet_plane_stress,
    )
    from fem_inhouse.spectral2d.transform_factory import create_full_dirichlet_dsti_plan
    from fem_inhouse.spectral2d.transforms import SpectralTransformConfig
    from scripts.benchmark_tri2_j2_krylov import _load_case
    from scripts.qualify_crystal_tet2_p43 import _load_ebsd_orientation_crop

    mesh = arguments.crop_nodes[1] - arguments.crop_nodes[0]
    grid, _, yield_stress, coefficient, boundary = _load_case(mesh, arguments.crop_nodes)
    history = np.stack(
        [fraction * boundary for fraction in np.linspace(0.0, 1.0, arguments.increments + 1)]
    )
    material = create_plane_stress_material_batch(
        backend,
        np.repeat(yield_stress, 2),
        np.repeat(coefficient, 2),
        0.245,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1_000,
        first_positive_plastic_strain=1.0e-6,
        mfront_library=arguments.library,
        mfront_threads=arguments.mfront_threads,
        mfront_behaviour_id="fcc_forest_rubin_srix",
        local_plane_stress_options={
            "local_condition_check_mode": "on_failure",
            "local_transverse_predictor": "tangent",
        },
        constitutive_options={
            "paired_parameter_set": arguments.paired_parameter_set,
            "crystal_orientation": {
                "mode": "ebsd",
                "euler_bunge_deg": _load_ebsd_orientation_crop(
                    arguments.ebsd_orientation_h5, arguments.crop_nodes
                )[0],
            },
        },
    )
    recording = RecordingMaterial(material)
    config = EBISpectralSolverConfig(
        relative_equilibrium_tolerance=1e-8,
        maximum_newton_iterations=arguments.maximum_newton_iterations,
        krylov_method="lgmres",
        krylov_recycling=True,
        linear_tolerance_mode="eisenstat_walker",
        verify_final_state=False,
        transform=SpectralTransformConfig(
            backend="fftw",
            workers=1,
            fftw_planner_effort="measure",
            fftw_planning_time_limit_s=2.0,
            fftw_use_wisdom=False,
        ),
    )
    # The preconditioner's parameters: projected from the ZERO-strain tangent
    # on the VIRGIN material, exactly as the solver does before the loop.
    # (Evaluating it after the run would ask the deep plastic state for a
    # zero increment -- the known trap -- so it is done here, once.)
    from fem_inhouse.core.plane_stress_material import evaluate_in_plane_response
    from fem_inhouse.spectral2d.green import project_isotropic_plane_stress_tangent

    zero_trial = evaluate_in_plane_response(
        material,
        np.zeros((material.point_count, 3)),
        time_increment=1.0,
        response_level="tangent",
        consistent_tangent=True,
    )
    material.revert()
    tangent0 = np.asarray(zero_trial.tangent_in_plane_mpa).reshape(
        *grid.pixel_shape, 2, 3, 3
    )
    projected_lambda, projected_mu, _ = project_isotropic_plane_stress_tangent(
        tangent0.mean(axis=(0, 1, 2))
    )

    result = solve_two_state_dirichlet_plane_stress(
        grid=grid,
        material=recording,
        boundary_displacement_history=history[: arguments.increments + 1],
        config=config,
    )
    calls = recording.calls
    snapshots = recording.committed_snapshots

    # Rebuild the solver's own operators: kinematics, the two-state batch and
    # the spectral preconditioner (identical for both backends).
    kinematics = TwoSubcellDiagnostic2D(grid)
    plan = create_full_dirichlet_dsti_plan(grid, config.transform)
    workspace = TwoStateJacobianWorkspace.create(grid)
    interior_shape = (*grid.interior_shape, 2)
    spectral_buffer = np.empty(interior_shape, dtype=np.float64)
    green_buffer = np.empty_like(spectral_buffer)
    physical_buffer = np.empty_like(spectral_buffer)

    from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior_into

    size = 2 * (grid.nx - 1) * (grid.ny - 1)

    def jacobian_action(vector: np.ndarray, trial: object) -> np.ndarray:
        unpack_interior_into(vector, grid, workspace.nodal_increment)
        delta_stress = np.einsum(
            "xyqij,xyqj->xyqi", trial.algorithmic_tangent_in_plane_mpa,
            kinematics.strain_samples(workspace.nodal_increment),
        )
        kinematics.divergence_from_sample_stress_into(
            delta_stress, workspace.nodal_force
        )
        return pack_interior(workspace.nodal_force)

    def preconditioner_action(vector: np.ndarray) -> np.ndarray:
        interior = np.asarray(vector, dtype=np.float64).reshape(interior_shape)
        if hasattr(plan, "forward_into"):
            plan.forward_into(interior, spectral_buffer)
        else:
            spectral_buffer[...] = plan.forward_displacement(interior)
        green.apply_into(spectral_buffer, green_buffer)
        if hasattr(plan, "forward_into"):
            plan.inverse_into(green_buffer, physical_buffer)
        else:
            physical_buffer[...] = plan.inverse_displacement(green_buffer)
        return physical_buffer.reshape(-1).copy()

    def assemble_matrix(action: object) -> np.ndarray:
        columns = []
        for index in range(size):
            basis = np.zeros(size, dtype=float)
            basis[index] = 1.0
            columns.append(action(basis))
        return np.stack(columns, axis=1)

    # The preconditioner itself: the same green the solver applies, built
    # from the zero-strain projection above, identical for both backends.
    green = B0Green2D(
        kinematics.reference_operator_symbols(plan),
        lambda_0=(
            projected_lambda
            * config.reference_parameter_scale
            * config.reference_lambda_mu_ratio
        ),
        mu_0=projected_mu * config.reference_parameter_scale,
        symbol_null_tolerance=config.symbol_null_tolerance,
    )

    from scipy.linalg import eigvals

    # The matrices the solver ACTUALLY applied: the recorded tangents of its
    # own calls (last-sub-step tangent for the GPS included), assembled into
    # B^T C B. The committed-state matrices are identical for both backends
    # (tangents match to 1e-16 at paired states), so the interesting object is
    # the sequence along the iterations of each deep increment.
    class TangentOnlyTrial:
        """Minimal trial carrying just the tangent the assembly needs."""

        def __init__(self, tangent: np.ndarray) -> None:
            self.algorithmic_tangent_in_plane_mpa = tangent

    preconditioner_matrix = assemble_matrix(preconditioner_action)
    per_increment: dict[str, object] = {}
    for increment in ANALYSED_INCREMENTS:
        committed = increment - 1
        if committed >= len(snapshots):
            continue
        increment_calls = [
            call for call in calls if call["committed_before"] == committed
        ]
        if not increment_calls:
            continue
        per_call: list[dict[str, object]] = []
        for call_index, call in enumerate(increment_calls):
            tangent = call["tangent"]
            if tangent is None:
                continue
            trial = TangentOnlyTrial(
                np.asarray(tangent).reshape(*grid.pixel_shape, 2, 3, 3)
            )
            jacobian = assemble_matrix(
                lambda v, trial=trial: jacobian_action(v, trial)
            )
            raw = eigvals(jacobian)
            raw_abs = np.abs(raw)
            jp = jacobian @ preconditioner_matrix
            precond = eigvals(jp)
            precond_abs = np.abs(precond)
            per_call.append(
                {
                    "call": call_index + 1,
                    "substeps": int(call["substeps"]),
                    "raw_max_abs": float(raw_abs.max()),
                    "raw_min_abs": float(raw_abs.min()),
                    "raw_conditioning": float(
                        raw_abs.max() / max(raw_abs.min(), 1.0e-30)
                    ),
                    "preconditioned_max_abs": float(precond_abs.max()),
                    "preconditioned_min_abs": float(precond_abs.min()),
                    "preconditioned_conditioning": float(
                        precond_abs.max() / max(precond_abs.min(), 1.0e-30)
                    ),
                }
            )
        per_increment[str(increment)] = {
            "calls": len(increment_calls),
            "per_call": per_call,
        }
    material.revert()
    total_newton = int(sum(result.diagnostics.iterations_per_increment))
    return {
        "backend": backend,
        "newton_iterations": total_newton,
        "matrix_size": size,
        "per_increment": per_increment,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=CROP_20X20)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--library", default="build/mfront/src/libBehaviour.so")
    parser.add_argument("--ebsd-orientation-h5", type=Path, default=Path(EBSD_ORIENTATION_H5))
    parser.add_argument("--paired-parameter-set", default=PAIRED_PARAMETER_SET)
    parser.add_argument("--mfront-threads", type=int, default=4)
    parser.add_argument("--maximum-newton-iterations", type=int, default=40)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/_generated/performance/spectral_conditioning.json"),
    )
    arguments = parser.parse_args()
    backends = (
        "mfront-native-generalised-plane-stress",
        "mfront-3d-condensed-plane-stress",
    )
    records = [_run_backend(backend, arguments) for backend in backends]
    payload = {
        "schema_version": 1,
        "configuration": {
            "crop_nodes": arguments.crop_nodes,
            "increments": arguments.increments,
            "mfront_threads": arguments.mfront_threads,
            "analysed_increments": list(ANALYSED_INCREMENTS),
        },
        "backends": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for record in records:
        size = record["matrix_size"]
        print(
            f"=== {record['backend']}: {record['newton_iterations']} Newton, "
            f"matrix {size}x{size}"
        )
        for increment, data in record["per_increment"].items():
            calls = data["per_call"]
            raw_conds = [c["raw_conditioning"] for c in calls]
            precond_conds = [c["preconditioned_conditioning"] for c in calls]
            substeps = [c["substeps"] for c in calls]
            print(
                f"  inc {increment} ({data['calls']} calls): "
                f"raw cond {min(raw_conds):.2e}..{max(raw_conds):.2e} | "
                f"precond cond {min(precond_conds):.2e}..{max(precond_conds):.2e} | "
                f"substeps {substeps}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
