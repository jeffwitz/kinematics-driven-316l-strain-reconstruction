#!/usr/bin/env python3
"""Project the P43 DIC residual on observable directional modes.

This is a diagnostic around the exact b=0 J2/Ludwik baseline.  It computes
the Gauss--Newton quantities (g, H, rho and the predicted improvement) at
selected increments without running an outer optimizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fem_inhouse.core.driven_j2 import DrivenJ2PlaneStressBatch
from fem_inhouse.core.driven_j2_directional import DirectionalDrivenJ2PlaneStressBatch
from fem_inhouse.identification.dic_whitening import DICSpectralTransfer, DICSpectralWhitener
from fem_inhouse.measurement import image_flow_to_canonical
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.workflows.experimental_mechanical_oracle import (
    solve_fixed_plastic_increment_equilibrium,
)

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ROOT / "validation/_generated/performance/experimental_oracle_p43_m20/fields.npz"
NOISE = (
    ROOT / "validation/reference_data/dic_uncertainty_propagation_p0043_v1"
    / "centred_repeat_flow_pixels.npy"
)
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
BASIS = (
    ROOT / "validation/_generated/performance/experimental_oracle_p43_m20"
    / "direction_observability_40states_j2metric_v2/basis.npz"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--h", type=float, default=1.0e-3)
    parser.add_argument("--equilibrium-tolerance", type=float, default=1.0e-9)
    parser.add_argument(
        "--admissible-fraction",
        type=float,
        default=None,
        help=(
            "project Delta p onto [0, fraction * Delta p_max) of the state being "
            "integrated. Off by default. The prescribed history is admissible for its "
            "own trajectory, but the probe moves the flow direction and therefore the "
            "bound, so a Delta p that was admissible for the unperturbed state need "
            "not be for the perturbed one. 0.999 completes all four states; the "
            "projection engages where the unprojected run has no solution at all, and "
            "reproduces states 10 and 20 bit for bit."
        ),
    )
    parser.add_argument("--states", type=int, nargs="+", default=[10, 20, 30, 40])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fields = np.load(FIELDS, allow_pickle=False)
    measured = np.asarray(fields["measured_displacement_history"])
    ludwik = np.asarray(fields["ludwik_increment_history"])
    basis = np.load(BASIS, allow_pickle=False)["basis"][:, : args.rank]
    nx, ny = measured.shape[1] - 1, measured.shape[2] - 1
    grid = StructuredGrid2D(nx, ny, 0.00184 * nx, 0.00184 * ny)
    kinematics = TwoSubcellDiagnostic2D(grid)
    modes = basis.reshape(kinematics.material_point_count, 3, args.rank)
    directional_material = DirectionalDrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        modes,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    baseline_material = DrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )

    noise = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    canonical_noise = image_flow_to_canonical(
        np.asarray(noise[:512, :512]), pixel_size_mm=0.00184
    )
    support = np.ones((*grid.node_shape, 2), dtype=np.float64)
    support[[0, -1], :, :] = 0.0
    support[:, [0, -1], :] = 0.0
    whitener = DICSpectralWhitener.from_stationary_noise_field(
        canonical_noise,
        target_shape=grid.node_shape,
        sample_count=256,
        seed=42,
        remove_spatial_mean=False,
        support_mask=support,
    )
    transfer = DICSpectralTransfer.from_sinusoidal_csv(TRANSFER)
    selected = {state - 1 for state in args.states}
    last_step = max(selected) if selected else len(ludwik) - 1
    displacement = measured[0].copy()
    records: list[dict[str, object]] = []
    replay_failure: dict[str, object] | None = None

    def observed_residual(value: np.ndarray, target: np.ndarray) -> np.ndarray:
        raw = transfer.apply(value) - target
        return whitener.apply(raw)

    def solve_candidate(
        coefficients: np.ndarray,
        increment: np.ndarray,
        initial: np.ndarray,
        target: np.ndarray,
    ):
        directional_material.set_direction_coefficients(coefficients)
        try:
            result = solve_fixed_plastic_increment_equilibrium(
                material=directional_material,
                kinematics=kinematics,
                boundary_displacement=target,
                equivalent_plastic_increment=increment,
                initial_displacement=initial,
                time_increment=1.0,
                equilibrium_rms_tolerance=args.equilibrium_tolerance,
            )
            return result
        finally:
            directional_material.revert()

    def admissible(material, increment: np.ndarray, displacement_field: np.ndarray):
        """Clip Delta p to the admissible set of the baseline state being integrated.

        Only the baseline replay is clipped. The directional prototype prescribes
        its own signed direction basis, so the associated-J2 bound does not
        describe its admissible set, and clipping it there would silently change
        the very sensitivity the probe measures.
        """

        if args.admissible_fraction is None:
            return increment
        strain = np.asarray(kinematics.strain(displacement_field)).reshape(-1, 3)
        bound = material.maximum_admissible_equivalent_plastic_increment(strain)
        return np.minimum(
            increment, (args.admissible_fraction * bound).reshape(np.shape(increment))
        )

    for step, increment in enumerate(ludwik[: last_step + 1]):
        target = measured[step + 1]
        zero = np.zeros(args.rank)
        committed_plastic = baseline_material.committed_plastic_strain
        committed_peeq = baseline_material.committed_equivalent_plastic_strain
        directional_material.set_committed_state(committed_plastic, committed_peeq)
        try:
            baseline_result = solve_fixed_plastic_increment_equilibrium(
                material=baseline_material,
                kinematics=kinematics,
                boundary_displacement=target,
                equivalent_plastic_increment=admissible(
                    baseline_material, increment, displacement
                ),
                initial_displacement=displacement,
                time_increment=1.0,
                equilibrium_rms_tolerance=args.equilibrium_tolerance,
            )
        except Exception as error:
            replay_failure = {
                "state": step + 1,
                "type": type(error).__name__,
                "message": str(error),
                "diagnostics": getattr(error, "diagnostics", None),
            }
            break
        baseline = baseline_result.displacement.copy()
        baseline_eq = float(baseline_result.equilibrium_rms)

        if step in selected:
            directional_zero_result = solve_candidate(
                zero, increment, baseline, target
            )
            directional_zero = directional_zero_result.displacement.copy()
            directional_zero_eq = float(directional_zero_result.equilibrium_rms)
            zero_difference = directional_zero - baseline
            zero_relative_error = float(
                np.linalg.norm(zero_difference)
                / max(np.linalg.norm(baseline), 1.0e-30)
            )
            r0 = observed_residual(baseline, target)
            baseline_trial = baseline_result.linearisation.trial
            directional_trial = directional_zero_result.linearisation.trial
            stress_error = float(
                np.linalg.norm(
                    directional_trial.stress_in_plane_mpa
                    - baseline_trial.stress_in_plane_mpa
                )
                / max(np.linalg.norm(baseline_trial.stress_in_plane_mpa), 1.0e-30)
            )
            plastic_error = float(
                np.linalg.norm(
                    directional_trial.observables["plastic_strain_2d"]
                    - baseline_trial.observables["plastic_strain_2d"]
                )
                / max(
                    np.linalg.norm(baseline_trial.observables["plastic_strain_2d"]),
                    1.0e-30,
                )
            )
            responses = []
            equilibria = []
            for component in range(args.rank):
                plus = zero.copy()
                minus = zero.copy()
                plus[component] += args.h
                minus[component] -= args.h
                plus_result = solve_candidate(plus, increment, baseline, target)
                minus_result = solve_candidate(minus, increment, baseline, target)
                u_plus = plus_result.displacement
                u_minus = minus_result.displacement
                y = (
                    observed_residual(u_plus, target)
                    - observed_residual(u_minus, target)
                ) / (2.0 * args.h)
                responses.append(y)
                equilibria.append(
                    {
                        "plus": float(plus_result.equilibrium_rms),
                        "minus": float(minus_result.equilibrium_rms),
                    }
                )
            response_matrix = np.stack(responses, axis=1)
            flat = response_matrix.reshape(-1, args.rank)
            flat_r0 = r0.ravel()
            count = flat_r0.size
            g = flat.T @ flat_r0 / count
            hessian = flat.T @ flat / count
            inverse = np.linalg.pinv(hessian, rcond=1.0e-12)
            b_gn = -inverse @ g
            delta_j = 0.5 * float(g @ inverse @ g)
            objective_j0 = 0.5 * float(np.vdot(flat_r0, flat_r0).real) / count
            response_norms = np.linalg.norm(flat, axis=0)
            residual_norm = float(np.linalg.norm(flat_r0))
            rho = (flat.T @ flat_r0) / np.maximum(response_norms * residual_norm, 1.0e-300)
            single_mode_fraction = (g * g) / np.maximum(
                2.0 * np.diag(hessian) * objective_j0, 1.0e-300
            )
            records.append(
                {
                    "state": step + 1,
                    "objective_j0": objective_j0,
                    "single_mode_predicted_fraction": single_mode_fraction.tolist(),
                    "single_mode_rho_squared": (rho * rho).tolist(),
                    "equilibrium_rms": baseline_eq,
                    "directional_zero_equilibrium_rms": directional_zero_eq,
                    "directional_zero_relative_error": zero_relative_error,
                    "directional_zero_absolute_error": float(np.max(np.abs(zero_difference))),
                    "directional_zero_stress_relative_error": stress_error,
                    "directional_zero_plastic_strain_relative_error": plastic_error,
                    "directional_zero_objective": 0.5 * float(
                        np.vdot(observed_residual(directional_zero, target),
                                observed_residual(directional_zero, target)).real
                    ) / r0.size,
                    "g": g.tolist(),
                    "H": hessian.tolist(),
                    "rho": rho.tolist(),
                    "b_gn": b_gn.tolist(),
                    "delta_j_gn": delta_j,
                    "finite_difference_step": args.h,
                    "candidate_equilibrium_rms": equilibria,
                }
            )
            print(json.dumps(records[-1]), flush=True)

        # Commit only the accepted b=0 J2/Ludwik baseline, after all candidates
        # have been evaluated from the same n-1 constitutive state.
        baseline_material.commit()
        displacement = baseline

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "states": args.states,
        "rank": args.rank,
        "finite_difference_step": args.h,
        "equilibrium_tolerance": args.equilibrium_tolerance,
        "objective_definition": "0.5 * ||W_D (M_D u - u_DIC)||^2 / N_D",
        "completed_states": [record["state"] for record in records],
        "replay_failure": replay_failure,
        "records": records,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
