#!/usr/bin/env python3
"""First nonlinear directional oracle pilot for P43 M20."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from fem_inhouse.core.constitutive import PLANE_STRESS_VON_MISES_METRIC, von_mises
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
    ROOT
    / "validation/reference_data/dic_uncertainty_propagation_p0043_v1"
    / "centred_repeat_flow_pixels.npy"
)
TRANSFER = ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
BASIS = (
    ROOT
    / "validation/_generated/performance/experimental_oracle_p43_m20"
    / "direction_observability_40states_j2metric_v2/basis.npz"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, choices=(1, 2, 3, 5), default=2)
    parser.add_argument("--increments", type=int, default=1)
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
    material = DirectionalDrivenJ2PlaneStressBatch(
        kinematics.material_point_count,
        modes,
        young_modulus_mpa=205_000.0,
        poisson_ratio=0.30,
    )
    noise = np.load(NOISE, mmap_mode="r", allow_pickle=False)
    canonical_noise = image_flow_to_canonical(np.asarray(noise[:512, :512]), pixel_size_mm=0.00184)
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
    displacement = measured[0].copy()
    displacement_history = [displacement.copy()]
    coefficient_history = []
    objective_history = []
    objective_scale = 1.0e6

    def objective(coefficients: np.ndarray, increment: np.ndarray, initial: np.ndarray) -> float:
        material.set_direction_coefficients(coefficients)
        try:
            equilibrium = solve_fixed_plastic_increment_equilibrium(
                material=material,
                kinematics=kinematics,
                boundary_displacement=measured[step + 1],
                equivalent_plastic_increment=increment,
                initial_displacement=initial,
                time_increment=1.0,
                equilibrium_rms_tolerance=1.0e-6,
            )
            observed = transfer.apply(equilibrium.displacement)
            residual = observed - measured[step + 1]
            return objective_scale * float(
                whitener.quadratic_misfit(residual) / residual.size
            )
        except Exception:
            material.revert()
            return 1.0e6 + float(np.dot(coefficients, coefficients))
        finally:
            material.revert()

    def objective_and_gradient(
        coefficients: np.ndarray, increment: np.ndarray, initial: np.ndarray
    ) -> tuple[float, np.ndarray]:
        value = objective(coefficients, increment, initial)
        gradient = np.zeros_like(coefficients)
        step = 1.0e-3
        for component in range(coefficients.size):
            plus = coefficients.copy()
            minus = coefficients.copy()
            plus[component] += step
            minus[component] -= step
            gradient[component] = (
                objective(plus, increment, initial) - objective(minus, increment, initial)
            ) / (2.0 * step)
        return value, gradient

    for step in range(min(args.increments, ludwik.shape[0])):
        coefficients = np.zeros(args.rank)
        increment = ludwik[step].copy()
        initial = displacement.copy()
        result = minimize(
            objective_and_gradient,
            coefficients,
            args=(increment, initial),
            method="L-BFGS-B",
            jac=True,
            bounds=[(-0.5, 0.5)] * args.rank,
            options={"maxiter": 20, "ftol": 1.0e-12, "gtol": 1.0e-8},
        )
        material.set_direction_coefficients(result.x)
        equilibrium = solve_fixed_plastic_increment_equilibrium(
            material=material,
            kinematics=kinematics,
            boundary_displacement=measured[step + 1],
            equivalent_plastic_increment=increment,
            initial_displacement=initial,
            time_increment=1.0,
            equilibrium_rms_tolerance=1.0e-6,
        )
        displacement = equilibrium.displacement.copy()
        material.commit()
        flow = equilibrium.linearisation.trial.observables["flow_direction"]
        stress = equilibrium.linearisation.trial.stress_in_plane_mpa
        n_j2 = np.einsum("ij,pj->pi", PLANE_STRESS_VON_MISES_METRIC, stress)
        n_j2 /= np.maximum(von_mises(stress), 1.0e-14)[:, None]
        inverse_metric = np.linalg.inv(PLANE_STRESS_VON_MISES_METRIC)
        cosine = np.einsum("pi,ij,pj->p", n_j2, inverse_metric, flow)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        active = increment.ravel() > 1.0e-14
        active_angles = angle[active]
        coefficient_history.append(result.x.copy())
        objective_history.append(float(result.fun) / objective_scale)
        displacement_history.append(displacement.copy())
        print(
            json.dumps(
                {
                    "increment": step + 1,
                    "success": bool(result.success),
                    "objective": float(result.fun) / objective_scale,
                    "coefficients": result.x.tolist(),
                    "gradient": np.asarray(result.jac).tolist(),
                    "angle_rms": float(np.sqrt(np.mean(active_angles**2)))
                    if np.any(active)
                    else 0.0,
                    "angle_p95": float(np.quantile(active_angles, 0.95)) if np.any(active) else 0.0,
                    "equilibrium_rms": float(equilibrium.equilibrium_rms),
                }
            ),
            flush=True,
        )

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "fields.npz",
        displacement_history=np.asarray(displacement_history),
        coefficients=np.asarray(coefficient_history),
        objective=np.asarray(objective_history),
    )
    (output / "report.json").write_text(
        json.dumps(
            {
                "rank": args.rank,
                "increments": len(coefficient_history),
                "coefficients": np.asarray(coefficient_history).tolist(),
                "objective": objective_history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
