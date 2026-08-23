#!/usr/bin/env python3
"""Identify the uncertain SRIX parameters against the experimental plastic power.

Per `validation/srix_power_identification_preregistration.md`: the compiled
SRIX law, integrated pointwise on the reconstructed total-strain paths with
the per-point EBSD orientation, least squares on the logarithms of
`(tau0, R, Q, b, C, d)` against `D_exp = sigma_pred : Delta eps^inel`, with
central finite-difference gradients, fitted on the training states only and
validated leave-one-state-out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize

from fem_inhouse.core.crystal_orientation import rotations_from_euler_bunge_deg
from fem_inhouse.core.fcc_interaction_matrix import SLIP_SYSTEMS
from fem_inhouse.core.mfront_3d import MFront3DMaterialPointBatch
from fem_inhouse.core.mfront_behaviours import MFRONT_BEHAVIOURS
from fem_inhouse.core.srix_parameters import (
    DEFAULT_PARAMETER_SET,
    SRIX_PARAMETER_SETS,
    resolve_srix_parameters,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/_generated/shared_tensor_generator"
LIBRARY = ROOT / "build/mfront/src/libBehaviour.so"
EBSD_PATH = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")
BEHAVIOUR_IDENTIFIER = "fcc_forest_rubin_srix"
PARAMETERS = ("tau0", "overstress_modulus", "q", "b", "c", "d")
ORIGIN = (1580, 1030)
PIXELS = 100
SUBCELLS = 2
N_STATES = 20
HELDOUT = (24, 28, 32, 36, 40)
# Fast-iteration window: the central 20x20 elements of the 100x100 grid.
WINDOW = 20
WINDOW_OFFSET = (40, 40)


def slip_tensors() -> np.ndarray:
    tensors = np.empty((12, 3, 3))
    for index, (burgers, normal) in enumerate(SLIP_SYSTEMS):
        s = np.asarray(burgers, dtype=np.float64)
        m = np.asarray(normal, dtype=np.float64)
        s /= np.linalg.norm(s)
        m /= np.linalg.norm(m)
        tensors[index] = 0.5 * (np.outer(s, m) + np.outer(m, s))
    return tensors


def orientation_maps() -> np.ndarray:
    x0, y0 = ORIGIN
    with h5py.File(EBSD_PATH, "r") as handle:
        phi1 = np.asarray(handle["/orientation/phi1"])[x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1]
        phi = np.asarray(handle["/orientation/Phi"])[x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1]
        phi2 = np.asarray(handle["/orientation/phi2"])[x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1]

    def to_points(field: np.ndarray) -> np.ndarray:
        element_mean = 0.25 * (field[1:, 1:] + field[:-1, :-1] + field[1:, :-1] + field[:-1, 1:])
        return np.repeat(element_mean[:, :, None], SUBCELLS, axis=2).reshape(-1)

    return np.stack([to_points(phi1), to_points(phi), to_points(phi2)], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "srix_power_identification.json")
    parser.add_argument("--maxiter", type=int, default=30)
    arguments = parser.parse_args()

    tr = np.load(OUT / "krylov_trajectories.r16.npz", allow_pickle=False)
    n_states = N_STATES

    def crop_window(array: np.ndarray) -> np.ndarray:
        """Crop the central 20x20 window out of the (state, 100, 100, 2, ...) layout."""

        shaped = array.reshape(n_states, PIXELS, PIXELS, SUBCELLS, -1)
        i0, j0 = WINDOW_OFFSET
        window = shaped[:, i0 : i0 + WINDOW, j0 : j0 + WINDOW, :, :]
        return window.reshape(n_states, -1, array.shape[-1])

    stress = crop_window(tr["stress"])  # (states, points, 3) predictor stress, Kelvin
    eps_elastic = crop_window(tr["eps_elastic"])
    eps_inel = crop_window(tr["eps_inel_observable"])
    d_eps_inel = crop_window(tr["d_eps_inel_observable"])
    n_points = stress.shape[1]

    # Total strain per state, 6-component Kelvin: [xx, yy, zz, sqrt2 yz, sqrt2 xz, sqrt2 xy]
    nu = 0.30
    def zz_elastic(eps_el: np.ndarray, stress_k: np.ndarray) -> np.ndarray:
        xx, yy = eps_el[..., 0], eps_el[..., 1]
        return -(nu / (1.0 - nu)) * (xx + yy)

    total_strain = np.zeros((n_states, n_points, 6))
    for n in range(n_states):
        xx, yy = eps_elastic[n, :, 0], eps_elastic[n, :, 1]
        xy_k = eps_elastic[n, :, 2]
        inel_zz = -(eps_inel[n, :, 0] + eps_inel[n, :, 1])
        el_zz = -(nu / (1.0 - nu)) * (xx + yy)
        total_strain[n, :, 0] = xx + eps_inel[n, :, 0]
        total_strain[n, :, 1] = yy + eps_inel[n, :, 1]
        total_strain[n, :, 2] = el_zz + inel_zz
        total_strain[n, :, 5] = xy_k + eps_inel[n, :, 2]  # sqrt2 xy
        total_strain[n, :, 3] = 0.0
        total_strain[n, :, 4] = 0.0

    # Experimental power per increment (predictor-stress power of the observable increment).
    def inplane_power(sigma_k: np.ndarray, deps_k: np.ndarray) -> np.ndarray:
        return (
            sigma_k[..., 0] * deps_k[..., 0]
            + sigma_k[..., 1] * deps_k[..., 1]
            + sigma_k[..., 2] * deps_k[..., 2]
        )

    d_exp = np.stack(
        [inplane_power(stress[n], d_eps_inel[n]) for n in range(n_states)]
    )  # (states, points)

    # Orientations and the systems in the global frame, on the same window.
    angles = orientation_maps().reshape(PIXELS, PIXELS, SUBCELLS, 3)
    i0, j0 = WINDOW_OFFSET
    angles = angles[i0 : i0 + WINDOW, j0 : j0 + WINDOW, :, :].reshape(-1, 3)
    rotations = rotations_from_euler_bunge_deg(angles)  # Q_global_to_material
    material = slip_tensors()
    material_to_global = np.swapaxes(rotations, 1, 2)
    systems_global = np.einsum(
        "pia,cab,pjb->pijc", material_to_global, material, material_to_global
    )

    default = SRIX_PARAMETER_SETS[DEFAULT_PARAMETER_SET]
    base_props, _ = resolve_srix_parameters(parameter_set=DEFAULT_PARAMETER_SET)
    attributes = ("tau0_mpa", "overstress_modulus_mpa", "q_mpa", "b", "c_mpa", "d")
    theta0 = np.log(np.asarray([getattr(default, attr) for attr in attributes], dtype=np.float64))

    spec = MFRONT_BEHAVIOURS.get(BEHAVIOUR_IDENTIFIER)

    def integrate(theta_log: np.ndarray) -> np.ndarray:
        """Run the law on every point, return the SRIX power per increment."""

        values = np.exp(theta_log)
        key_map = ("tau0", "SrixOverstressModulus", "Q", "b", "C", "d")
        behaviour_parameters = dict(base_props)
        for index, key in enumerate(key_map):
            behaviour_parameters[key] = float(values[index])
        batch = MFront3DMaterialPointBatch(
            LIBRARY,
            behaviour_spec=spec,
            point_count=n_points,
            behaviour_parameters=behaviour_parameters,
            rotation_global_to_material=rotations,
            behaviour_name=spec.tridimensional_behaviour,
        )
        powers = np.zeros((n_states, n_points))
        failures = 0
        previous_slip = np.zeros((n_points, 12))
        previous_strain = np.zeros((n_points, 6))
        for n in range(n_states):
            try:
                trial = batch.evaluate(
                    total_strain[n], time_increment=1.0, collect_observables=True
                )
                batch.commit()
            except Exception:
                # Uniform substepping: the reconstructed path can exceed what
                # one step of the local Newton admits (already true at the
                # default parameters beyond state 26). Each successful
                # sub-step commits immediately; a failure abandons the state
                # and the objective penalises it.
                trial = None
                for sub in range(1, 5):
                    try:
                        trial = batch.evaluate(
                            previous_strain + (sub / 4.0) * (total_strain[n] - previous_strain),
                            time_increment=0.25,
                            collect_observables=True,
                        )
                        batch.commit()
                    except Exception:
                        trial = None
                        break
                if trial is None:
                    failures += 1
                    previous_strain = total_strain[n].copy()
                    continue
            previous_strain = total_strain[n].copy()
            slip = np.asarray(trial.observables["equivalent_plastic_slip"])  # (points, 12)
            delta_slip = slip - previous_slip
            previous_slip = slip.copy()
            sigma6 = np.asarray(trial.stress_kelvin_mpa)  # (points, 6)
            s3 = np.zeros((n_points, 3, 3))
            s3[:, 0, 0] = sigma6[:, 0]
            s3[:, 1, 1] = sigma6[:, 1]
            s3[:, 2, 2] = sigma6[:, 2]
            s3[:, 0, 1] = s3[:, 1, 0] = sigma6[:, 5] / np.sqrt(2.0)
            s3[:, 0, 2] = s3[:, 2, 0] = sigma6[:, 4] / np.sqrt(2.0)
            s3[:, 1, 2] = s3[:, 2, 1] = sigma6[:, 3] / np.sqrt(2.0)
            tau = np.einsum("pij,pijc->pc", s3, systems_global)
            powers[n] = np.sum(tau * delta_slip, axis=1)
        return powers, failures

    def objective(theta_log: np.ndarray, mask_states: list[int]) -> float:
        powers, failures = integrate(theta_log)
        mask = np.zeros(n_states, dtype=bool)
        mask[mask_states] = True
        num = np.sum((powers[mask] - d_exp[mask]) ** 2)
        den = max(np.sum(d_exp[mask] ** 2), 1e-300)
        return float(num / den + 10.0 * failures)

    training = [n for n in range(n_states) if n not in [s - 21 for s in HELDOUT]]
    heldout = [s - 21 for s in HELDOUT]

    def fd_gradient(theta_log: np.ndarray, mask_states: list[int]) -> np.ndarray:
        gradient = np.zeros_like(theta_log)
        for index in range(len(theta_log)):
            step = 1e-4 * max(1.0, abs(theta_log[index]))
            plus = theta_log.copy()
            minus = theta_log.copy()
            plus[index] += step
            minus[index] -= step
            gradient[index] = (objective(plus, mask_states) - objective(minus, mask_states)) / (
                2 * step
            )
        return gradient

    bounds = [(value - 1.5, value + 1.5) for value in theta0]
    result = minimize(
        objective,
        theta0,
        args=(training,),
        method="L-BFGS-B",
        jac=fd_gradient,
        bounds=bounds,
        options={"maxiter": arguments.maxiter, "ftol": 1e-12, "gtol": 1e-8},
    )
    fitted = np.exp(result.x)
    names = ("tau0", "R", "Q", "b", "C", "d")
    defaults = np.asarray([getattr(default, attr) for attr in attributes])
    fitted_powers, final_failures = integrate(result.x)
    heldout_r2 = {}
    for s in heldout:
        num = np.sum((fitted_powers[s] - d_exp[s]) ** 2)
        den = max(np.sum((d_exp[s] - np.mean(d_exp[s])) ** 2), 1e-300)
        heldout_r2[int(s) + 21] = 1.0 - num / den
    mean_r2 = float(np.mean(list(heldout_r2.values())))
    payload = {
        "schema_version": 1,
        "bars": {"explains": 0.30, "partial": 0.10},
        "fitted_parameters": {
            name: float(value) for name, value in zip(names, fitted, strict=True)
        },
        "default_parameters": {
            name: float(value) for name, value in zip(names, defaults, strict=True)
        },
        "ratios_to_default": {name: float(fitted[i] / defaults[i]) for i, name in enumerate(names)},
        "optimization_success": bool(result.success),
        "final_objective": float(result.fun),
        "integration_failures_at_optimum": int(final_failures),
        "heldout_r2_per_state": heldout_r2,
        "heldout_r2_mean": mean_r2,
        "reading": "explains" if mean_r2 >= 0.30 else "partial" if mean_r2 >= 0.10 else "negative",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
