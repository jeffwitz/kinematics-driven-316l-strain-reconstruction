#!/usr/bin/env python3
"""FCC slip decomposition of the effective inelastic increments.

Per `validation/fcc_slip_decomposition_preregistration.md`: resolve the
predictor stress onto the twelve octahedral systems in the specimen frame
(Bunge rotations from the EBSD map, the repo's own slip-system order), and
decompose each observable-projected inelastic increment onto them — the
unconstrained least squares as the representability ceiling, and two
projected-FISTA variants (diffuse L2, sparse-favouring L1) under the
per-system dissipation constraint — then the per-system k-NN conditioning
on a subsample.

Chunked and vectorised over the 400 000 points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

from fem_inhouse.core.crystal_orientation import rotations_from_euler_bunge_deg
from fem_inhouse.core.fcc_interaction_matrix import SLIP_SYSTEMS
from fem_inhouse.core.kelvin import PLANE_STRESS_PLASTIC_GAUGE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/_generated/shared_tensor_generator"
EBSD_PATH = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")
ORIGIN = (1580, 1030)
PIXELS = 100
SUBCELLS = 2
N_SYSTEMS = 12
FISTA_ITERATIONS = 200
LAMBDA_SPARSE = 1e-6
K = 50
SUBSAMPLE = 20000
CHUNK = 100000

GAUGE = np.asarray(PLANE_STRESS_PLASTIC_GAUGE, dtype=np.float64)


def slip_tensors(specimen_frame: bool = False) -> np.ndarray:
    """The twelve Schmid tensors, normalised, in the material frame."""

    tensors = np.empty((N_SYSTEMS, 3, 3))
    for index, (burgers, normal) in enumerate(SLIP_SYSTEMS):
        s = np.asarray(burgers, dtype=np.float64)
        m = np.asarray(normal, dtype=np.float64)
        s /= np.linalg.norm(s)
        m /= np.linalg.norm(m)
        tensors[index] = 0.5 * (np.outer(s, m) + np.outer(m, s))
    return tensors


def orientation_maps() -> np.ndarray:
    """Bunge angles per material point on the campaign window."""

    x0, y0 = ORIGIN
    with h5py.File(EBSD_PATH, "r") as handle:
        phi1 = np.asarray(handle["/orientation/phi1"])[x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1]
        phi = np.asarray(handle["/orientation/Phi"])[x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1]
        phi2 = np.asarray(handle["/orientation/phi2"])[x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1]

    def to_points(field: np.ndarray) -> np.ndarray:
        element_mean = 0.25 * (
            field[1:, 1:] + field[:-1, :-1] + field[1:, :-1] + field[:-1, 1:]
        )
        return np.repeat(element_mean[:, :, None], SUBCELLS, axis=2).reshape(-1)

    return np.stack([to_points(phi1), to_points(phi), to_points(phi2)], axis=1)


def gauge_norm_3d(field: np.ndarray) -> np.ndarray:
    """Plastic-gauge norm of 3-D deviatoric fields `sqrt((2/3) tr(f^2))`."""

    return np.sqrt((2.0 / 3.0) * np.einsum("pij,pij->p", field, field))


def increment_3d(d_eps: np.ndarray) -> np.ndarray:
    """In-plane Kelvin increment -> full 3-D tensor with the zz closure."""

    xx, yy, xy_kelvin = d_eps[:, 0], d_eps[:, 1], d_eps[:, 2]
    xy = xy_kelvin / np.sqrt(2.0)
    result = np.zeros((d_eps.shape[0], 3, 3))
    result[:, 0, 0] = xx
    result[:, 1, 1] = yy
    result[:, 2, 2] = -(xx + yy)  # plastic-incompressibility closure
    result[:, 0, 1] = xy
    result[:, 1, 0] = xy
    return result


def stress_3d(stress: np.ndarray) -> np.ndarray:
    """In-plane Kelvin stress -> full 3-D tensor (zz = xz = yz = 0)."""

    result = np.zeros((stress.shape[0], 3, 3))
    result[:, 0, 0] = stress[:, 0]
    result[:, 1, 1] = stress[:, 1]
    result[:, 0, 1] = stress[:, 2] / np.sqrt(2.0)
    result[:, 1, 0] = stress[:, 2] / np.sqrt(2.0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "fcc_slip_decomposition.json")
    parser.add_argument(
        "--save-fields",
        type=Path,
        default=None,
        help="save tau, gamma (constrained L2), cumulative Gamma and the "
        "state indices for the slip-geometry figures",
    )
    arguments = parser.parse_args()

    tr = np.load(OUT / "krylov_trajectories.r16.npz", allow_pickle=False)
    n_states, n_points, _ = tr["stress"].shape
    stress = tr["stress"].reshape(-1, 3)
    d_eps = tr["d_eps_inel_observable"].reshape(-1, 3)
    d_eps_raw = tr["d_eps_inel"].reshape(-1, 3)
    states = np.repeat(np.arange(n_states), n_points)

    material = slip_tensors()  # (12, 3, 3)
    angles = orientation_maps()  # (n_points, 3) Bunge degrees
    rotations = rotations_from_euler_bunge_deg(angles)  # Q_global_to_material (n_points, 3, 3)
    material_to_global = np.swapaxes(rotations, 1, 2)
    # P^alpha in the specimen frame: Q^T P_m Q, one tensor per system,
    # tiled across states to align with the (state, point) trajectories.
    systems_global = np.einsum(
        "pia,cab,pjb->pijc", material_to_global, material, material_to_global
    )
    systems_global = np.tile(systems_global, (n_states, 1, 1, 1))
    # tau^alpha = sigma : P^alpha (full 3-D stress, zz components zero)
    sigma3 = stress_3d(stress)
    tau = np.einsum("pij,pijc->pc", sigma3, systems_global)

    d3 = increment_3d(d_eps)
    d3_raw = increment_3d(d_eps_raw)
    norms = gauge_norm_3d(d3)
    active = norms > 1e-6 * np.quantile(norms[norms > 0], 0.9)
    print(f"active points: {int(active.sum())} of {len(norms)}", flush=True)

    # Gram and RHS of the unconstrained problem, per point.
    gram = np.einsum("pija,pijb->pab", systems_global, systems_global)
    rhs = np.einsum("pij,pijc->pc", d3, systems_global)
    gram_raw_rhs = np.einsum("pij,pijc->pc", d3_raw, systems_global)

    # Unconstrained least squares (representability ceiling). The 12 systems
    # span only the 5-D deviatoric space, so the Gram is rank-deficient; a
    # 1e-12 ridge makes the batched solve well-posed without moving the ceiling.
    ridge = 1e-12 * np.einsum("pab->p", gram)[:, None, None] / N_SYSTEMS * np.eye(N_SYSTEMS)
    ceiling = np.linalg.solve(gram + ridge, rhs[..., None])[..., 0]

    def fista(
        objective_gram: np.ndarray, objective_rhs: np.ndarray,
        objective_tau: np.ndarray, lam: float,
    ) -> np.ndarray:
        """Projected FISTA: sign cone per tau, optional L1 sparsity term."""

        gamma = np.zeros_like(objective_rhs)
        gamma_prev = gamma.copy()
        steps = np.einsum("pab->p", objective_gram)
        for _iteration in range(FISTA_ITERATIONS):
            momentum = gamma + 0.5 * (gamma - gamma_prev)
            gradient = np.einsum("pab,pb->pa", objective_gram, momentum) - objective_rhs
            step = 1.0 / np.maximum(steps, 1e-300)
            proposal = momentum - step[:, None] * gradient
            if lam > 0.0:
                soft = np.abs(proposal) - lam * step[:, None]
                proposal = np.sign(proposal) * np.maximum(soft, 0.0)
            # sign cone: sign(gamma^alpha) = sign(tau^alpha) or zero
            positive = objective_tau >= 0.0
            proposal = np.where(positive, np.maximum(proposal, 0.0), np.minimum(proposal, 0.0))
            gamma_prev, gamma = gamma, proposal
        return gamma

    # Constrained variants, chunked to bound memory.
    def solve_chunked(objective_rhs: np.ndarray, lam: float) -> np.ndarray:
        result = np.empty_like(objective_rhs)
        for start in range(0, objective_rhs.shape[0], CHUNK):
            stop = min(start + CHUNK, objective_rhs.shape[0])
            result[start:stop] = fista(
                gram[start:stop], objective_rhs[start:stop], tau[start:stop], lam
            )
        return result

    print("solving the constrained variants (200 FISTA iterations each)...", flush=True)
    gamma_l2 = solve_chunked(rhs, 0.0)
    scale = float(np.median(np.abs(ceiling)[active]))
    gamma_l1 = solve_chunked(rhs, LAMBDA_SPARSE * max(scale, 1e-300))

    # Third admissible decomposition: causally time-regularised. The penalty
    # (gamma_n - gamma_{n-1})^2 ties each increment to its predecessor, the
    # previous solution already computed -- a gauge variant, not a law change.
    LAMBDA_TIME = 1e-2

    def fista_temporal(
        objective_gram: np.ndarray, objective_rhs: np.ndarray,
        objective_tau: np.ndarray, previous: np.ndarray, lam_t: float,
    ) -> np.ndarray:
        gamma = previous.copy()
        gamma_prev = gamma.copy()
        steps = np.einsum("pab->p", objective_gram)
        for _iteration in range(FISTA_ITERATIONS):
            momentum = gamma + 0.5 * (gamma - gamma_prev)
            gradient = np.einsum("pab,pb->pa", objective_gram, momentum) - objective_rhs
            gradient = gradient + lam_t * (momentum - previous)
            step = 1.0 / np.maximum(steps + lam_t, 1e-300)
            proposal = momentum - step[:, None] * gradient
            positive = objective_tau >= 0.0
            proposal = np.where(positive, np.maximum(proposal, 0.0), np.minimum(proposal, 0.0))
            gamma_prev, gamma = gamma, proposal
        return gamma

    print("solving the time-regularised variant (state by state)...", flush=True)
    gamma_time = np.empty_like(rhs)
    previous = np.zeros((n_points, N_SYSTEMS))
    for state in range(n_states):
        mask = np.arange(state * n_points, (state + 1) * n_points)
        previous = fista_temporal(
            gram[mask], rhs[mask], tau[mask], previous, LAMBDA_TIME
        )
        gamma_time[mask] = previous
        print(f"  state {state + 1}/{n_states} done", flush=True)

    def represented(gamma: np.ndarray) -> np.ndarray:
        return np.einsum("pc,pijc->pij", gamma, systems_global)

    def metrics(d3_field: np.ndarray, gamma: np.ndarray) -> dict:
        rep = represented(gamma)
        residual = d3_field - rep
        e_fcc = gauge_norm_3d(residual) / np.maximum(gauge_norm_3d(d3_field), 1e-300)
        rho = gauge_norm_3d(rep) / np.maximum(gauge_norm_3d(d3_field), 1e-300)
        work = tau * gamma
        shares = np.abs(work).sum(axis=0) / max(np.abs(work).sum(), 1e-300)
        return {
            "e_fcc_median": float(np.median(e_fcc[active])),
            "e_fcc_q25": float(np.quantile(e_fcc[active], 0.25)),
            "e_fcc_q75": float(np.quantile(e_fcc[active], 0.75)),
            "rho_median": float(np.median(rho[active])),
            "negative_work_share": float(np.minimum(work, 0).sum() / max(np.abs(work).sum(), 1e-300)),
            "per_system_work_share": shares.tolist(),
            "active_fraction": float(np.mean(np.abs(gamma[active]).sum(axis=1) > 1e-12)),
        }

    unconstrained = metrics(d3, ceiling)
    l2 = metrics(d3, gamma_l2)
    l1 = metrics(d3, gamma_l1)
    temporal = metrics(d3, gamma_time)
    raw_field = metrics(d3_raw, ceiling)

    if arguments.save_fields is not None:
        cumulative_gamma = np.cumsum(
            np.abs(gamma_l2).reshape(n_states, n_points, N_SYSTEMS), axis=0
        ).reshape(-1, N_SYSTEMS)
        cumulative_time = np.cumsum(
            np.abs(gamma_time).reshape(n_states, n_points, N_SYSTEMS), axis=0
        ).reshape(-1, N_SYSTEMS)
        np.savez_compressed(
            arguments.save_fields,
            tau=tau,
            gamma=gamma_l2,
            gamma_cumulative=cumulative_gamma,
            gamma_temporal=gamma_time,
            gamma_temporal_cumulative=cumulative_time,
            states=states,
        )
        print(f"saved slip fields: {arguments.save_fields}", flush=True)

    rep_l2 = represented(gamma_l2)
    rep_l1 = represented(gamma_l1)
    correlation = float(
        np.corrcoef(
            gauge_norm_3d(rep_l2)[active], gauge_norm_3d(rep_l1)[active]
        )[0, 1]
    )
    print(
        f"ceiling: e {unconstrained['e_fcc_median']:.3f} rho {unconstrained['rho_median']:.3f}  "
        f"L2: e {l2['e_fcc_median']:.3f} rho {l2['rho_median']:.3f}  "
        f"L1: e {l1['e_fcc_median']:.3f} rho {l1['rho_median']:.3f}  "
        f"variant corr {correlation:.3f}",
        flush=True,
    )

    # Per-system conditioning on a subsample: (tau^alpha, Gamma^alpha) -> d_gamma^alpha
    rng = np.random.default_rng(20260817)
    sub = rng.choice(np.where(active)[0], size=SUBSAMPLE, replace=False)
    sub_states = states[sub]
    gamma_use = gamma_l2
    cumulative = np.cumsum(np.abs(gamma_use).reshape(n_states, n_points, N_SYSTEMS), axis=0)
    system_r2: dict[str, float] = {}
    per_system = {}
    for alpha in range(N_SYSTEMS):
        tau_a = tau[sub, alpha]
        gamma_a = gamma_use[sub, alpha]
        gamma_hist = cumulative[sub_states, sub % n_points, alpha] - np.abs(gamma_a)
        significant = np.abs(tau_a) > 1.0
        if significant.sum() < 2000:
            continue
        features = np.stack([tau_a, gamma_hist], axis=1)
        scores = []
        for state in range(n_states):
            test = (sub_states == state) & significant
            train = (sub_states != state) & significant
            if test.sum() < 100 or train.sum() < 100:
                continue
            scaler = StandardScaler().fit(features[train])
            tree = cKDTree(scaler.transform(features[train]))
            _, idx = tree.query(scaler.transform(features[test]), k=K)
            prediction = gamma_a[train][idx].mean(axis=1)
            residual = gamma_a[test] - prediction
            scores.append(
                1.0
                - float(np.sum(residual**2))
                / max(float(np.sum((gamma_a[test] - gamma_a[train].mean()) ** 2)), 1e-300)
            )
        r2 = float(np.mean(scores))
        per_system[alpha] = {
            "r2": r2,
            "n_significant": int(significant.sum()),
            "median_activity": float(np.median(np.abs(gamma_a[significant]))),
            "work_share": l2["per_system_work_share"][alpha],
        }
        system_r2[str(alpha)] = r2
        print(f"  system {alpha:2d}: R2 {r2:+.3f}  n {per_system[alpha]['n_significant']}", flush=True)

    payload = {
        "schema_version": 1,
        "bars": {"e_fcc": 0.5, "rho": 0.5, "variant_correlation": 0.9, "slip_r2": 0.5},
        "unconstrained": unconstrained,
        "constrained_l2": l2,
        "constrained_l1": l1,
        "constrained_temporal": temporal,
        "raw_field_unconstrained": raw_field,
        "variant_correlation": correlation,
        "per_system_conditioning": per_system,
        "weighted_system_r2": float(
            np.average(
                [per_system[a]["r2"] for a in per_system],
                weights=[per_system[a]["work_share"] for a in per_system],
            )
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "per_system_conditioning"},
                     indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
