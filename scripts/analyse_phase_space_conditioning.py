#!/usr/bin/env python3
"""Continuous conditioning of the response by the local state, held out.

Per `validation/phase_space_conditioning_preregistration.md`: a k-NN local
estimator (k=50) over standardised features, evaluated with leave-one-state-out
(the strong test) and a random point split (the weak reference), on the
amplitude `log Delta p` and the wrapped flow direction
`Delta theta = wrap(theta_n - theta_s)` (J2 gives exactly 0), across the
feature ladder A -> B -> C -> D.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

from fem_inhouse.core.kelvin import PLANE_STRESS_PLASTIC_GAUGE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/_generated/shared_tensor_generator"
EBSD_PATH = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")
GAUGE = np.asarray(PLANE_STRESS_PLASTIC_GAUGE, dtype=np.float64)
ORIGIN = (1580, 1030)
PIXELS = 100
SUBCELLS = 2
K = 50


def deviator(stress: np.ndarray) -> np.ndarray:
    pressure = (stress[:, 0] + stress[:, 1]) / 3.0
    result = np.empty_like(stress)
    result[:, 0] = stress[:, 0] - pressure
    result[:, 1] = stress[:, 1] - pressure
    result[:, 2] = stress[:, 2]
    return result


def gauge_norm(field: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.einsum("pi,ij,pj->p", field, GAUGE, field), 0.0))


def orientation_features() -> np.ndarray:
    x0, y0 = ORIGIN
    with h5py.File(EBSD_PATH, "r") as handle:
        phi1 = np.asarray(handle["/orientation/phi1"])[x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1]
        phi = np.asarray(handle["/orientation/Phi"])[x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1]
        phi2 = np.asarray(handle["/orientation/phi2"])[x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1]
        schmid = np.asarray(handle["/schmid/max_schmid_factor"])[
            x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1
        ]

    def to_points(field: np.ndarray) -> np.ndarray:
        element_mean = 0.25 * (
            field[1:, 1:] + field[:-1, :-1] + field[1:, :-1] + field[:-1, 1:]
        )
        return np.repeat(element_mean[:, :, None], SUBCELLS, axis=2).reshape(-1)

    return np.stack(
        [to_points(phi1), to_points(phi), to_points(phi2), to_points(schmid)], axis=1
    )


def circular_mean(angles: np.ndarray, axis: int | None = None) -> np.ndarray:
    return np.arctan2(np.mean(np.sin(angles), axis=axis), np.mean(np.cos(angles), axis=axis))


def circular_r2(residuals: np.ndarray, targets: np.ndarray) -> float:
    residual_msr = 1.0 - abs(np.mean(np.exp(1j * residuals)))
    global_msr = 1.0 - abs(np.mean(np.exp(1j * targets)))
    return 1.0 - residual_msr / max(global_msr, 1e-300)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "phase_conditioning.json")
    arguments = parser.parse_args()

    tr = np.load(OUT / "krylov_trajectories.r16.npz", allow_pickle=False)
    n_states, n_points, _ = tr["stress"].shape
    stress = tr["stress"].reshape(-1, 3)
    d_eps = tr["d_eps_inel_observable"].reshape(-1, 3)
    eps_inel = tr["eps_inel_observable"].reshape(-1, 3)
    states = np.repeat(np.arange(n_states), n_points)
    orientation = np.tile(orientation_features(), (n_states, 1))

    s = deviator(stress)
    s_norm = np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2 + (s[:, 0] + s[:, 1]) ** 2)
    sigma_eq = np.sqrt(1.5) * s_norm
    p = -(stress[:, 0] + stress[:, 1]) / 3.0
    angle_s = np.arctan2(s[:, 2], (s[:, 0] - s[:, 1]) / np.sqrt(2.0))
    p_eq = gauge_norm(eps_inel)
    dp = gauge_norm(d_eps)
    n_dir = deviator(d_eps)
    theta_n = np.arctan2(n_dir[:, 2], (n_dir[:, 0] - n_dir[:, 1]) / np.sqrt(2.0))
    delta_theta = np.angle(np.exp(1j * (theta_n - angle_s)))

    loaded = s_norm > 1e-2 * s_norm.max()
    active = loaded & (dp > 1e-6 * np.quantile(dp[dp > 0], 0.9))
    print(f"active samples: {int(active.sum())} of {len(active)}", flush=True)

    phi1, phi, phi2, schmid = orientation.T
    ladders = {
        "A": np.stack([sigma_eq, p_eq], axis=1),
        "B": np.stack([sigma_eq, p_eq, p, np.sin(angle_s), np.cos(angle_s)], axis=1),
        "C": np.stack(
            [sigma_eq, p_eq, p, np.sin(angle_s), np.cos(angle_s), schmid], axis=1
        ),
        "D": np.stack(
            [sigma_eq, p_eq, p, np.sin(angle_s), np.cos(angle_s), schmid,
             np.sin(phi1), np.cos(phi1), np.sin(phi), np.cos(phi), np.sin(phi2), np.cos(phi2)],
            axis=1,
        ),
    }
    log_dp = np.log(dp)
    report: dict[str, dict] = {}

    def evaluate(features: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray) -> dict:
        scaler = StandardScaler().fit(features[train_mask])
        tree = cKDTree(scaler.transform(features[train_mask]))
        distances, indices = tree.query(scaler.transform(features[test_mask]), k=K)
        neighbour_log = log_dp[train_mask][indices]
        amplitude_pred = neighbour_log.mean(axis=1)
        residual = log_dp[test_mask] - amplitude_pred
        r2 = 1.0 - float(np.sum(residual**2)) / max(
            float(np.sum((log_dp[test_mask] - np.mean(log_dp[train_mask])) ** 2)), 1e-300
        )
        neighbour_angles = delta_theta[train_mask][indices]
        direction_pred = circular_mean(neighbour_angles, axis=1)
        residuals_angle = np.angle(np.exp(1j * (delta_theta[test_mask] - direction_pred)))
        r2_circ = circular_r2(residuals_angle, delta_theta[test_mask])
        mae_deg = float(np.degrees(np.mean(np.abs(residuals_angle))))
        return {
            "r2_log_dp": float(r2),
            "r2_circular_direction": float(r2_circ),
            "direction_mae_degrees": mae_deg,
        }

    # Weak reference: random point split.
    rng = np.random.default_rng(20260817)
    split = rng.random(int(active.sum())) < 0.5
    active_indices = np.where(active)[0]
    train_idx = active_indices[split]
    test_idx = active_indices[~split]
    for name, features in ladders.items():
        report[f"random_{name}"] = evaluate(features, train_idx, test_idx)

    # Strong test: leave-one-state-out.
    for name, features in ladders.items():
        loso = []
        for state in range(n_states):
            test_idx = np.where(active & (states == state))[0]
            train_idx = np.where(active & (states != state))[0]
            loso.append(evaluate(features, train_idx, test_idx))
        report[f"loso_{name}"] = {
            key: float(np.mean([row[key] for row in loso])) for key in loso[0]
        }
        report[f"loso_{name}"]["per_state_r2_log_dp"] = [row["r2_log_dp"] for row in loso]
        report[f"loso_{name}"]["per_state_r2_circ"] = [
            row["r2_circular_direction"] for row in loso
        ]
        print(
            f"loso {name}: R2_amp {report[f'loso_{name}']['r2_log_dp']:.3f}  "
            f"R2_circ {report[f'loso_{name}']['r2_circular_direction']:.3f}  "
            f"MAE {report[f'loso_{name}']['direction_mae_degrees']:.1f} deg",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "k": K,
        "active_samples": int(active.sum()),
        "bars": {"amplitude": 0.5, "circular_direction": 0.5},
        "results": report,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2, sort_keys=True))

    # Direction panels, per the registered outputs: wrapped Delta theta against
    # sigma_eq for each p_eq quantile, and against the max Schmid factor.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quantile = np.digitize(p_eq, np.quantile(p_eq, np.linspace(0, 1, 5))[1:-1])
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    for ax, q in zip(axes.ravel(), range(4), strict=True):
        mask = active & (quantile == q)
        hb = ax.hexbin(
            sigma_eq[mask], delta_theta[mask], gridsize=90, bins="log",
            cmap="viridis", extent=(sigma_eq.min(), sigma_eq.max(), -np.pi, np.pi),
            linewidths=0,
        )
        ax.axhline(0.0, color="white", lw=0.8, ls="--")
        ax.set_title(f"p_eq quantile {q + 1}")
        if q in (2, 3):
            ax.set_xlabel(r"$\sigma_{\rm eq}$ [MPa]")
        if q in (0, 2):
            ax.set_ylabel(r"$\Delta\theta = \theta_n - \theta_s$ [rad]")
    fig.suptitle(r"wrapped flow angle vs $\sigma_{\rm eq}$ (J2 would sit on the dashed line)")
    fig.tight_layout()
    fig.savefig(OUT / "phase_geometry_direction_dtheta.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    hb = ax.hexbin(
        schmid[active], delta_theta[active], gridsize=80, bins="log",
        cmap="viridis", extent=(schmid.min(), schmid.max(), -np.pi, np.pi),
        linewidths=0,
    )
    ax.axhline(0.0, color="white", lw=0.8, ls="--")
    ax.set_title(r"wrapped flow angle vs max Schmid factor")
    ax.set_xlabel("max Schmid factor")
    ax.set_ylabel(r"$\Delta\theta$ [rad]")
    fig.colorbar(hb, ax=ax, label="log count")
    fig.tight_layout()
    fig.savefig(OUT / "phase_geometry_dtheta_schmid.png")
    plt.close(fig)
    print("saved direction panels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
