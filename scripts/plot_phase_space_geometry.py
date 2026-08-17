#!/usr/bin/env python3
"""Plot the raw geometry of the admissible constitutive experiments.

The question is visual and precedes every tool: what geometry do the
400 000 local experiments draw in constitutive space — a curve, several
branches, clusters, or a shapeless cloud? No clustering, no regression.

All quantities are the observable-projected ones of the predictive
projected-Krylov line at r=16 (the kernel of the displacement operator is
removed first, per the registered pipeline):

* `sigma_eq` — von Mises equivalent of the predictor stress;
* `Delta p`  — plastic-gauge norm of the inelastic increment;
* `p_eq`    — plastic-gauge norm of the cumulative inelastic field;
* the deviatoric direction angles of the stress and of the increment.

Magnitude colorings use one sequential ramp (viridis); the 2-D direction
histogram is a density map. Plots are saved beside the phase-space
artifacts.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fem_inhouse.core.kelvin import PLANE_STRESS_PLASTIC_GAUGE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/_generated/shared_tensor_generator"
EBSD_PATH = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")
GAUGE = np.asarray(PLANE_STRESS_PLASTIC_GAUGE, dtype=np.float64)
ORIGIN = (1580, 1030)
PIXELS = 100
SUBCELLS = 2
SUBSAMPLE = 20000
RNG = np.random.default_rng(20260817)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    }
)


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
        schmid = np.asarray(handle["/schmid/max_schmid_factor"])[
            x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1
        ]
    element_mean = 0.25 * (schmid[1:, 1:] + schmid[:-1, :-1] + schmid[1:, :-1] + schmid[:-1, 1:])
    return np.repeat(element_mean[:, :, None], SUBCELLS, axis=2).reshape(-1)


def main() -> int:
    tr = np.load(OUT / "krylov_trajectories.r16.npz", allow_pickle=False)
    n_states, n_points, _ = tr["stress"].shape
    stress = tr["stress"].reshape(-1, 3)
    d_eps = tr["d_eps_inel_observable"].reshape(-1, 3)
    eps_inel = tr["eps_inel_observable"].reshape(-1, 3)
    states = np.repeat(np.arange(n_states), n_points)
    schmid = np.tile(orientation_features(), n_states)
    position = np.tile(np.repeat(np.arange(PIXELS * SUBCELLS), n_points // (PIXELS * SUBCELLS)), n_states)

    s = deviator(stress)
    sigma_eq = np.sqrt(1.5) * np.sqrt(
        s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2 + (s[:, 0] + s[:, 1]) ** 2
    )
    dp = gauge_norm(d_eps)
    p_eq = gauge_norm(eps_inel)
    n_dir = deviator(d_eps)
    theta_s = np.arctan2(s[:, 2], (s[:, 0] - s[:, 1]) / np.sqrt(2.0))
    theta_n = np.arctan2(n_dir[:, 2], (n_dir[:, 0] - n_dir[:, 1]) / np.sqrt(2.0))

    idx = RNG.choice(n_states * n_points, size=SUBSAMPLE, replace=False)

    def hexbin(ax, x, y, title, xlabel, ylabel):
        hb = ax.hexbin(x, y, gridsize=120, bins="log", cmap="viridis", linewidths=0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return hb

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    hb = hexbin(ax, sigma_eq, dp, r"$\sigma_{\rm eq}$ vs $\Delta p$ (400k experiments, observable part)", r"$\sigma_{\rm eq}$ [MPa]", r"$\Delta p$ [gauge]")
    fig.colorbar(hb, ax=ax, label="log count")
    fig.tight_layout()
    fig.savefig(OUT / "phase_geometry_sigmaeq_dp.png")
    plt.close(fig)

    for name, values, label in (
        ("time", states, "state index (time)"),
        ("schmid", schmid, "max Schmid factor"),
        ("position", position, "x position [element x subcell]"),
    ):
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        scatter = ax.scatter(
            sigma_eq[idx], dp[idx], s=1.5, c=values[idx], cmap="viridis",
            linewidths=0, alpha=0.7,
        )
        ax.set_title(f"colored by {label}")
        ax.set_xlabel(r"$\sigma_{\rm eq}$ [MPa]")
        ax.set_ylabel(r"$\Delta p$ [gauge]")
        fig.colorbar(scatter, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(OUT / f"phase_geometry_sigmaeq_dp_{name}.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    hb = hexbin(ax, p_eq, dp, r"$p_{\rm eq}$ vs $\Delta p$", r"$p_{\rm eq}$ [gauge]", r"$\Delta p$ [gauge]")
    fig.colorbar(hb, ax=ax, label="log count")
    fig.tight_layout()
    fig.savefig(OUT / "phase_geometry_peq_dp.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    hb = hexbin(ax, sigma_eq, p_eq, r"$\sigma_{\rm eq}$ vs $p_{\rm eq}$ (visited domain)", r"$\sigma_{\rm eq}$ [MPa]", r"$p_{\rm eq}$ [gauge]")
    fig.colorbar(hb, ax=ax, label="log count")
    fig.tight_layout()
    fig.savefig(OUT / "phase_geometry_sigmaeq_peq.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    hb = ax.hexbin(
        theta_s, theta_n, gridsize=140, bins="log", cmap="viridis",
        extent=(-np.pi, np.pi, -np.pi, np.pi), linewidths=0,
    )
    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], color="white", lw=1.0, ls="--")
    ax.set_title(r"flow direction vs stress direction (J2 would lie on the dashed line)")
    ax.set_xlabel(r"$\theta_s$ [rad]")
    ax.set_ylabel(r"$\theta_n$ [rad]")
    fig.colorbar(hb, ax=ax, label="log count")
    fig.tight_layout()
    fig.savefig(OUT / "phase_geometry_flow_direction.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    p_eq_quantile = np.zeros_like(p_eq)
    edges = np.quantile(p_eq, np.linspace(0, 1, 6))
    p_eq_quantile = np.digitize(p_eq, edges[1:-1])
    scatter = ax.scatter(
        sigma_eq[idx], dp[idx], s=1.5, c=p_eq_quantile[idx], cmap="viridis",
        linewidths=0, alpha=0.7, vmin=0, vmax=5,
    )
    ax.set_title(r"branches test: $\Delta p$ vs $\sigma_{\rm eq}$, colored by $p_{\rm eq}$ quantile")
    ax.set_xlabel(r"$\sigma_{\rm eq}$ [MPa]")
    ax.set_ylabel(r"$\Delta p$ [gauge]")
    fig.colorbar(scatter, ax=ax, label=r"$p_{\rm eq}$ quantile")
    fig.tight_layout()
    fig.savefig(OUT / "phase_geometry_amplitude_branches.png")
    plt.close(fig)

    print("saved 8 figures to", OUT)
    print(
        f"sigma_eq range {sigma_eq.min():.1f}-{sigma_eq.max():.1f} MPa, "
        f"dp median {np.median(dp):.2e}, p_eq range {p_eq.min():.2e}-{p_eq.max():.2e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
