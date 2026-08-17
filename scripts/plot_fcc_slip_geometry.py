#!/usr/bin/env python3
"""Slip-coordinate geometry figures, the counterpart of the tensor-phase ones.

Per system and pooled: the driving force `tau^alpha` against the activity
`Delta gamma^alpha` (the slip-law test — J2-like laws would draw a curve
through the origin per system), and the accumulated history `Gamma^alpha`
against the activity (the hardening fan, the counterpart of `p_eq -> Delta p`).
The activity is the sign-constrained L2 decomposition; all figures are
hexbins on the same sequential ramp, saved beside the other phase-space
artifacts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/_generated/shared_tensor_generator"
SUBSAMPLE = 20000
RNG = np.random.default_rng(20260817)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    }
)


def main() -> int:
    fields = np.load(OUT / "fcc_slip_fields.npz", allow_pickle=False)
    tau = fields["tau"]  # (samples, 12)
    gamma = fields["gamma"]  # (samples, 12)
    cumulative = fields["gamma_cumulative"]  # (samples, 12)
    states = fields["states"]
    n_systems = tau.shape[1]

    subsample = RNG.choice(tau.shape[0], size=SUBSAMPLE, replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=False, sharey=False)
    for system in range(n_systems):
        ax = axes.flat[system]
        sample = RNG.choice(tau.shape[0], size=100000, replace=False)
        ax.hexbin(tau[sample, system], gamma[sample, system], gridsize=80,
                  bins="log", cmap="viridis", linewidths=0)
        ax.axhline(0.0, color="white", lw=0.8, ls="--")
        ax.axvline(0.0, color="white", lw=0.8, ls="--")
        ax.set_title(f"system {system + 1}")
        if system >= 8:
            ax.set_xlabel(r"$\tau^\alpha$ [MPa]")
        if system % 4 == 0:
            ax.set_ylabel(r"$\Delta\gamma^\alpha$")
    fig.suptitle(r"driving force vs activity, per system (sign cone: activity only where $\tau^\alpha \Delta\gamma^\alpha \geq 0$)")
    fig.tight_layout()
    fig.savefig(OUT / "fcc_geometry_tau_gamma_panels.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for system in range(n_systems):
        ax = axes.flat[system]
        sample = RNG.choice(tau.shape[0], size=100000, replace=False)
        ax.hexbin(cumulative[sample, system], gamma[sample, system], gridsize=80,
                  bins="log", cmap="viridis", linewidths=0)
        ax.axhline(0.0, color="white", lw=0.8, ls="--")
        ax.set_title(f"system {system + 1}")
        if system >= 8:
            ax.set_xlabel(r"$\Gamma^\alpha$")
        if system % 4 == 0:
            ax.set_ylabel(r"$\Delta\gamma^\alpha$")
    fig.suptitle(r"accumulated slip history vs activity, per system (the hardening fan)")
    fig.tight_layout()
    fig.savefig(OUT / "fcc_geometry_gamma_cumulative_panels.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    hb = axes[0].hexbin(tau.ravel(), gamma.ravel(), gridsize=120, bins="log",
                        cmap="viridis", linewidths=0)
    axes[0].axhline(0.0, color="white", lw=0.8, ls="--")
    axes[0].axvline(0.0, color="white", lw=0.8, ls="--")
    axes[0].set_title(r"pooled: $\tau^\alpha$ vs $\Delta\gamma^\alpha$ (4.8M system-samples)")
    axes[0].set_xlabel(r"$\tau^\alpha$ [MPa]")
    axes[0].set_ylabel(r"$\Delta\gamma^\alpha$")
    fig.colorbar(hb, ax=axes[0], label="log count")
    hb = axes[1].hexbin(cumulative.ravel(), gamma.ravel(), gridsize=120, bins="log",
                        cmap="viridis", linewidths=0)
    axes[1].axhline(0.0, color="white", lw=0.8, ls="--")
    axes[1].set_title(r"pooled: $\Gamma^\alpha$ vs $\Delta\gamma^\alpha$")
    axes[1].set_xlabel(r"$\Gamma^\alpha$")
    axes[1].set_ylabel(r"$\Delta\gamma^\alpha$")
    fig.colorbar(hb, ax=axes[1], label="log count")
    fig.tight_layout()
    fig.savefig(OUT / "fcc_geometry_pooled_hexbin.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sample_systems = RNG.choice(n_systems, size=SUBSAMPLE, replace=True)
    scatter = ax.scatter(
        tau[subsample, sample_systems], gamma[subsample, sample_systems],
        s=1.5, c=states[subsample], cmap="viridis", linewidths=0, alpha=0.7,
    )
    ax.axhline(0.0, color="white", lw=0.8, ls="--")
    ax.axvline(0.0, color="white", lw=0.8, ls="--")
    ax.set_title(r"pooled $\tau^\alpha$ vs $\Delta\gamma^\alpha$, colored by state (time)")
    ax.set_xlabel(r"$\tau^\alpha$ [MPa]")
    ax.set_ylabel(r"$\Delta\gamma^\alpha$")
    fig.colorbar(scatter, ax=ax, label="state index (time)")
    fig.tight_layout()
    fig.savefig(OUT / "fcc_geometry_pooled_time.png")
    plt.close(fig)

    print(f"saved 4 slip-geometry figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
