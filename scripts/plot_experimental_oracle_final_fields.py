#!/usr/bin/env python3
"""Plot final mechanical fields from the completed P43 M20 oracle runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _vm_strain(u: np.ndarray) -> np.ndarray:
    """2-D von-Mises equivalent strain from nodal displacement."""
    ux, uy = u[..., 0], u[..., 1]
    dx = 1.0 / max(ux.shape[0] - 1, 1)
    dy = 1.0 / max(ux.shape[1] - 1, 1)
    exx = np.gradient(ux, dx, axis=0)
    eyy = np.gradient(uy, dy, axis=1)
    exy = 0.5 * (np.gradient(ux, dy, axis=1) + np.gradient(uy, dx, axis=0))
    return np.sqrt(exx * exx - exx * eyy + eyy * eyy + 3.0 * exy * exy)


def _stress_vm(stress: np.ndarray) -> np.ndarray:
    sxx, syy, sxy = stress[..., 0], stress[..., 1], stress[..., 2]
    return np.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)


def _panel(ax, field, title, cmap="viridis", vmin=None, vmax=None, unit=""):
    image = ax.imshow(field.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if unit:
        ax.set_xlabel(unit, fontsize=8)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        type=Path,
        default=Path(
            "validation/_generated/performance/experimental_oracle_p43_m20/"
            "oracle_transfer_augmented/fields.npz"
        ),
    )
    parser.add_argument(
        "--reduced",
        type=Path,
        default=Path(
            "validation/_generated/performance/experimental_oracle_p43_m20/"
            "reduced_transfer_rank2_prior003/fields.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "validation/_generated/performance/experimental_oracle_p43_m20/"
            "final_mechanical_fields_comparison.png"
        ),
    )
    args = parser.parse_args()

    full = np.load(args.full)
    reduced = np.load(args.reduced)
    u_full = full["oracle_displacement_history"][-1]
    u_reduced = reduced["displacement_history"][-1]
    dp_full = full["oracle_increment_history"][-1].mean(axis=-1)
    dp_reduced = reduced["increment_history"][-1].mean(axis=-1)
    peeq = full["oracle_equivalent_plastic_strain_history"][-1].mean(axis=-1)
    stress = full["oracle_stress_history_mpa"][-1]
    stress_vm = _stress_vm(stress).mean(axis=-1)

    # Nodal fields are compared on the common interior material grid.
    uf = u_full[1:-1, 1:-1]
    ur = u_reduced[1:-1, 1:-1]
    fields = [
        (uf[..., 0], "Full: uₓ (mm)", "coolwarm"),
        (ur[..., 0], "Réduit: uₓ (mm)", "coolwarm"),
        (ur[..., 0] - uf[..., 0], "Réduit - full: uₓ", "coolwarm"),
        (uf[..., 1], "Full: uᵧ (mm)", "coolwarm"),
        (ur[..., 1], "Réduit: uᵧ (mm)", "coolwarm"),
        (ur[..., 1] - uf[..., 1], "Réduit - full: uᵧ", "coolwarm"),
        (np.linalg.norm(uf, axis=-1), "Full: |u| (mm)", "viridis"),
        (np.linalg.norm(ur, axis=-1), "Réduit: |u| (mm)", "viridis"),
        (np.linalg.norm(ur - uf, axis=-1), "Réduit - full: |Δu|", "magma"),
        (_vm_strain(uf), "Full: εVM (grad. déplacement)", "magma"),
        (_vm_strain(ur), "Réduit: εVM (grad. déplacement)", "magma"),
        (_vm_strain(ur) - _vm_strain(uf), "Réduit - full: εVM", "coolwarm"),
        (dp_full, "Full: Δp final", "magma"),
        (dp_reduced, "Réduit: Δp final", "magma"),
        (dp_reduced - dp_full, "Réduit - full: Δp", "coolwarm"),
        (peeq, "Full: PEEQ", "magma"),
        (stress_vm, "Full: stress VM (MPa)", "magma"),
        (np.zeros_like(peeq), "Réduit: PEEQ/stress VM non archivés", "Greys"),
    ]
    fig, axes = plt.subplots(6, 3, figsize=(13, 22), constrained_layout=True)
    for row in range(6):
        for col in range(3):
            field, title, cmap = fields[3 * row + col]
            if "diff" in title.lower() or "Δ" in title:
                scale = max(float(np.max(np.abs(field))), 1e-15)
                _panel(axes[row, col], field, title, cmap, -scale, scale)
            else:
                _panel(axes[row, col], field, title, cmap)
    fig.suptitle("P43 M20 — champs mécaniques finaux, full vs oracle réduit", fontsize=16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
