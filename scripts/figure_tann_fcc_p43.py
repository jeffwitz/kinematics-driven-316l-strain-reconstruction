#!/usr/bin/env python3
"""Figures A-G of the causal TANN-FCC P43 run, from the artifact only.

Generated from `tann_fcc_p43_run.json` and its companion `.npz` (the
per-state fields), never from copied values. G is read from the material
qualification artifact. Palette: the validated default (blue = train,
orange = holdout, sequential blue for field maps), thin marks, direct
labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e5e3de"
HOLDOUT = {24, 28, 32, 36, 39}


def _axes_clean(axis) -> None:
    axis.grid(alpha=0.35, color=GRID, linewidth=0.8)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.tick_params(colors=MUTED, length=3)


def figure_a(artifact: dict, output: Path) -> None:
    """E_n vs state, elastic = 1, train/holdout distinction."""

    step = artifact["steps"][-1]
    states = sorted({int(s) for group in (step["e_train"], step["e_holdout"]) for s in group})
    fig, axis = plt.subplots(figsize=(7.2, 4.0))
    axis.axhline(1.0, color=INK, linewidth=1.0, linestyle="--", alpha=0.7)
    axis.text(states[0], 1.01, "elastic = 1", color=INK, fontsize=9)
    train = [(s, step["e_train"][str(s)]) for s in states if str(s) in step["e_train"]]
    holdout = [(s, step["e_holdout"][str(s)]) for s in states if str(s) in step["e_holdout"]]
    for label, points, color in (("train", train, BLUE), ("holdout", holdout, ORANGE)):
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        axis.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=label)
        for x, y in points:
            if label == "holdout":
                axis.text(x, y, f"{y:.3f}", color=color, fontsize=8, ha="center", va="bottom")
    axis.set_xlabel("state")
    axis.set_ylabel("E_n")
    axis.set_title(f"Figure A -- E_n vs state, step {step['step']} (margin 0.0202)")
    axis.legend(frameon=False, loc="upper left")
    _axes_clean(axis)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _state_triplet(artifact: dict) -> list[int]:
    return sorted(
        {
            min(artifact["states"]),
            artifact["states"][len(artifact["states"]) // 2],
            max(artifact["states"]),
        }
    )


def figure_b(artifact: dict, fields: dict, output: Path) -> None:
    """u_y (axial): DIC vs elastic vs TANN at low/mid/high states."""

    states = _state_triplet(artifact)
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 8.2))
    for column, state in enumerate(states):
        prefix = f"state_{state}"
        u_meas = fields[f"{prefix}_u_meas"]
        vmax = np.abs(u_meas).max()
        for row, (name, field) in enumerate(
            (
                ("DIC", u_meas),
                ("elastic", fields[f"{prefix}_u_elastic"]),
                ("TANN", fields[f"{prefix}_u_sim"]),
            )
        ):
            axis = axes[row, column]
            axis.imshow(field[..., 1].T, origin="lower", cmap="viridis", vmin=-vmax, vmax=vmax)
            axis.set_title(f"{name}, state {state}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_ylabel("u_y", fontsize=9)
    fig.suptitle("Figure B -- axial displacement", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def figure_c(artifact: dict, fields: dict, output: Path) -> None:
    """Spatial DIC residual (u - u_DIC): elastic vs TANN, same states."""

    states = _state_triplet(artifact)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.8))
    for column, state in enumerate(states):
        prefix = f"state_{state}"
        u_meas = fields[f"{prefix}_u_meas"]
        residual_el = fields[f"{prefix}_u_elastic"] - u_meas
        residual_tann = fields[f"{prefix}_u_sim"] - u_meas
        vmax = max(np.abs(residual_el).max(), np.abs(residual_tann).max())
        for row, (name, resid) in enumerate((("elastic", residual_el), ("TANN", residual_tann))):
            axis = axes[row, column]
            axis.imshow(resid[..., 1].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axis.set_title(f"{name}, state {state}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("Figure C -- DIC residual (u_y component)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def figure_d(artifact: dict, fields: dict, output: Path) -> None:
    """Total slip activity per state: sum_alpha |Delta gamma| (diagnostic)."""

    states = sorted(artifact["states"])
    activity = []
    previous = None
    for state in states:
        gamma = fields[f"state_{state}_committed_state"][..., 0]  # (P, 12)
        if previous is not None:
            activity.append(float(np.sum(np.abs(gamma - previous))))
        previous = gamma
    fig, axis = plt.subplots(figsize=(7.2, 4.0))
    axis.plot(states[1:], activity, color=BLUE, linewidth=2, marker="o", markersize=4)
    for state, value in zip(states[1:], activity, strict=True):
        if state in HOLDOUT:
            axis.plot([state], [value], color=ORANGE, marker="o", markersize=6)
    axis.set_xlabel("state")
    axis.set_ylabel("sum |Delta gamma|")
    axis.set_title("Figure D -- slip activity (diagnostic only)")
    _axes_clean(axis)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def figure_e(artifact: dict, fields: dict, output: Path) -> None:
    """Generalised dissipation per state: mean and total."""

    states = sorted(artifact["states"])
    means = [float(fields[f"state_{state}_dissipation"].mean()) for state in states]
    totals = [float(fields[f"state_{state}_dissipation"].sum()) for state in states]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for axis, values, name in ((ax1, means, "mean D"), (ax2, totals, "total D")):
        axis.plot(states, values, color=BLUE, linewidth=2, marker="o", markersize=4)
        for state, value in zip(states, values, strict=True):
            if state in HOLDOUT:
                axis.plot([state], [value], color=ORANGE, marker="o", markersize=6)
        axis.set_xlabel("state")
        axis.set_ylabel(name)
        _axes_clean(axis)
    fig.suptitle("Figure E -- generalised dissipation", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def figure_f(artifact: dict, fields: dict, output: Path) -> None:
    """Latent components per state (no physical interpretation attached)."""

    states = sorted(artifact["states"])
    means = [
        fields[f"state_{state}_committed_state"][..., 1:].reshape(-1, 2).mean(axis=0)
        for state in states
    ]
    fig, axis = plt.subplots(figsize=(7.2, 4.0))
    for component in range(2):
        axis.plot(
            states,
            [m[component] for m in means],
            linewidth=2,
            color=(BLUE, ORANGE)[component],
            label=f"z_{component} mean",
        )
    axis.set_xlabel("state")
    axis.set_ylabel("mean latent component")
    axis.set_title("Figure F -- latent evolution (diagnostic only)")
    axis.legend(frameon=False)
    _axes_clean(axis)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def figure_g(qualification_path: Path, output: Path) -> None:
    """Substep invariance from the material qualification artifact."""

    report = json.loads(qualification_path.read_text(encoding="utf-8"))
    values = report["substepping"]
    labels = sorted(values, key=int)
    errors = [values[label] for label in labels]
    fig, axis = plt.subplots(figsize=(7.2, 4.0))
    axis.semilogy(
        [int(label) for label in labels], errors, color=BLUE, linewidth=2, marker="o", markersize=5
    )
    axis.set_xlabel("substeps")
    axis.set_ylabel("max |state difference| from 8 substeps")
    axis.set_title("Figure G -- substep invariance")
    _axes_clean(axis)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "validation/_generated/shared_tensor_generator/tann_fcc_p43_run.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "validation/figures/tann_fcc_p43")
    arguments = parser.parse_args()
    artifact = json.loads(arguments.artifact.read_text(encoding="utf-8"))
    fields_path = Path(artifact["fields_path"])
    fields = dict(np.load(fields_path, allow_pickle=False))
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    figure_a(artifact, arguments.output_dir / "A_e_metric.png")
    figure_b(artifact, fields, arguments.output_dir / "B_displacements.png")
    figure_c(artifact, fields, arguments.output_dir / "C_residuals.png")
    figure_d(artifact, fields, arguments.output_dir / "D_slip_activity.png")
    figure_e(artifact, fields, arguments.output_dir / "E_dissipation.png")
    figure_f(artifact, fields, arguments.output_dir / "F_latent.png")
    qualification = (
        ROOT / "validation/_generated/shared_tensor_generator/tann_fcc_material_qualification.json"
    )
    if qualification.exists():
        figure_g(qualification, arguments.output_dir / "G_substepping.png")
    print(f"wrote figures to {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
