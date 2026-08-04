#!/usr/bin/env python3
"""Build small, source-linked figures for the spectral mechanics documentation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "validation" / "_generated" / "ebi_tet"
TARGET = ROOT / "docs" / "_static" / "spectral_mechanics"


def load(mesh: int) -> dict:
    return json.loads((SOURCE / f"state_sharing_m{mesh}.json").read_text())


def save(fig: plt.Figure, name: str) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(TARGET / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    m8, m12, m24 = (load(mesh) for mesh in (8, 12, 24))

    errors = m24["errors"]
    labels = ["TET2 / CPS4", "EBI / TET2", "EBI / CPS4"]
    values = [errors["tet_cps4"]["E_Gamma"], errors["ebi_tet"]["E_Gamma"], errors["ebi_cps4"]["E_Gamma"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(labels, np.asarray(values) * 100.0, color=["#2b6cb0", "#dd6b20", "#805ad5"])
    ax.set_ylabel("Accumulated-slip error (%)")
    ax.set_title("Causal error decomposition at 24x24")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "error_decomposition.png")

    meshes = np.array([8, 12, 24])
    tet = np.array([load(mesh)["errors"]["tet_cps4"]["E_Gamma"] for mesh in meshes]) * 100.0
    ebi = np.array([load(mesh)["errors"]["ebi_tet"]["E_Gamma"] for mesh in meshes]) * 100.0
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(meshes, tet, "o-", label="TET2 / CPS4")
    ax.plot(meshes, ebi, "s-", label="EBI / TET2")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="1%")
    ax.set_xlabel("Pixels per direction")
    ax.set_ylabel("Accumulated-slip error (%)")
    ax.set_title("Refinement separates spatial and state-sharing errors")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "refinement_accumulated_slip.png")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(["EBI-TET", "TET2"], [m24["timings"]["ebi_seconds"], m24["timings"]["tet_two_state_seconds"]], color=["#805ad5", "#2b6cb0"])
    ax.set_ylabel("Single-run time (s)")
    ax.set_title("Indicative 24x24 runtime")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "runtime_comparison.png")


if __name__ == "__main__":
    main()
