"""Spatial diagnostic for the CPS4R qualification.

Section 13 of the reduced-integration specification: the hourglass energy has to
be read next to the constitutive activity, because a small domain average can
hide a concentration sitting exactly inside a plastic band.

Consumes the arrays written by `scripts/qualify_reduced_integration.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _panel(axes, field: np.ndarray, title: str, *, cmap: str) -> None:
    # Transposed with a lower origin so the plot axes are the physical (x, y)
    # of the project convention rather than array rows and columns.
    image = axes.imshow(field.T, origin="lower", cmap=cmap, aspect="equal")
    axes.set_title(title, fontsize=9)
    axes.set_xlabel("x (element)", fontsize=8)
    axes.set_ylabel("y (element)", fontsize=8)
    axes.tick_params(labelsize=7)
    bar = plt.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    bar.ax.tick_params(labelsize=7)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("validation/_generated/cps4r_qualification")
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (args.input / "cps4r_spatial_diagnostic.png")

    peeq = np.load(args.input / "c1_reference_peeq.npy")
    peeq_reduced = np.load(args.input / "c1_peeq_beta_1.npy")
    hourglass = np.load(args.input / "c1_hourglass_energy_beta_1.0.npy")
    hourglass_soft = np.load(args.input / "c1_hourglass_energy_beta_0.1.npy")

    figure, grid = plt.subplots(2, 3, figsize=(12.5, 7.4))

    _panel(grid[0, 0], peeq, "CPS4 equivalent plastic strain", cmap="viridis")
    _panel(
        grid[0, 1],
        np.abs(peeq_reduced - peeq),
        r"$|$PEEQ$_{CPS4R,\beta=1}$ - PEEQ$_{CPS4}|$",
        cmap="magma",
    )
    _panel(grid[0, 2], hourglass, r"hourglass energy, $\beta=1$ (N mm)", cmap="inferno")
    _panel(
        grid[1, 0], hourglass_soft, r"hourglass energy, $\beta=0.1$ (N mm)", cmap="inferno"
    )

    # The claim under test: does the stabilisation energy sit where the plastic
    # activity sits? If it does, a global average cannot be trusted as a gate.
    axes = grid[1, 1]
    axes.scatter(peeq.ravel(), hourglass.ravel(), s=6, alpha=0.5, color="#B4413C")
    correlation = float(np.corrcoef(peeq.ravel(), hourglass.ravel())[0, 1])
    axes.set_title(
        f"co-location, Pearson r = {correlation:.3f}",
        fontsize=9,
    )
    axes.set_xlabel("PEEQ (CPS4)", fontsize=8)
    axes.set_ylabel(r"hourglass energy, $\beta=1$", fontsize=8)
    axes.tick_params(labelsize=7)
    axes.grid(alpha=0.25, linewidth=0.5)

    # The error the diagnostic is supposed to predict, against the diagnostic.
    axes = grid[1, 2]
    error = np.abs(peeq_reduced - peeq)
    axes.scatter(hourglass.ravel(), error.ravel(), s=6, alpha=0.5, color="#1F6F78")
    error_correlation = float(np.corrcoef(hourglass.ravel(), error.ravel())[0, 1])
    axes.set_title(
        f"does energy predict error? r = {error_correlation:.3f}",
        fontsize=9,
    )
    axes.set_xlabel(r"hourglass energy, $\beta=1$", fontsize=8)
    axes.set_ylabel("PEEQ absolute error", fontsize=8)
    axes.tick_params(labelsize=7)
    axes.grid(alpha=0.25, linewidth=0.5)

    figure.suptitle(
        "CPS4R spatial diagnostic, heterogeneous J2, 32x32 "
        f"(PEEQ co-location r = {correlation:.3f}, "
        f"error prediction r = {error_correlation:.3f})",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    print(f"wrote {output}")
    print(f"peeq/hourglass correlation      {correlation:.4f}")
    print(f"hourglass/error correlation     {error_correlation:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
