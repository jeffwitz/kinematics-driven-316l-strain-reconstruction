#!/usr/bin/env python3
"""Compare historical C and exploratory corrected-F RAW P43 EVM maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "validation/reference_data/p0043_experimental_raw_svd7_provisional_v3"
F = ROOT / "validation/reference_data/p0043_experimental_raw_svd7_f_provisional_v1"
OUT = ROOT / "validation/reference_data/raw_femu_f_v1/c_vs_f_reidentified_evm.png"


def main() -> int:
    c = np.load(C / "p0043_raw_svd7_evm_fields.npz")
    f = np.load(F / "p0043_raw_svd7_evm_fields.npz")
    state = -1
    fields = [
        c["dic"][state],
        c["final"][state],
        f["prior"][state],
        f["final"][state],
        c["final"][state] - c["dic"][state],
        f["prior"][state] - f["dic"][state],
        f["final"][state] - f["dic"][state],
        f["final"][state] - c["final"][state],
    ]
    titles = [
        "DIC", "C historique optimisé", "F prior", "F exploratoire",
        "C - DIC", "F prior - DIC", "F final - DIC", "F final - C final",
    ]
    common = max(float(np.max(field)) for field in fields[:4])
    difference = max(float(np.max(np.abs(field))) for field in fields[4:])
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for index, (axis, field, title) in enumerate(zip(axes.flat, fields, titles, strict=True)):
        is_difference = index >= 4
        image = axis.imshow(
            100.0 * field.T,
            origin="lower",
            aspect="equal",
            cmap="coolwarm" if is_difference else "viridis",
            vmin=-100.0 * difference if is_difference else 0.0,
            vmax=100.0 * difference if is_difference else 100.0 * common,
        )
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("x index")
        axis.set_ylabel("y index")
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle("P43 M20 — comparaison EVM C historique / F corrigé", fontsize=14)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=220)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
