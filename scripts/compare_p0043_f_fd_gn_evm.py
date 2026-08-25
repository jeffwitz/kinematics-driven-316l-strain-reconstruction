#!/usr/bin/env python3
"""Compare DIC, historical C, F prior and the official F FD-GN arrival EVM."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "validation/reference_data/p0043_experimental_raw_svd7_provisional_v3/p0043_raw_svd7_evm_fields.npz"
F_PRIOR = ROOT / "validation/reference_data/p0043_experimental_raw_svd7_f_provisional_v1/p0043_raw_svd7_evm_fields.npz"
F_ARRIVAL = ROOT / "validation/reference_data/p0043_f_mapping_reidentification_v1/fd_gn_one_step_maps/p0043_raw_svd7_evm_fields.npz"
OUT = ROOT / "validation/reference_data/p0043_f_mapping_reidentification_v1/fd_gn_one_step_maps/c_f_fd_gn_comparison.png"


def main() -> int:
    c = np.load(C); f0 = np.load(F_PRIOR); f1 = np.load(F_ARRIVAL)
    state = -1
    fields = [c["dic"][state], c["final"][state], f0["prior"][state], f1["final"][state],
              c["final"][state] - c["dic"][state], f0["prior"][state] - f0["dic"][state],
              f1["final"][state] - f1["dic"][state], f1["final"][state] - f0["prior"][state]]
    titles = ["DIC", "C historique", "F départ", "F arrivée (FD-GN)",
              "C−DIC", "F départ−DIC", "F arrivée−DIC", "F arrivée−départ"]
    common = max(float(np.nanmax(x)) for x in fields[:4])
    diff = max(float(np.nanmax(np.abs(x))) for x in fields[4:])
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for i, (ax, field, title) in enumerate(zip(axes.flat, fields, titles, strict=True)):
        is_diff = i >= 4
        im = ax.imshow(100.0 * field.T, origin="lower", aspect="equal",
                       cmap="coolwarm" if is_diff else "viridis",
                       vmin=-100.0 * diff if is_diff else 0.0,
                       vmax=100.0 * diff if is_diff else 100.0 * common)
        ax.set_title(title, fontsize=9); ax.set_xlabel("x index"); ax.set_ylabel("y index")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("P43 M20 — EVM : DIC, C historique et F corrigé FD-GN", fontsize=14)
    OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT, dpi=220); plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
