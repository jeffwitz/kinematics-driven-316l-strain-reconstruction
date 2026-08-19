#!/usr/bin/env python3
"""Experimental vs computed equivalent-strain fields of the P43 run.

Experimental: the archived EVM history (von Mises of the first-difference
strain, incompressible closure -- the qualified DIC quantity), cropped to
the 100x100 campaign window and aligned by a shift scan against the
simulation's own kinematics. Computed: the same definition applied to
the elastic and TANN displacement fields of the run artifact. Rows are
states, columns exp./elastic/TANN, one shared scale per row.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_disflow_profiles_p43 import equivalent_strain  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/p0043_evm_history.h5")
DEFAULT_ARTIFACT = ROOT / "validation/_generated/shared_tensor_generator/tann_fcc_p43_run.npz"
ORIGIN = (1580, 1030)
PIXELS = 100
STATES = (25, 32, 40)


def computed_evm(displacement: np.ndarray) -> np.ndarray:
    """The archived definition on a nodal `(nx+1, ny+1, 2)` field in mm."""

    along_rows = displacement[..., 1]  # u_y
    along_columns = displacement[..., 0]  # u_x
    return equivalent_strain(along_rows, along_columns)


def best_shift(measured: np.ndarray, archived: np.ndarray) -> tuple[int, int]:
    """Cross-correlation shift scan, the repository's registration idiom."""

    best = (-np.inf, (0, 0))
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            shifted = archived[
                max(dx, 0) : archived.shape[0] - max(-dx, 0),
                max(dy, 0) : archived.shape[1] - max(-dy, 0),
            ]
            core = measured[
                max(-dx, 0) : measured.shape[0] - max(dx, 0),
                max(-dy, 0) : measured.shape[1] - max(dy, 0),
            ]
            score = float(np.corrcoef(core.ravel(), shifted.ravel())[0, 1])
            if score > best[0]:
                best = (score, (dx, dy))
    return best[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/figures/tann_fcc_p43/EVM_exp_vs_calc.png")
    arguments = parser.parse_args()

    fields = dict(np.load(arguments.artifact, allow_pickle=False))
    with h5py.File(ARCHIVE, "r") as handle:
        evm = np.asarray(handle["evm"])  # (41, H, W)
    report = __import__("json").loads(
        (ROOT / "validation/reference_data/dic_multistep_history_p0043_repaired_v1/report.json").read_text()
    )
    bounds = list(map(int, report["solve_bounds"]))
    x0, y0 = ORIGIN

    # alignment: the archived EVM grid vs the simulation pixels
    measured_probe = computed_evm(fields["state_40_u_meas"])
    archived_probe = evm[
        40, x0 - bounds[0] : x0 - bounds[0] + PIXELS, y0 - bounds[2] : y0 - bounds[2] + PIXELS
    ]
    shift = best_shift(measured_probe, archived_probe)
    print(f"alignment shift (archived minus simulation): {shift}", flush=True)

    figure, axes = plt.subplots(len(STATES), 3, figsize=(11.5, 9.8))
    for row, state in enumerate(STATES):
        archived = evm[
            state,
            x0 - bounds[0] + shift[0] : x0 - bounds[0] + shift[0] + PIXELS,
            y0 - bounds[2] + shift[1] : y0 - bounds[2] + shift[1] + PIXELS,
        ]
        elastic = computed_evm(fields[f"state_{state}_u_elastic"])
        tann = computed_evm(fields[f"state_{state}_u_sim"])
        vmax = float(np.percentile(archived, 99.5))
        for column, (name, field) in enumerate(
            (("DIC EVM", archived), ("elastic", elastic), ("TANN", tann))
        ):
            axis = axes[row, column]
            image = axis.imshow(field.T, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax)
            axis.set_title(f"{name}, state {state}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle("Equivalent strain: experimental DIC vs computed (primary run, sigma_ref = 2 mu)", fontsize=11)
    figure.tight_layout()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
