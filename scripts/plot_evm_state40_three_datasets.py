#!/usr/bin/env python3
"""State-40 equivalent strain, the received field against both DISFlow profiles.

One shared colour scale, because the whole point is that these three fields are
not interchangeable and a per-panel scale would hide it. The limits are the
5th and 95th percentiles pooled over the three, so no single dataset sets the
scale for the others.

Two rows, and the second is the one that matters. The full field is 3600x3100
and cannot be shown honestly on a screen: block-averaging by four is a display
choice that suppresses exactly the pixel-scale content under discussion. So the
top row is the block-averaged overview, for the large-scale layout, and the
bottom row a native-resolution window where the fine structure is real pixels.
The radial spectrum says the received field carries thirty to fifty times more
power below a twelve-pixel wavelength; the bottom row is where that is visible
rather than tabulated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from compare_disflow_profiles_p43 import equivalent_strain  # type: ignore[import-not-found]
from numpy.typing import NDArray
from tune_disflow_alpha_against_received import _flow_field  # type: ignore[import-not-found]

FloatArray = NDArray[np.float64]

ROOT = Path(__file__).resolve().parents[1]
RECEIVED_ROOT = ROOT / "data/raw/case_study"
DATA_ROOT = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical")


def _block_mean(field: FloatArray, factor: int) -> FloatArray:
    rows = field.shape[0] // factor * factor
    columns = field.shape[1] // factor * factor
    return (
        field[:rows, :columns]
        .reshape(rows // factor, factor, columns // factor, factor)
        .mean(axis=(1, 3))
    )


def _from_history(path: Path, state: int) -> FloatArray:
    with h5py.File(path, "r") as handle:
        rows = np.asarray(handle["displacement_pixel/along_rows"][state], dtype=np.float64)
        columns = np.asarray(
            handle["displacement_pixel/along_columns"][state], dtype=np.float64
        )
    return equivalent_strain(rows, columns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=int, default=40)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--zoom-size", type=int, default=600)
    # The converged settings are recomputed here rather than read from a stored
    # history: one image pair costs a minute, and the whole point of the panel
    # is to compare against fields that were produced with too few refinement
    # iterations to have erased the matching grid.
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    images = sorted(
        (Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/DIC_images")).glob("*.tif")
    )
    converged_rows, converged_columns = _flow_field(
        cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE),
        cv2.imread(str(images[arguments.state]), cv2.IMREAD_GRAYSCALE),
        alpha=arguments.alpha,
        iterations=arguments.iterations,
        patch_size=4,
        patch_stride=1,
        epsilon=arguments.epsilon,
    )

    fields = {
        "received (Adil)": equivalent_strain(
            np.asarray(np.load(RECEIVED_ROOT / "V_40.npy"), dtype=np.float64),
            np.asarray(np.load(RECEIVED_ROOT / "U_40.npy"), dtype=np.float64),
        ),
        "unconverged\nalpha 100, 30 iterations": _from_history(
            DATA_ROOT / "p0043_disflow_history_patch4.h5", arguments.state
        ),
        f"converged\nalpha {arguments.alpha:g}, eps {arguments.epsilon:g},"
        f" {arguments.iterations} iterations": equivalent_strain(
            converged_rows, converged_columns
        ),
    }

    pooled = np.concatenate([field[::7, ::7].ravel() for field in fields.values()])
    low, high = np.percentile(pooled, (5.0, 95.0))
    print(f"shared colour limits, pooled 5th-95th percentile: {low:.4e} to {high:.4e}")

    size = arguments.zoom_size
    shape = next(iter(fields.values())).shape
    top = shape[0] // 2 - size // 2
    left = shape[1] // 2 - size // 2

    figure, axes = plt.subplots(2, 3, figsize=(13.0, 8.6), constrained_layout=True)
    for column, (label, field) in enumerate(fields.items()):
        overview = _block_mean(field, arguments.factor)
        image = axes[0, column].imshow(
            overview, vmin=low, vmax=high, cmap="inferno", origin="lower"
        )
        axes[0, column].set_title(
            f"{label}\nRMS {np.sqrt((field**2).mean()):.3e}", fontsize=10
        )
        axes[1, column].imshow(
            field[top : top + size, left : left + size],
            vmin=low,
            vmax=high,
            cmap="inferno",
            origin="lower",
        )
        axes[1, column].set_title(f"{size}x{size} px window, native resolution", fontsize=9)
        for row in (0, 1):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    axes[0, 0].set_ylabel(f"full field, block-averaged x{arguments.factor}", fontsize=9)
    figure.colorbar(
        image, ax=axes, shrink=0.6, label="equivalent strain (incompressible closure)"
    )
    figure.suptitle(
        f"State {arguments.state} equivalent strain, one shared scale "
        f"({low:.2e} to {high:.2e}, pooled 5th-95th percentile)",
        fontsize=11,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=170)
    plt.close(figure)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
