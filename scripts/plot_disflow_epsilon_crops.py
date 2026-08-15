#!/usr/bin/env python3
"""Does the L1-to-L2 knob change the texture, at fixed smoothing weight?

`VariationalRefinementEpsilon` is the constant in the robust penalty
`psi(s^2) = sqrt(s^2 + eps^2)`. Small values approach an L1 penalty, which
tolerates discontinuities and keeps sharp bands; large values approach L2,
which spreads gradient over the neighbourhood and rounds them off. The
repository default is 0.002, already deep in the L1 regime, so the question is
what is being given up by not moving towards L2 -- or gained.

Alpha is held at 15 throughout, so any difference is the penalty shape alone
and not the amount of smoothing. A hundred-pixel window is used because that is
where the distinction lives: at full field the four are indistinguishable, and
the earlier comparison already showed that a block-averaged view hides exactly
the content in question.

The window is chosen rather than picked by eye: the received field is scanned
for the hundred-pixel square with the largest equivalent-strain standard
deviation, so the crop sits on real structure and the same coordinates are used
for every panel including the received one, which is shown alongside as the
target rather than as another variant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from compare_disflow_profiles_p43 import (  # type: ignore[import-not-found]
    _radial_spectrum,
    equivalent_strain,
)
from numpy.typing import NDArray
from tune_disflow_alpha_against_received import _flow_field  # type: ignore[import-not-found]

FloatArray = NDArray[np.float64]

ROOT = Path(__file__).resolve().parents[1]
RECEIVED_ROOT = ROOT / "data/raw/case_study"
IMAGE_ROOT = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/DIC_images")


def _busiest_window(field: FloatArray, size: int, stride: int = 50) -> tuple[int, int]:
    """Top-left corner of the `size` square with the most structure.

    Standard deviation rather than mean, because a uniformly highly strained
    region says nothing about the penalty shape; what discriminates L1 from L2
    is a sharp band next to a quiet neighbourhood.
    """

    best = (0, 0)
    score = -1.0
    margin = 300
    for row in range(margin, field.shape[0] - size - margin, stride):
        for column in range(margin, field.shape[1] - size - margin, stride):
            window = field[row : row + size, column : column + size]
            deviation = float(window.std())
            if deviation > score:
                score, best = deviation, (row, column)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.001, 0.01, 0.05, 0.1])
    parser.add_argument("--iterations", type=int, default=30)
    # The grid the matching stage leaves behind is what the refinement exists to
    # erase, so the iteration budget is a sweep axis in its own right. When
    # given, it replaces the epsilon sweep and the first epsilon is held fixed.
    parser.add_argument("--iteration-values", nargs="+", type=int, default=None)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    received = equivalent_strain(
        np.asarray(np.load(RECEIVED_ROOT / "V_40.npy"), dtype=np.float64),
        np.asarray(np.load(RECEIVED_ROOT / "U_40.npy"), dtype=np.float64),
    )
    _, received_power = _radial_spectrum(received)
    frequency, _ = _radial_spectrum(received)
    band = (frequency > 0.01) & (frequency < 0.45)
    row, column = _busiest_window(received, arguments.size)
    print(f"window at row {row}, column {column} of the 3600x3100 support", flush=True)

    images = sorted(IMAGE_ROOT.glob("*.tif"))
    reference = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(str(images[40]), cv2.IMREAD_GRAYSCALE)

    panels = [("received (Adil)", received, None)]
    summary = []
    sweeping_iterations = arguments.iteration_values is not None
    cases = (
        [(arguments.epsilons[0], count) for count in arguments.iteration_values]
        if sweeping_iterations
        else [(epsilon, arguments.iterations) for epsilon in arguments.epsilons]
    )
    for epsilon, iterations in cases:
        rows, columns = _flow_field(
            reference,
            target,
            alpha=arguments.alpha,
            iterations=iterations,
            patch_size=4,
            patch_stride=1,
            epsilon=epsilon,
        )
        field = equivalent_strain(rows, columns)
        _, power = _radial_spectrum(field)
        score = float(
            np.mean(np.abs(np.log(np.maximum(power[band], 1e-300) / received_power[band])))
        )
        correlation = float(np.corrcoef(field.ravel(), received.ravel())[0, 1])
        entry = {
            "epsilon": epsilon,
            "iterations": iterations,
            "spectral_score": score,
            "evm_correlation": correlation,
            "evm_rms_ratio": float(
                np.sqrt((field**2).mean()) / np.sqrt((received**2).mean())
            ),
        }
        summary.append(entry)
        label = (
            f"{iterations} VR iterations" if sweeping_iterations else f"epsilon = {epsilon:g}"
        )
        panels.append((label, field, entry))
        print(
            f"  epsilon {epsilon:6.3f} x{iterations:4d}: score {score:.3f}  "
            f"corr {correlation:.4f}  "
            f"rms ratio {entry['evm_rms_ratio']:.3f}",
            flush=True,
        )

    windows = [
        field[row : row + arguments.size, column : column + arguments.size]
        for _, field, _ in panels
    ]
    pooled = np.concatenate([window.ravel() for window in windows])
    low, high = np.percentile(pooled, (5.0, 95.0))

    figure, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.9),
                                constrained_layout=True)
    for axis, (label, _, entry), window in zip(axes, panels, windows, strict=True):
        image = axis.imshow(window, vmin=low, vmax=high, cmap="inferno", origin="lower")
        subtitle = "" if entry is None else f"\nscore {entry['spectral_score']:.2f}"
        axis.set_title(f"{label}{subtitle}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(image, ax=axes, shrink=0.8, label="equivalent strain")
    figure.suptitle(
        f"{arguments.size}x{arguments.size} px at ({row}, {column}), alpha = {arguments.alpha:g}, "
        + (
            f"epsilon = {arguments.epsilons[0]:g}"
            if sweeping_iterations
            else f"{arguments.iterations} iterations"
        )
        + f", shared scale {low:.2e} to {high:.2e}",
        fontsize=10,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    plt.close(figure)
    arguments.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "alpha": arguments.alpha,
                "iterations": arguments.iterations,
                "window_row_column": [row, column],
                "results": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        "utf-8",
    )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
