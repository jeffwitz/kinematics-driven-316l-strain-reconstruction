#!/usr/bin/env python
"""One observed-EVM image per computation, for side-by-side visual comparison.

Reads archived fields and plots them; nothing is recomputed, no mechanics and
no DISFlow.

Every image shares one colour scale, taken from **percentiles of the DIC** so a
few extreme pixels cannot stretch the range and flatten everything else, and a
**fixed layout**: the field occupies exactly the same pixels in every file, so
flipping through them in a file browser shows only what actually changes.

Written as PNG and SVG. Names sort as DIC first, then the matrix by alpha then
ell, then the two negative controls, so a directory listing is already a sweep.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OBS = ROOT / "validation/reference_data/p0043_matrix_observations_v1"
CTL = ROOT / "validation/reference_data/observed_evm_controls_p0043_v1"
SYM = ROOT / "validation/reference_data/dic_symmetric_observation_p0043_v1"

ALPHAS = (0.5, 1.0, 2.0, 4.0)
ELLS = (20.0, 40.0, 58.88, 90.0)
ARCHIVED = {(1.0, 58.88), (2.0, 58.88), (4.0, 58.88)}

#: The EVM has a long upper tail, so min-max limits would push almost every
#: pixel into the bottom of the map and hide the band structure.
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0

#: Fixed geometry, in figure fractions. Identical in every file so the eye can
#: flip between images without the field moving under it.
FIGURE_SIZE = (6.4, 6.0)
FIELD_RECT = (0.06, 0.06, 0.74, 0.82)
COLOURBAR_RECT = (0.83, 0.06, 0.035, 0.82)


def _tag_alpha(alpha: float) -> str:
    return f"a{alpha:.1f}".replace(".", "p")


def _tag_ell(ell: float) -> str:
    return f"ell{ell:06.2f}".replace(".", "p")


def evm_path(alpha: float, ell: float, profile: str) -> Path:
    label = f"a{alpha:g}-ell{ell:g}".replace(".", "p")
    name = f"archived-{label}" if (alpha, ell) in ARCHIVED else label
    return OBS / f"{name}_{profile}" / "fem_observed_evm.npy"


def write_image(
    field: np.ndarray,
    destination: Path,
    *,
    title: str,
    subtitle: str,
    vmin: float,
    vmax: float,
) -> None:
    """One field, fixed layout, both formats."""

    figure = plt.figure(figsize=FIGURE_SIZE)
    axis = figure.add_axes(FIELD_RECT)
    image = axis.imshow(field, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])
    bar_axis = figure.add_axes(COLOURBAR_RECT)
    figure.colorbar(image, cax=bar_axis)
    figure.text(0.06, 0.955, title, fontsize=13, va="center")
    figure.text(0.06, 0.915, subtitle, fontsize=8, va="center", color="0.3")
    for suffix in (".png", ".svg"):
        figure.savefig(destination.with_suffix(suffix), dpi=130)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="legacy_script_2021")
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()
    profile = arguments.profile
    output = Path(
        arguments.output
        or ROOT / f"validation/reference_data/p0043_small_parameter_matrix_v1/evm_fields_{profile}"
    )
    output.mkdir(parents=True, exist_ok=True)

    dic = np.load(SYM / "local_legacy_script_2021/dic_evm.npy", allow_pickle=False)
    vmin = float(np.percentile(dic, LOW_PERCENTILE))
    vmax = float(np.percentile(dic, HIGH_PERCENTILE))
    scale = (
        f"common scale {vmin:.4g} to {vmax:.4g}, "
        f"DIC percentiles {LOW_PERCENTILE:g}-{HIGH_PERCENTILE:g} "
        f"(DIC max {dic.max():.4g})"
    )
    print(f"clim {vmin:.4g} .. {vmax:.4g}   DIC range {dic.min():.4g} .. {dic.max():.4g}")

    written = 0
    write_image(
        dic,
        output / "evm_DIC",
        title="DIC, measured",
        subtitle=f"q95={np.percentile(dic, 95):.4g}   {scale}",
        vmin=vmin,
        vmax=vmax,
    )
    written += 1

    for alpha in ALPHAS:
        for ell in ELLS:
            path = evm_path(alpha, ell, profile)
            name = f"evm_{_tag_alpha(alpha)}_{_tag_ell(ell)}"
            if not path.is_file():
                print(f"  skipped {name}: no convergence")
                continue
            field = np.load(path, allow_pickle=False)
            archived = " (archived run)" if (alpha, ell) in ARCHIVED else ""
            write_image(
                field,
                output / name,
                title=f"alpha = {alpha:g},  ell = {ell:g} um{archived}",
                subtitle=f"q95={np.percentile(field, 95):.4g}   {scale}",
                vmin=vmin,
                vmax=vmax,
            )
            written += 1

    for control in ("homogeneous", "translated"):
        field = np.load(CTL / control / "fem_observed_evm.npy", allow_pickle=False)
        write_image(
            field,
            output / f"evm_zz_control_{control}",
            title=f"negative control, {control}",
            subtitle=f"q95={np.percentile(field, 95):.4g}   {scale}",
            vmin=vmin,
            vmax=vmax,
        )
        written += 1

    print(f"{written} fields written to {output} as PNG and SVG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
