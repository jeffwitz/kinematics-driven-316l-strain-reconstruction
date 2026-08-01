#!/usr/bin/env python
"""One DIC EVM image per partition, for choosing the next ROI by eye.

The DIC side of the qualification needs **no mechanics at all**: it is the
measured displacement through the historical EVM operator. This writes it for
any set of partitions so a ROI can be judged before a single FEM run.

Same conventions as the P43 sweep: colour limits from percentiles of that
partition's own DIC, fixed layout so files can be flipped, PNG and SVG.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fem_inhouse.partitioning import PartitionLayout
from fem_inhouse.workflows.nonlocality_diagnostic import reconstruct_historical_evm

ROOT = Path(__file__).resolve().parent.parent
PREPARED = ROOT / "data/processed/case_study"
RANKING = ROOT / "validation/dic_partition_heterogeneity_10x10.json"
PIXEL_SIZE_MM = 0.00184
POISSON_RATIO = 0.3

LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0
FIGURE_SIZE = (6.4, 6.0)
FIELD_RECT = (0.06, 0.06, 0.74, 0.82)
COLOURBAR_RECT = (0.83, 0.06, 0.035, 0.82)

#: Measured MTF-50 of the chain, the length below which a band is not resolved.
MTF50_PIXELS = 49.0


def dic_evm(partition) -> np.ndarray:
    """Measured EVM on one partition core, through the historical operator.

    The ranking file records only the core, so the padded solve bounds come
    from the layout rather than being guessed.
    """

    sx0, sx1, sy0, sy1 = (int(v) for v in partition.solve_bounds)
    cx0, cx1, cy0, cy1 = (int(v) for v in partition.core_bounds)
    ux = np.load(PREPARED / "displacement_x_mm.npy", mmap_mode="r", allow_pickle=False)
    uy = np.load(PREPARED / "displacement_y_mm.npy", mmap_mode="r", allow_pickle=False)
    displacement = np.stack(
        (
            np.asarray(ux[sx0 : sx1 + 1, sy0 : sy1 + 1], dtype=np.float64),
            np.asarray(uy[sx0 : sx1 + 1, sy0 : sy1 + 1], dtype=np.float64),
        ),
        axis=-1,
    )
    evm = reconstruct_historical_evm(
        displacement,
        spacing_x_mm=PIXEL_SIZE_MM,
        spacing_y_mm=PIXEL_SIZE_MM,
        poisson_ratio=POISSON_RATIO,
    )
    return np.ascontiguousarray(evm[cx0 - sx0 : cx1 - sx0, cy0 - sy0 : cy1 - sy0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partitions",
        default="17,84,58,44,92,7,43,48,53,80",
        help="comma separated partition ids; default is the top eight plus the two widest",
    )
    parser.add_argument(
        "--output", default=str(ROOT / "validation/reference_data/dic_evm_partitions_v1")
    )
    arguments = parser.parse_args()
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)

    ranking = json.loads(RANKING.read_text(encoding="utf-8"))
    by_id = {int(p["partition_id"]): p for p in ranking["partitions"]}
    layout = PartitionLayout(
        tuple(ranking["global_shape"]),
        tuple(ranking["partition_shape"]),
        padding=int(ranking["padding"]),
    )
    wanted = [int(v) for v in arguments.partitions.split(",")]

    for partition_id in wanted:
        entry = by_id[partition_id]
        field = dic_evm(layout.get(partition_id))
        vmin = float(np.percentile(field, LOW_PERCENTILE))
        vmax = float(np.percentile(field, HIGH_PERCENTILE))

        figure = plt.figure(figsize=FIGURE_SIZE)
        axis = figure.add_axes(FIELD_RECT)
        image = axis.imshow(field, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, cax=figure.add_axes(COLOURBAR_RECT))
        width = entry["band_average_width_px"]
        figure.text(
            0.06,
            0.955,
            f"DIC EVM, partition {partition_id:03d}",
            fontsize=13,
            va="center",
        )
        figure.text(
            0.06,
            0.918,
            f"rank score {entry['band_score']:.2f}   aspect {entry['band_aspect_ratio']:.2f}   "
            f"edge contacts {entry['band_boundary_contacts']}",
            fontsize=8,
            va="center",
            color="0.3",
        )
        figure.text(
            0.06,
            0.893,
            f"selection width {width:.1f} px = {width / MTF50_PIXELS:.2f} MTF-50   "
            f"clim {vmin:.3g}-{vmax:.3g}",
            fontsize=8,
            va="center",
            color="crimson" if width < MTF50_PIXELS else "0.3",
        )
        base = output / f"dic_evm_p{partition_id:03d}"
        for suffix in (".png", ".svg"):
            figure.savefig(base.with_suffix(suffix), dpi=130)
        plt.close(figure)
        print(
            f"  p{partition_id:03d}  score {entry['band_score']:5.2f}  "
            f"width {width:5.1f} px ({width / MTF50_PIXELS:.2f} MTF-50)  "
            f"aspect {entry['band_aspect_ratio']:4.2f}  contacts {entry['band_boundary_contacts']}"
        )

    print(f"\n{len(wanted)} partitions written to {output} as PNG and SVG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
