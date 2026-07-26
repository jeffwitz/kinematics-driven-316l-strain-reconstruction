#!/usr/bin/env python3
"""Generate the strictly local four-panel defect figure used by the science narrative."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from fem_inhouse.workflows.coupled_alpha_visualization import (
    prepare_coupled_alpha_fields,
)

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def generate(
    *,
    input_directory: Path,
    local_campaign: Path,
    comparison_campaigns: tuple[Path, Path, Path],
    partition_id: int,
    output: Path,
) -> None:
    """Reconstruct DIC/local EVM and local PEEQ, then render one common provenance figure."""
    data, _metadata = prepare_coupled_alpha_fields(
        input_directory=input_directory,
        campaigns=(local_campaign, *comparison_campaigns),
        partition_id=partition_id,
        alpha_values=(0.0, 1.0, 2.0, 4.0),
    )
    dic = data.dic_evm
    local = data.evm_by_alpha[0]
    error = local - dic
    peeq = data.peeq_by_alpha[0]
    evm_max = float(max(np.max(dic), np.max(local)))
    error_max = float(np.max(np.abs(error)))
    extent = data.extent_mm
    figure, axes = plt.subplots(1, 4, figsize=(15, 3.7), constrained_layout=True)
    image = None
    for axis, field, title in zip(
        axes[:2], (dic, local), ("DIC total EVM", "Local FEM total EVM"), strict=True
    ):
        image = axis.imshow(
            field.T,
            origin="lower",
            extent=extent,
            vmin=0.0,
            vmax=evm_max,
            cmap="viridis",
            aspect="equal",
        )
        axis.set_title(title)
    assert image is not None
    figure.colorbar(image, ax=axes[:2], label="Total equivalent strain, EVM", shrink=0.78)
    difference_image = axes[2].imshow(
        error.T,
        origin="lower",
        extent=extent,
        vmin=-error_max,
        vmax=error_max,
        cmap="coolwarm",
        aspect="equal",
    )
    axes[2].set_title("Local FEM - DIC")
    figure.colorbar(difference_image, ax=axes[2], label="EVM difference", shrink=0.78)
    peeq_image = axes[3].imshow(
        peeq.T,
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=float(np.max(peeq)),
        cmap="magma",
        aspect="equal",
    )
    axes[3].set_title("Local PEEQ (internal)")
    figure.colorbar(peeq_image, ax=axes[3], label="PEEQ", shrink=0.78)
    for axis in axes:
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--local-campaign", type=Path, required=True)
    parser.add_argument("--comparison-campaign", type=Path, action="append", required=True)
    parser.add_argument("--partition-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.comparison_campaign) != 3:
        parser.error("--comparison-campaign must be supplied exactly three times")
    generate(
        input_directory=args.input,
        local_campaign=args.local_campaign,
        comparison_campaigns=tuple(args.comparison_campaign),
        partition_id=args.partition_id,
        output=args.output,
    )


if __name__ == "__main__":
    main()
