#!/usr/bin/env python3
"""Incremental DIC, elastic and TANN equivalent strains from one artifact.

All three columns are reconstructed from the displacement fields stored by
the run and share the same reference state.  The earlier version mixed an
absolute archived DIC EVM with FEM increments from state 20; its spatial
comparison was invalid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from compare_disflow_profiles_p43 import equivalent_strain  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "validation/_generated/shared_tensor_generator/tann_fcc_p43_run.npz"
PIXEL_MM = 1.84e-3
STATES = (25, 32, 40)


def computed_evm(displacement: np.ndarray) -> np.ndarray:
    """The archived definition on a nodal `(nx+1, ny+1, 2)` field in mm.

    The first difference of a mm field gives mm/pixel; the strain is the
    difference divided by the pixel size -- the archived pipeline stores
    pixel displacements, so its first difference is already dimensionless.
    Omitting the pixel size makes the computed EVM ~550x too small, which
    a Dirichlet comparison forbids (the boundary strains must be of the
    same order by construction).
    """

    along_rows = displacement[..., 1] / PIXEL_MM  # u_y -> pixels
    along_columns = displacement[..., 0] / PIXEL_MM  # u_x -> pixels
    return equivalent_strain(along_rows, along_columns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation/figures/tann_fcc_p43/EVM_incremental_diagnostic.png",
    )
    arguments = parser.parse_args()

    fields = dict(np.load(arguments.artifact, allow_pickle=False))

    figure, axes = plt.subplots(len(STATES), 3, figsize=(11.5, 9.8))
    for row, state in enumerate(STATES):
        measured = computed_evm(fields[f"state_{state}_u_meas"])
        elastic = computed_evm(fields[f"state_{state}_u_elastic"])
        tann = computed_evm(fields[f"state_{state}_u_sim"])
        vmax = float(np.percentile(np.stack([measured, elastic, tann]), 99.5))
        for column, (name, field) in enumerate(
            (("DIC increment", measured), ("elastic increment", elastic), ("TANN increment", tann))
        ):
            axis = axes[row, column]
            image = axis.imshow(field.T, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax)
            axis.set_title(f"{name}, state {state}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle(
        "Equivalent strain increments — historical primary run, invalid for model claims",
        fontsize=11,
    )
    figure.tight_layout()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
