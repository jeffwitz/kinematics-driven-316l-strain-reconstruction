"""Plot DIC, homogeneous and EBSD equivalent-strain maps for P43 M200."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fem_inhouse.postprocessing.kinematics import (
    plane_stress_equivalent_strain,
    strain_from_displacement,
)

ROOT = Path(__file__).resolve().parents[1]
PERF = ROOT / "validation/_generated/performance"
DATA = ROOT / "data/processed/case_study"


def _displacement(path: Path) -> np.ndarray:
    with np.load(path) as arrays:
        return np.asarray(arrays["displacement"], dtype=float)


def _evm(displacement: np.ndarray, spacing: float) -> np.ndarray:
    strain = strain_from_displacement(
        displacement[..., 0],
        displacement[..., 1],
        spacing_x=spacing,
        spacing_y=spacing,
    )
    return plane_stress_equivalent_strain(
        strain.epsilon_xx,
        strain.epsilon_yy,
        strain.gamma_xy,
        poisson_ratio=0.30,
        shear_convention="engineering",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behaviour", choices=("srix", "meric"), default="srix")
    arguments = parser.parse_args()
    if arguments.behaviour == "meric":
        stem = "meric_p43_m200"
        title_prefix = "Méric"
    else:
        stem = "srix_p43_m200"
        title_prefix = "SRIX"
    crop = (1520, 1720, 985, 1185)
    x0, x1, y0, y1 = crop
    spacing = 0.00184
    dic = np.stack(
        (
            np.load(DATA / "displacement_x_mm.npy", mmap_mode="r")[x0 : x1 + 1, y0 : y1 + 1],
            np.load(DATA / "displacement_y_mm.npy", mmap_mode="r")[x0 : x1 + 1, y0 : y1 + 1],
        ),
        axis=-1,
    )
    fields = {
        "dic": _evm(dic, spacing),
        "homogeneous": _evm(
            _displacement(PERF / f"{stem}_homogeneous_structural_fd.fields.npz"),
            spacing,
        ),
        "ebsd": _evm(
            _displacement(PERF / f"{stem}_ebsd_structural_fd.fields.npz"),
            spacing,
        ),
    }
    fields["homogeneous_minus_dic"] = fields["homogeneous"] - fields["dic"]
    fields["ebsd_minus_dic"] = fields["ebsd"] - fields["dic"]
    fields["ebsd_minus_homogeneous"] = fields["ebsd"] - fields["homogeneous"]

    output_npz = PERF / f"{stem}_equivalent_strain_maps.npz"
    np.savez_compressed(output_npz, **fields)

    common_min = float(min(np.min(fields[name]) for name in ("dic", "homogeneous", "ebsd")))
    common_max = float(max(np.max(fields[name]) for name in ("dic", "homogeneous", "ebsd")))
    diff_max = float(
        max(
            np.max(np.abs(fields[name]))
            for name in ("homogeneous_minus_dic", "ebsd_minus_dic", "ebsd_minus_homogeneous")
        )
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = (
        ("dic", "DIC equivalent strain"),
        ("homogeneous", f"{title_prefix} - homogeneous orientation"),
        ("ebsd", f"{title_prefix} - EBSD orientation"),
        ("homogeneous_minus_dic", "Homogeneous - DIC"),
        ("ebsd_minus_dic", "EBSD - DIC"),
        ("ebsd_minus_homogeneous", "EBSD - homogeneous"),
    )
    for index, (name, title) in enumerate(panels):
        axis = axes.flat[index]
        values = fields[name]
        if "minus" in name:
            image = axis.imshow(
                values.T * 100.0,
                origin="lower",
                cmap="coolwarm",
                vmin=-100.0 * diff_max,
                vmax=100.0 * diff_max,
                aspect="equal",
            )
            label = "percentage points"
        else:
            image = axis.imshow(
                values.T * 100.0,
                origin="lower",
                cmap="viridis",
                vmin=100.0 * common_min,
                vmax=100.0 * common_max,
                aspect="equal",
            )
            label = "%"
        axis.set_title(title)
        axis.set_xlabel("x node index")
        axis.set_ylabel("y node index")
        figure.colorbar(image, ax=axis, label=label)
    output_png = PERF / f"{stem}_equivalent_strain_maps.png"
    figure.savefig(output_png, dpi=220)
    plt.close(figure)

    summary = {
        "status": "completed_m200_equivalent_strain_maps",
        "behaviour": arguments.behaviour,
        "crop_nodes": list(crop),
        "spacing_mm": spacing,
        "poisson_ratio": 0.30,
        "measure": "plane-stress von Mises equivalent total strain",
        "shear_convention": "engineering",
        "common_min": common_min,
        "common_max": common_max,
        "difference_abs_max": diff_max,
        "npz": str(output_npz),
        "png": str(output_png),
    }
    (PERF / f"{stem}_equivalent_strain_maps.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
