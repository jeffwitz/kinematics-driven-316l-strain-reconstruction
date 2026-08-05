"""Create side-by-side SRIX/Méric slip maps for the same P43 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fem_inhouse.core.fcc_interaction_matrix import SLIP_SYSTEMS


def _label(index: int) -> str:
    burgers, normal = SLIP_SYSTEMS[index]
    return (
        f"{index + 1:02d}  b[{','.join(map(str, burgers))}] "
        f"n[{','.join(map(str, normal))}]"
    )


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _load_mean(fields: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(fields) as archive:
        values = np.asarray(archive[key], dtype=np.float64)
    expected = (100, 100, 2, 12)
    if values.shape != expected:
        raise SystemExit(f"{fields}: {key} has shape {values.shape}, expected {expected}")
    triangles = np.moveaxis(values, -1, 0)
    return triangles, triangles.mean(axis=3)


def _plot_pair(
    srix: np.ndarray,
    meric: np.ndarray,
    *,
    labels: list[str],
    title: str,
    output: Path,
    cmap: str,
    symmetric: bool,
) -> None:
    figure, axes = plt.subplots(12, 2, figsize=(9, 29), constrained_layout=True)
    finite = np.concatenate((srix[np.isfinite(srix)], meric[np.isfinite(meric)]))
    if symmetric:
        bound = float(np.max(np.abs(finite))) if finite.size else 0.0
        lower, upper = -bound, bound
    else:
        lower = float(np.min(finite)) if finite.size else 0.0
        upper = float(np.max(finite)) if finite.size else 0.0
    image = None
    for index, label in enumerate(labels):
        for column, (law, values) in enumerate((("SRIX", srix), ("Méric", meric))):
            axis = axes[index, column]
            image = axis.imshow(
                values[index].T,
                origin="lower",
                cmap=cmap,
                vmin=lower,
                vmax=upper,
                interpolation="nearest",
                aspect="equal",
            )
            axis.set_title(f"{label} — {law}", fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(title)
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.45)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_individual_pairs(
    srix: np.ndarray,
    meric: np.ndarray,
    *,
    labels: list[str],
    prefix: str,
    output: Path,
    cmap: str,
    symmetric: bool,
) -> None:
    finite = np.concatenate((srix[np.isfinite(srix)], meric[np.isfinite(meric)]))
    if symmetric:
        bound = float(np.max(np.abs(finite))) if finite.size else 0.0
        lower, upper = -bound, bound
    else:
        lower = float(np.min(finite)) if finite.size else 0.0
        upper = float(np.max(finite)) if finite.size else 0.0
    for index, label in enumerate(labels):
        figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        image = None
        for axis, law, values in zip(axes, ("SRIX", "Méric"), (srix, meric), strict=True):
            image = axis.imshow(
                values[index].T,
                origin="lower",
                cmap=cmap,
                vmin=lower,
                vmax=upper,
                interpolation="nearest",
                aspect="equal",
            )
            axis.set_title(f"{label} — {law}")
            axis.set_xlabel("pixel x")
            axis.set_ylabel("pixel y")
        if image is not None:
            figure.colorbar(image, ax=axes, label="slip amplitude")
        figure.savefig(output / f"{prefix}_{index + 1:02d}.png", dpi=180)
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--srix-fields", type=Path, required=True)
    parser.add_argument("--meric-fields", type=Path, required=True)
    parser.add_argument("--srix-report", type=Path, required=True)
    parser.add_argument("--meric-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    srix_report = json.loads(arguments.srix_report.read_text())
    meric_report = json.loads(arguments.meric_report.read_text())
    for key in ("crop_nodes", "mesh", "orientation", "increments"):
        if srix_report.get(key) != meric_report.get(key):
            raise SystemExit(f"reports do not match for {key}")
    srix_backbone = srix_report["crystal_material"]["backbone"]["sha256"]
    meric_backbone = meric_report["crystal_material"]["backbone"]["sha256"]
    if srix_backbone != meric_backbone:
        raise SystemExit("reports do not use the same 316L backbone")

    srix_signed_tri, srix_signed = _load_mean(arguments.srix_fields, "plastic_slip")
    meric_signed_tri, meric_signed = _load_mean(arguments.meric_fields, "plastic_slip")
    srix_equiv_tri, srix_equiv = _load_mean(
        arguments.srix_fields, "equivalent_plastic_slip"
    )
    meric_equiv_tri, meric_equiv = _load_mean(
        arguments.meric_fields, "equivalent_plastic_slip"
    )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    labels = [_label(index) for index in range(12)]
    _plot_pair(
        srix_equiv,
        meric_equiv,
        labels=labels,
        title="P43 100x100, 16 increments: accumulated equivalent slip",
        output=arguments.output_dir / "equivalent_plastic_slip_srix_vs_meric.png",
        cmap="magma",
        symmetric=False,
    )
    _plot_pair(
        srix_signed,
        meric_signed,
        labels=labels,
        title="P43 100x100, 16 increments: signed plastic slip",
        output=arguments.output_dir / "plastic_slip_srix_vs_meric.png",
        cmap="coolwarm",
        symmetric=True,
    )
    _plot_individual_pairs(
        srix_equiv,
        meric_equiv,
        labels=labels,
        prefix="equivalent_plastic_slip_system",
        output=arguments.output_dir,
        cmap="magma",
        symmetric=False,
    )
    _plot_individual_pairs(
        srix_signed,
        meric_signed,
        labels=labels,
        prefix="plastic_slip_system",
        output=arguments.output_dir,
        cmap="coolwarm",
        symmetric=True,
    )
    np.savez_compressed(
        arguments.output_dir / "srix_vs_meric_slip_maps.npz",
        srix_plastic_slip_triangles=srix_signed_tri,
        meric_plastic_slip_triangles=meric_signed_tri,
        srix_equivalent_plastic_slip_triangles=srix_equiv_tri,
        meric_equivalent_plastic_slip_triangles=meric_equiv_tri,
        srix_plastic_slip_pixel_mean=srix_signed,
        meric_plastic_slip_pixel_mean=meric_signed,
        srix_equivalent_plastic_slip_pixel_mean=srix_equiv,
        meric_equivalent_plastic_slip_pixel_mean=meric_equiv,
    )

    metadata = {
        "status": "srix_meric_p43_m100_slip_maps_compared",
        "srix_report": str(arguments.srix_report),
        "meric_report": str(arguments.meric_report),
        "mesh": srix_report["mesh"],
        "crop_nodes": srix_report["crop_nodes"],
        "increments": srix_report["increments"],
        "orientation": srix_report["orientation"],
        "paired_parameter_set": srix_report["crystal_material"]["paired_parameter_set"],
        "backbone_sha256": srix_backbone,
        "slip_system_order": labels,
        "raw_shape": [100, 100, 2, 12],
        "comparison": (
            "Both laws use the same crop, orientation, paired 316L backbone, "
            "and 16 proportional increments. The pixel mean averages the two TRI2 states."
        ),
        "field_hashes": {
            "srix_equivalent_plastic_slip": _hash(srix_equiv_tri),
            "meric_equivalent_plastic_slip": _hash(meric_equiv_tri),
            "srix_plastic_slip": _hash(srix_signed_tri),
            "meric_plastic_slip": _hash(meric_signed_tri),
        },
        "max_pixel_mean_equivalent_slip": {
            "srix": float(np.max(srix_equiv)),
            "meric": float(np.max(meric_equiv)),
        },
    }
    (arguments.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
