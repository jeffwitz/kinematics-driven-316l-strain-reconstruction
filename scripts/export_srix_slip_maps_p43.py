"""Export per-slip-system TRI2 maps from a P43 SRIX field archive."""

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


def _system_label(index: int) -> str:
    burgers, normal = SLIP_SYSTEMS[index]
    b = ",".join(str(value) for value in burgers)
    n = ",".join(str(value) for value in normal)
    return f"{index + 1:02d}_b[{b}]_n[{n}]"


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _plot_maps(
    maps: np.ndarray,
    *,
    labels: list[str],
    title: str,
    output: Path,
    cmap: str,
    symmetric: bool,
) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
    finite = maps[np.isfinite(maps)]
    if finite.size == 0:
        lower = upper = 0.0
    elif symmetric:
        bound = float(np.max(np.abs(finite)))
        lower, upper = -bound, bound
    else:
        lower, upper = float(np.min(finite)), float(np.max(finite))
    image = None
    for index, axis in enumerate(axes.flat):
        image = axis.imshow(
            maps[index].T,
            origin="lower",
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(labels[index], fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title)
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.78)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_individual_maps(
    maps: np.ndarray,
    *,
    labels: list[str],
    prefix: str,
    output: Path,
    cmap: str,
    symmetric: bool,
) -> None:
    finite = maps[np.isfinite(maps)]
    if finite.size == 0:
        lower = upper = 0.0
    elif symmetric:
        bound = float(np.max(np.abs(finite)))
        lower, upper = -bound, bound
    else:
        lower, upper = float(np.min(finite)), float(np.max(finite))
    for index, label in enumerate(labels):
        figure, axis = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
        image = axis.imshow(
            maps[index].T,
            origin="lower",
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(label)
        axis.set_xlabel("pixel x")
        axis.set_ylabel("pixel y")
        figure.colorbar(image, ax=axis, label="slip amplitude")
        figure.savefig(output / f"{prefix}_{index + 1:02d}.png", dpi=180)
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    report = json.loads(arguments.report.read_text())
    with np.load(arguments.fields) as archive:
        required = {"plastic_slip", "equivalent_plastic_slip"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise SystemExit(
                "field archive does not contain per-system slip observables: "
                + ", ".join(missing)
            )
        plastic_slip = np.asarray(archive["plastic_slip"], dtype=np.float64)
        equivalent_slip = np.asarray(archive["equivalent_plastic_slip"], dtype=np.float64)

    expected = (100, 100, 2, 12)
    for name, values in {
        "plastic_slip": plastic_slip,
        "equivalent_plastic_slip": equivalent_slip,
    }.items():
        if values.shape != expected:
            raise SystemExit(f"{name} has shape {values.shape}, expected {expected}")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    labels = [_system_label(index) for index in range(12)]
    plastic_by_system = np.moveaxis(plastic_slip, -1, 0)
    equivalent_by_system = np.moveaxis(equivalent_slip, -1, 0)
    plastic_mean = plastic_by_system.mean(axis=3)
    equivalent_mean = equivalent_by_system.mean(axis=3)
    output_npz = arguments.output_dir / "srix_p43_m100_slip_maps.npz"
    np.savez_compressed(
        output_npz,
        plastic_slip_triangles=plastic_by_system,
        equivalent_plastic_slip_triangles=equivalent_by_system,
        plastic_slip_pixel_mean=plastic_mean,
        equivalent_plastic_slip_pixel_mean=equivalent_mean,
    )

    _plot_maps(
        plastic_mean,
        labels=labels,
        title="P43 100x100 SRIX: signed plastic slip, pixel mean",
        output=arguments.output_dir / "plastic_slip_pixel_mean_contact_sheet.png",
        cmap="coolwarm",
        symmetric=True,
    )
    _plot_maps(
        equivalent_mean,
        labels=labels,
        title="P43 100x100 SRIX: accumulated equivalent slip, pixel mean",
        output=arguments.output_dir / "equivalent_plastic_slip_pixel_mean_contact_sheet.png",
        cmap="magma",
        symmetric=False,
    )
    _write_individual_maps(
        plastic_mean,
        labels=labels,
        prefix="plastic_slip_pixel_mean_system",
        output=arguments.output_dir,
        cmap="coolwarm",
        symmetric=True,
    )
    _write_individual_maps(
        equivalent_mean,
        labels=labels,
        prefix="equivalent_plastic_slip_pixel_mean_system",
        output=arguments.output_dir,
        cmap="magma",
        symmetric=False,
    )

    metadata = {
        "status": "srix_p43_m100_slip_maps_exported",
        "source_report": str(arguments.report),
        "source_fields": str(arguments.fields),
        "source_field_hashes": {
            "plastic_slip": _hash(plastic_slip),
            "equivalent_plastic_slip": _hash(equivalent_slip),
        },
        "mesh": report.get("mesh"),
        "crop_nodes": report.get("crop_nodes"),
        "orientation": report.get("orientation"),
        "behaviour": report.get("behaviour"),
        "paired_parameter_set": report.get("crystal_material", {}).get(
            "paired_parameter_set"
        ),
        "units": {
            "plastic_slip": "dimensionless signed slip amplitude",
            "equivalent_plastic_slip": "dimensionless accumulated slip amplitude",
        },
        "raw_shape": list(expected),
        "exported_npz_shapes": {
            "*_triangles": [12, 100, 100, 2],
            "*_pixel_mean": [12, 100, 100],
        },
        "slip_system_order": labels,
        "note": (
            "TRI2 retains two independent material states per pixel; pixel_mean "
            "is the arithmetic mean of the two triangle fields."
        ),
    }
    (arguments.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
