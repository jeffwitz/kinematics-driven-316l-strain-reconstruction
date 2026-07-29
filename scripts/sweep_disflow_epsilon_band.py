#!/usr/bin/env python3
"""Sweep Charbonnier epsilon on the pre-registered 32 px synthetic EVM band."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image

from fem_inhouse.measurement import DISFlowConfig, run_disflow, warp_image
from fem_inhouse.workflows.dic_measurement_chain import (
    BAND_WIDTHS,
    CROP_COLUMNS,
    CROP_ROWS,
    PIXEL_SIZE_UM,
    TRANSFER_SIZE,
    _band_displacement,
    _central_window,
    _half_maximum_width,
    image_flow_to_historical_evm,
    queried_disflow_configuration,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

DEFAULT_EPSILONS = (0.0002, 0.002, 0.02, 0.2, 2.0)
BAND_WIDTH_PIXELS = max(BAND_WIDTHS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _load_reference(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"), dtype=np.uint8)
    if values.shape != (4_400, 5_400):
        raise ValueError(f"unexpected reference image shape: {values.shape}")
    crop = np.ascontiguousarray(values[CROP_ROWS, CROP_COLUMNS])
    return _central_window(crop, size=TRANSFER_SIZE)


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _normal_profile(evm: np.ndarray) -> np.ndarray:
    return np.asarray(np.median(evm, axis=1), dtype=np.float64)


def _plot(
    output: Path,
    *,
    cases: list[dict[str, Any]],
    imposed_map: np.ndarray,
    imposed_profile: np.ndarray,
    coordinate_um: np.ndarray,
) -> None:
    figure, axes = plt.subplots(
        len(cases),
        2,
        figsize=(12, 3.1 * len(cases)),
        constrained_layout=True,
    )
    centre = int(np.argmax(imposed_profile))
    along_centre = imposed_map.shape[1] // 2
    half_extent = 96
    normal_slice = slice(centre - half_extent, centre + half_extent)
    along_slice = slice(along_centre - half_extent, along_centre + half_extent)
    imposed_crop = imposed_map[normal_slice, along_slice]
    common_maximum = max(
        float(np.max(imposed_crop)),
        *(float(np.max(case["evm_crop"])) for case in cases),
    )
    extent_um = half_extent * PIXEL_SIZE_UM
    map_extent = (-extent_um, extent_um, -extent_um, extent_um)
    width_um = BAND_WIDTH_PIXELS * PIXEL_SIZE_UM
    step = np.where(
        np.abs(coordinate_um) <= 0.5 * width_um,
        float(np.max(imposed_profile)),
        0.0,
    )

    for row_index, case in enumerate(cases):
        image = axes[row_index, 0].imshow(
            case["evm_crop"].T,
            origin="lower",
            extent=map_extent,
            cmap="magma",
            vmin=0.0,
            vmax=common_maximum,
            aspect="equal",
        )
        axes[row_index, 0].axhline(0.0, color="cyan", linewidth=1.2)
        axes[row_index, 0].set_title(
            f"Recovered EVM — epsilon={case['epsilon']:g}\n"
            f"FWHM={case['fwhm_pixels']:.0f} px, "
            f"along-band CV={case['along_band_peak_cv']:.3f}"
        )
        axes[row_index, 0].set_xlabel("Normal coordinate (µm)")
        axes[row_index, 0].set_ylabel("Along-band coordinate (µm)")
        figure.colorbar(image, ax=axes[row_index, 0], label="Total equivalent strain, EVM")

        axes[row_index, 1].plot(
            coordinate_um,
            imposed_profile,
            color="black",
            linewidth=1.8,
            label="Exact imposed Gaussian EVM",
        )
        axes[row_index, 1].plot(
            coordinate_um,
            case["profile"],
            color="#d95f02",
            linewidth=1.6,
            label="Recovered EVM",
        )
        axes[row_index, 1].step(
            coordinate_um,
            step,
            where="mid",
            color="#1b9e77",
            linestyle="--",
            label="32 px FWHM reference step",
        )
        axes[row_index, 1].set_xlim(-4.0 * width_um, 4.0 * width_um)
        axes[row_index, 1].set_xlabel("Coordinate normal to the band (µm)")
        axes[row_index, 1].set_ylabel("Total equivalent strain, EVM")
        axes[row_index, 1].grid(alpha=0.25)
        axes[row_index, 1].legend(fontsize=8)
        secondary = axes[row_index, 1].secondary_xaxis(
            "top",
            functions=(
                lambda values: values / PIXEL_SIZE_UM,
                lambda values: values * PIXEL_SIZE_UM,
            ),
        )
        secondary.set_xlabel("Coordinate normal to the band (px)")

    figure.suptitle(
        "Charbonnier epsilon sensitivity — 32 px synthetic EVM band\n"
        "All maps share one colour scale; cyan line is the normal section"
    )
    figure.savefig(output / "epsilon_band32_evm_sections.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    epsilons = np.asarray([case["epsilon"] for case in cases])
    axes[0].semilogx(epsilons, [case["fwhm_pixels"] for case in cases], marker="o")
    axes[0].axhline(BAND_WIDTH_PIXELS, color="black", linestyle="--")
    axes[0].set_ylabel("Recovered FWHM (px)")
    axes[1].semilogx(epsilons, [case["peak_gain"] for case in cases], marker="o")
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set_ylabel("Recovered peak / imposed peak")
    axes[2].semilogx(
        epsilons,
        [case["along_band_peak_cv"] for case in cases],
        marker="o",
    )
    axes[2].set_ylabel("Along-band peak coefficient of variation")
    for axis in axes:
        axis.axvline(0.002, color="0.4", linestyle=":", label="production epsilon")
        axis.set_xlabel("Charbonnier epsilon")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Width, amplitude and waviness must be read together")
    figure.savefig(output / "epsilon_band32_metrics.png", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    _prepare_directory(output, overwrite=args.overwrite)
    reference_path = Path(args.reference_image)
    reference = _load_reference(reference_path)
    imposed, _ = _band_displacement(
        reference.shape,
        width_pixels=BAND_WIDTH_PIXELS,
        orientation="horizontal",
    )
    deformed = warp_image(reference, imposed)
    imposed_evm = image_flow_to_historical_evm(imposed)
    imposed_profile = _normal_profile(imposed_evm)
    centre = int(np.argmax(imposed_profile))
    coordinate_um = (
        np.arange(imposed_profile.size, dtype=np.float64) - centre
    ) * PIXEL_SIZE_UM
    half_extent = 96
    along_centre = imposed_evm.shape[1] // 2
    normal_slice = slice(centre - half_extent, centre + half_extent)
    along_slice = slice(along_centre - half_extent, along_centre + half_extent)
    imposed_core = imposed_evm[normal_slice, 16:-16]

    rows: list[dict[str, float]] = []
    cases: list[dict[str, Any]] = []
    queried: dict[str, dict[str, Any]] = {}
    for epsilon in args.epsilons:
        config = DISFlowConfig(variational_refinement_epsilon=epsilon)
        queried[str(epsilon)] = queried_disflow_configuration(config)
        recovered = run_disflow(reference, deformed, config=config)
        evm = image_flow_to_historical_evm(recovered)
        profile = _normal_profile(evm)
        fwhm, peak_index = _half_maximum_width(profile)
        evm_core = evm[normal_slice, 16:-16]
        along_peak = np.max(evm_core, axis=0)
        relative_l2 = float(
            np.linalg.norm(evm_core - imposed_core) / np.linalg.norm(imposed_core)
        )
        row = {
            "epsilon": float(epsilon),
            "fwhm_pixels": fwhm,
            "fwhm_um": fwhm * PIXEL_SIZE_UM,
            "peak_gain": float(np.max(profile) / np.max(imposed_profile)),
            "centroid_shift_pixels": float(peak_index - centre),
            "relative_l2": relative_l2,
            "along_band_peak_mean": float(np.mean(along_peak)),
            "along_band_peak_standard_deviation": float(np.std(along_peak)),
            "along_band_peak_cv": float(np.std(along_peak) / np.mean(along_peak)),
        }
        rows.append(row)
        cases.append(
            {
                **row,
                "profile": profile,
                "evm_crop": evm[normal_slice, along_slice],
            }
        )

    _write_csv(output / "epsilon_band32_metrics.csv", rows)
    _plot(
        output,
        cases=cases,
        imposed_map=imposed_evm,
        imposed_profile=imposed_profile,
        coordinate_um=coordinate_um,
    )
    manifest = {
        "schema_version": 1,
        "status": "completed_exploratory_sensitivity",
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "reference_image": {
            "path": str(reference_path.resolve()),
            "sha256": _sha256(reference_path),
        },
        "fixed": {
            "band_width_pixels": BAND_WIDTH_PIXELS,
            "pixel_size_um": PIXEL_SIZE_UM,
            "maximum_displacement_pixels": 1.5,
            "finest_scale": 0,
        },
        "epsilons": list(args.epsilons),
        "queried_disflow": queried,
        "metrics": rows,
        "claim_boundary": "measurement-chain sensitivity only; no parameter selected",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--reference-image", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--epsilons", type=float, nargs="+", default=DEFAULT_EPSILONS)
    result.add_argument("--overwrite", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
