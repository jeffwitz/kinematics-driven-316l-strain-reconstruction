"""Independent EBSD/Schmid structural-length measurement workflow."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

from fem_inhouse.postprocessing.spatial_correlation import (
    CorrelationProfile,
    DecayFit,
    StructuralCorrelationResult,
    structural_correlation,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_dict(fit: DecayFit, spacing_um: float) -> dict[str, Any]:
    data = asdict(fit)
    data["length_um"] = fit.length_pixels * spacing_um
    return data


def _profile_rows(profile: CorrelationProfile, spacing_um: float) -> np.ndarray:
    return np.column_stack(
        (
            profile.distance_pixels,
            profile.distance_pixels * spacing_um,
            profile.correlation,
            profile.pair_weight,
        )
    )


def _bootstrap_median(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True)
    medians = np.median(draws, axis=1)
    percentiles = np.percentile(medians, (2.5, 50.0, 97.5))
    return {
        "lower_2p5_um": float(percentiles[0]),
        "median_um": float(percentiles[1]),
        "upper_97p5_um": float(percentiles[2]),
        "sample_count": float(samples),
        "valid_block_count": float(values.size),
        "seed": float(seed),
    }


def _plot_profiles(
    result: StructuralCorrelationResult,
    *,
    spacing_um: float,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for label, profile, fit in (
        ("radial", result.radial, result.radial_decay),
        ("x direction", result.x_direction, result.x_decay),
        ("y direction", result.y_direction, result.y_decay),
    ):
        distance = profile.distance_pixels * spacing_um
        axis.plot(distance, profile.correlation, label=label)
        selected = slice(fit.first_index, fit.last_index + 1)
        fitted = np.exp(
            fit.intercept + fit.slope_per_pixel * profile.distance_pixels[selected]
        )
        axis.plot(distance[selected], fitted, linestyle="--", linewidth=1.2)
    axis.axhspan(0.15, 0.60, color="0.9", zorder=-1, label="frozen fit interval")
    axis.axhline(0.0, color="0.25", linewidth=0.8)
    axis.set(xlabel="lag (µm)", ylabel="normalized autocorrelation", ylim=(-0.15, 1.02))
    axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def measure_ebsd_structural_length(
    input_path: Path,
    output_directory: Path,
    *,
    overwrite: bool = False,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20_260_727,
) -> dict[str, Any]:
    """Measure a preregistered structural length and write reproducible artefacts."""

    if output_directory.exists() and any(output_directory.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as handle:
        schmid_dataset = handle["/schmid/max_schmid_factor"]
        schmid = np.asarray(schmid_dataset, dtype=np.float64)
        phi1 = np.asarray(handle["/orientation/phi1"], dtype=np.float64)
        phi = np.asarray(handle["/orientation/Phi"], dtype=np.float64)
        phi2 = np.asarray(handle["/orientation/phi2"], dtype=np.float64)
        spacing_um = float(schmid_dataset.attrs["pixel_size_um"])
    if schmid.shape != phi1.shape or schmid.shape != phi.shape or schmid.shape != phi2.shape:
        raise ValueError("Schmid and Euler fields are not co-registered")
    orientation_zero = (phi1 == 0.0) & (phi == 0.0) & (phi2 == 0.0)
    mask = np.isfinite(schmid) & (schmid > 0.0) & (schmid <= 0.5) & ~orientation_zero
    maximum_lag = min(schmid.shape) // 4
    result = structural_correlation(
        schmid,
        valid_mask=mask,
        maximum_lag_pixels=maximum_lag,
    )

    block_lengths: list[float] = []
    block_records: list[dict[str, Any]] = []
    x_edges = np.linspace(0, schmid.shape[0], 5, dtype=int)
    y_edges = np.linspace(0, schmid.shape[1], 5, dtype=int)
    for ix in range(4):
        for iy in range(4):
            bounds = tuple(
                int(value)
                for value in (x_edges[ix], x_edges[ix + 1], y_edges[iy], y_edges[iy + 1])
            )
            block_field = schmid[bounds[0] : bounds[1], bounds[2] : bounds[3]]
            block_mask = mask[bounds[0] : bounds[1], bounds[2] : bounds[3]]
            record: dict[str, Any] = {
                "block": [ix, iy],
                "bounds_pixels": list(bounds),
                "valid_fraction": float(np.mean(block_mask)),
            }
            try:
                block_result = structural_correlation(block_field, valid_mask=block_mask)
            except ValueError as error:
                record.update(status="invalid", error=str(error))
            else:
                length_um = block_result.radial_decay.length_pixels * spacing_um
                block_lengths.append(length_um)
                record.update(status="valid", radial_decay_length_um=length_um)
            block_records.append(record)
    if not block_lengths:
        raise ValueError("none of the preregistered spatial blocks produced a valid fit")
    bootstrap = _bootstrap_median(
        np.asarray(block_lengths, dtype=np.float64),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    x_um = result.x_decay.length_pixels * spacing_um
    y_um = result.y_decay.length_pixels * spacing_um
    anisotropy = max(x_um, y_um) / min(x_um, y_um)
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input": {
            "path": str(input_path.resolve()),
            "sha256": _sha256(input_path),
            "dataset": "/schmid/max_schmid_factor",
            "shape": list(schmid.shape),
            "spacing_um": spacing_um,
        },
        "mask": {
            "rule": "finite and 0 < Schmid <= 0.5 and Euler triplet not all zero",
            "valid_count": int(np.count_nonzero(mask)),
            "invalid_count": int(mask.size - np.count_nonzero(mask)),
            "valid_fraction": result.valid_fraction,
        },
        "estimator": {
            "boundary": "circular FFT; interpreted only to one quarter of the shortest axis",
            "maximum_lag_pixels": maximum_lag,
            "fit_interval": [0.15, 0.60],
            "minimum_fit_points": 5,
            "block_layout": [4, 4],
        },
        "lengths": {
            "radial_decay": _fit_dict(result.radial_decay, spacing_um),
            "x_decay": _fit_dict(result.x_decay, spacing_um),
            "y_decay": _fit_dict(result.y_decay, spacing_um),
            "directional_anisotropy_ratio": anisotropy,
            "rms_radius_pixels": result.rms_radius_pixels,
            "rms_radius_um": result.rms_radius_pixels * spacing_um,
            "rms_control_length_pixels": result.rms_control_length_pixels,
            "rms_control_length_um": result.rms_control_length_pixels * spacing_um,
        },
        "blocks": block_records,
        "bootstrap_block_median": bootstrap,
        "claim_boundary": (
            "Independent EBSD/Schmid structural correlation scale; not a fitted "
            "micromorphic parameter and not a demonstrated material internal length."
        ),
    }
    (output_directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savetxt(
        output_directory / "radial_profile.csv",
        _profile_rows(result.radial, spacing_um),
        delimiter=",",
        header="lag_pixels,lag_um,correlation,pair_weight",
        comments="",
    )
    np.savetxt(
        output_directory / "direction_x_profile.csv",
        _profile_rows(result.x_direction, spacing_um),
        delimiter=",",
        header="lag_pixels,lag_um,correlation,pair_weight",
        comments="",
    )
    np.savetxt(
        output_directory / "direction_y_profile.csv",
        _profile_rows(result.y_direction, spacing_um),
        delimiter=",",
        header="lag_pixels,lag_um,correlation,pair_weight",
        comments="",
    )
    _plot_profiles(
        result,
        spacing_um=spacing_um,
        output_path=output_directory / "correlation_profiles.png",
    )
    return report
