"""Select DIC partitions with sufficient spatial heterogeneity for calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.stats import kurtosis

from fem_inhouse.partitioning import PartitionLayout
from fem_inhouse.workflows.nonlocality_diagnostic import reconstruct_historical_evm

FloatArray = NDArray[np.float64]


def _winsorized_kurtosis(values: FloatArray, low: float, high: float) -> float:
    clipped = np.clip(values, np.percentile(values, low), np.percentile(values, high))
    return float(kurtosis(clipped, fisher=True, bias=False))


def _band_morphology(
    field: FloatArray,
    *,
    spacing_x_mm: float,
    spacing_y_mm: float,
) -> dict[str, float | int]:
    """Characterise the dominant elongated high-strain component.

    The local q85 excursion set is evaluated after a light three-pixel
    Gaussian denoising. Small gaps and isolated pixels are removed before
    connected-component and principal-axis measurements. The score favours a
    long, contrasted component occupying a measurable fraction of the ROI;
    unlike kurtosis, it rejects an intense but nearly circular hotspot.
    """

    smoothed = ndimage.gaussian_filter(field, sigma=3.0, mode="nearest")
    threshold = float(np.quantile(smoothed, 0.85))
    mask = smoothed >= threshold
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    labels, count = ndimage.label(mask)
    minimum_area = max(9, int(np.ceil(0.005 * field.size)))
    candidates: list[dict[str, float | int]] = []
    field_std = max(float(np.std(field)), np.finfo(np.float64).eps)
    for label in range(1, count + 1):
        points = np.argwhere(labels == label)
        area = int(points.shape[0])
        if area < minimum_area:
            continue
        centred = points - np.mean(points, axis=0)
        covariance = np.cov(centred.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], np.finfo(np.float64).eps)
        principal_axes = eigenvectors[:, order]
        aspect_ratio = float(np.sqrt(eigenvalues[0] / eigenvalues[1]))
        projected = centred @ principal_axes
        major_extent_px = float(np.ptp(projected[:, 0]) + 1.0)
        average_width_px = float(area / major_extent_px)
        physical_points = points * np.array([spacing_x_mm, spacing_y_mm])
        physical_centred = physical_points - np.mean(physical_points, axis=0)
        physical_covariance = np.cov(physical_centred.T)
        _, physical_axes = np.linalg.eigh(physical_covariance)
        physical_major_axis = physical_axes[:, -1]
        physical_projection = physical_centred @ physical_major_axis
        major_extent_mm = float(
            np.ptp(physical_projection)
            + np.hypot(
                physical_major_axis[0] * spacing_x_mm,
                physical_major_axis[1] * spacing_y_mm,
            )
        )
        average_width_mm = float(
            area * spacing_x_mm * spacing_y_mm / major_extent_mm
        )
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        touches = sum(
            (
                bool(minimum[0] <= 3),
                bool(minimum[1] <= 3),
                bool(maximum[0] >= field.shape[0] - 4),
                bool(maximum[1] >= field.shape[1] - 4),
            )
        )
        component = labels == label
        background = ~component
        contrast_sigma = float(
            (np.mean(smoothed[component]) - np.mean(smoothed[background])) / field_std
        )
        area_fraction = float(area / field.size)
        edge_factor = 1.0 if touches <= 1 else 0.75 if touches == 2 else 0.5
        band_score = float(
            aspect_ratio
            * np.sqrt(area_fraction)
            * max(contrast_sigma, 0.0)
            * edge_factor
        )
        candidates.append(
            {
                "band_score": band_score,
                "band_threshold": threshold,
                "band_area_fraction": area_fraction,
                "band_aspect_ratio": aspect_ratio,
                "band_major_extent_px": major_extent_px,
                "band_average_width_px": average_width_px,
                "band_major_extent_mm": major_extent_mm,
                "band_average_width_mm": average_width_mm,
                "band_contrast_sigma": contrast_sigma,
                "band_boundary_contacts": touches,
            }
        )
    if candidates:
        return max(candidates, key=lambda item: float(item["band_score"]))
    return {
        "band_score": 0.0,
        "band_threshold": threshold,
        "band_area_fraction": 0.0,
        "band_aspect_ratio": 0.0,
        "band_major_extent_px": 0.0,
        "band_average_width_px": 0.0,
        "band_major_extent_mm": 0.0,
        "band_average_width_mm": 0.0,
        "band_contrast_sigma": 0.0,
        "band_boundary_contacts": 0,
    }


def scan_dic_partition_heterogeneity(
    *,
    input_directory: str | Path,
    parts_x: int = 10,
    parts_y: int = 10,
    padding: int = 150,
    spacing_x_mm: float = 0.00184,
    spacing_y_mm: float = 0.00184,
    poisson_ratio: float = 0.3,
) -> dict[str, Any]:
    """Compute spatial-band and heterogeneity indicators for every DIC partition.

    The primary score measures the morphology of the dominant elongated
    high-strain component. Distribution-only indicators remain in the report
    as diagnostics, but are not used for selection because kurtosis cannot
    distinguish a deformation band from an isolated hotspot. No FEM field is
    used in this diagnostic.
    """

    root = Path(input_directory)
    displacement_x = np.load(root / "displacement_x_mm.npy", mmap_mode="r", allow_pickle=False)
    displacement_y = np.load(root / "displacement_y_mm.npy", mmap_mode="r", allow_pickle=False)
    if displacement_x.shape != displacement_y.shape or displacement_x.ndim != 2:
        raise ValueError("DIC displacement fields must be matching two-dimensional arrays")
    displacement = np.stack((displacement_x, displacement_y), axis=-1)
    evm = reconstruct_historical_evm(
        displacement,
        spacing_x_mm=spacing_x_mm,
        spacing_y_mm=spacing_y_mm,
        poisson_ratio=poisson_ratio,
    )
    global_shape = (int(evm.shape[0]), int(evm.shape[1]))
    layout = PartitionLayout(global_shape, (parts_x, parts_y), padding=padding)
    records: list[dict[str, Any]] = []
    for partition in layout:
        field = np.asarray(evm[partition.core_element_slice_global], dtype=np.float64)
        values = field.ravel()
        q25, q50, q75, q95, q99 = np.percentile(values, [25, 50, 75, 95, 99])
        iqr = max(float(q75 - q25), np.finfo(float).eps)
        gradient_x = np.diff(field, axis=0) / spacing_x_mm
        gradient_y = np.diff(field, axis=1) / spacing_y_mm
        gradient_rms = float(np.sqrt(np.mean(gradient_x**2) + np.mean(gradient_y**2)))
        mean = float(np.mean(values))
        band = _band_morphology(
            field,
            spacing_x_mm=spacing_x_mm,
            spacing_y_mm=spacing_y_mm,
        )
        records.append(
            {
                "partition_id": partition.partition_id,
                "index": [partition.index_x, partition.index_y],
                "core_bounds": list(partition.core_bounds),
                "core_shape": list(partition.core_shape),
                "mean": mean,
                "std": float(np.std(values)),
                "coefficient_of_variation": float(np.std(values) / max(abs(mean), 1e-15)),
                "fisher_kurtosis": float(kurtosis(values, fisher=True, bias=False)),
                "winsorized_kurtosis_1_99": _winsorized_kurtosis(values, 1.0, 99.0),
                "winsorized_kurtosis_5_95": _winsorized_kurtosis(values, 5.0, 95.0),
                "q95_minus_q50_over_iqr": float((q95 - q50) / iqr),
                "q99_minus_q50_over_iqr": float((q99 - q50) / iqr),
                "gradient_rms": gradient_rms,
                "gradient_rms_relative": float(gradient_rms / max(mean, 1e-15)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                **band,
            }
        )
    ranked = sorted(records, key=lambda item: item["band_score"], reverse=True)
    return {
        "observable": "EVM_HISTORICAL reconstructed from DIC displacement",
        "input_directory": str(root),
        "global_shape": list(global_shape),
        "partition_shape": [parts_x, parts_y],
        "padding": padding,
        "spacing_x_mm": spacing_x_mm,
        "spacing_y_mm": spacing_y_mm,
        "poisson_ratio": poisson_ratio,
        "selection_indicator": "dic_band_morphology_score_q85",
        "band_morphology": {
            "gaussian_sigma_pixels": 3.0,
            "threshold_quantile": 0.85,
            "binary_closing_shape": [5, 5],
            "binary_opening_shape": [3, 3],
            "minimum_component_area_fraction": 0.005,
            "score": (
                "aspect_ratio * sqrt(area_fraction) * contrast_sigma * edge_factor"
            ),
        },
        "interpretation": (
            "Use the ranking to pre-register a ROI containing a coherent deformation "
            "band. Visual confirmation remains mandatory before any FEM campaign; "
            "the score is a DIC morphology diagnostic, not a material length."
        ),
        "partitions": ranked,
    }


def write_dic_partition_heterogeneity_report(report: dict[str, Any], output: str | Path) -> None:
    """Write a deterministic JSON report produced by the partition scan."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
