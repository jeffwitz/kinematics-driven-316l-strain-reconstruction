"""Select DIC partitions with sufficient spatial heterogeneity for calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis

from fem_inhouse.partitioning import PartitionLayout
from fem_inhouse.workflows.nonlocality_diagnostic import reconstruct_historical_evm

FloatArray = NDArray[np.float64]


def _winsorized_kurtosis(values: FloatArray, low: float, high: float) -> float:
    clipped = np.clip(values, np.percentile(values, low), np.percentile(values, high))
    return float(kurtosis(clipped, fisher=True, bias=False))


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
    """Compute robust heterogeneity indicators for every DIC partition.

    The reported score is deliberately based on winsorized statistics. This
    prevents one corrupted pixel from selecting a region while retaining the
    sensitivity to localized high-strain tails required for alpha selection.
    No FEM field is used in this diagnostic.
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
            }
        )
    ranked = sorted(records, key=lambda item: item["winsorized_kurtosis_1_99"], reverse=True)
    return {
        "observable": "EVM_HISTORICAL reconstructed from DIC displacement",
        "input_directory": str(root),
        "global_shape": list(global_shape),
        "partition_shape": [parts_x, parts_y],
        "padding": padding,
        "spacing_x_mm": spacing_x_mm,
        "spacing_y_mm": spacing_y_mm,
        "poisson_ratio": poisson_ratio,
        "selection_indicator": "winsorized_kurtosis_1_99",
        "interpretation": (
            "Use the ranking to pre-register a heterogeneous ROI. Confirm that "
            "the selected high-tail structure is not a one-pixel extraction artifact."
        ),
        "partitions": ranked,
    }


def write_dic_partition_heterogeneity_report(report: dict[str, Any], output: str | Path) -> None:
    """Write a deterministic JSON report produced by the partition scan."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
