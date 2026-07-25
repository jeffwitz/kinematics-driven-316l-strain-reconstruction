"""Reproducible diagnostic of spatial-width mismatch by Helmholtz filtering."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import scipy
import yaml
from numpy.typing import ArrayLike, NDArray

from fem_inhouse import __version__
from fem_inhouse.data_preparation import fingerprint_file
from fem_inhouse.partitioning import Partition, PartitionLayout, extract_partition_field
from fem_inhouse.postprocessing.helmholtz import helmholtz_filter_element_field
from fem_inhouse.postprocessing.kinematics import (
    cell_average,
    plane_stress_equivalent_strain,
    strain_from_displacement,
)
from fem_inhouse.postprocessing.metrics import (
    absolute_threshold_overlap_metrics,
    field_diffusivity_metrics,
    field_error_metrics,
    localization_overlap_metrics,
)

FloatArray = NDArray[np.float64]
DiagnosticMode = Literal["exploratory", "confirmatory"]
SaveFieldsMode = Literal["all", "best", "none"]
LengthUnit = Literal["mm", "um", "pixels"]
MetricRecord = dict[str, Any]


@dataclass(frozen=True, slots=True)
class LengthScale:
    """One physical length represented in all campaign units."""

    length_mm: float
    length_um: float
    length_pixels: float


@dataclass(frozen=True, slots=True)
class DecisionThresholds:
    """Pre-declared confirmatory gains for one held-out campaign."""

    minimum_correlation_gain: float
    minimum_relative_l2_reduction: float
    minimum_iou_gain: float
    maximum_relative_mean_drift: float

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("decision thresholds must be finite")
        for name in (
            "minimum_correlation_gain",
            "minimum_relative_l2_reduction",
            "minimum_iou_gain",
            "maximum_relative_mean_drift",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True, slots=True)
class NonlocalitySweep:
    """Filtered fields, metric rows and selections for one diagnostic sweep."""

    lengths: tuple[LengthScale, ...]
    filtered_evm: dict[float, FloatArray]
    filtered_peeq: dict[float, FloatArray]
    metrics: tuple[MetricRecord, ...]
    filter_checks: tuple[dict[str, Any], ...]
    selection: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing JSON file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_save_array(path: Path, values: ArrayLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(values), allow_pickle=False)
    temporary.replace(path)


def _git_state(repository: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def normalize_length_scales(
    values: Sequence[float],
    *,
    unit: LengthUnit,
    pixel_size_mm: float,
) -> tuple[LengthScale, ...]:
    """Validate, convert, deduplicate and sort a user-provided length sweep."""

    if not np.isfinite(pixel_size_mm) or pixel_size_mm <= 0:
        raise ValueError("pixel_size_mm must be finite and strictly positive")
    if unit not in {"mm", "um", "pixels"}:
        raise ValueError("unit must be one of 'mm', 'um', or 'pixels'")
    converted = [0.0]
    for value in values:
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0:
            raise ValueError("length scales must be finite and nonnegative")
        if unit == "mm":
            converted.append(numeric)
        elif unit == "um":
            converted.append(numeric / 1000.0)
        else:
            converted.append(numeric * pixel_size_mm)
    return tuple(
        LengthScale(
            length_mm=length,
            length_um=length * 1000.0,
            length_pixels=length / pixel_size_mm,
        )
        for length in sorted(set(converted))
    )


def load_decision_thresholds(path: str | Path) -> DecisionThresholds:
    """Load confirmatory thresholds from a JSON or YAML mapping."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing decision-threshold file: {source}")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("decision-threshold file must contain a mapping")
    nested = data.get("decision_thresholds", data)
    if not isinstance(nested, dict):
        raise ValueError("decision_thresholds must be a mapping")
    required = {
        "minimum_correlation_gain",
        "minimum_relative_l2_reduction",
        "minimum_iou_gain",
        "maximum_relative_mean_drift",
    }
    missing = required - set(nested)
    unknown = set(nested) - required
    if missing:
        raise ValueError(f"missing decision thresholds: {sorted(missing)}")
    if unknown:
        raise ValueError(f"unknown decision thresholds: {sorted(unknown)}")
    return DecisionThresholds(**{name: float(nested[name]) for name in required})


def reconstruct_historical_evm(
    displacement_mm: ArrayLike,
    *,
    spacing_x_mm: float,
    spacing_y_mm: float,
    poisson_ratio: float,
) -> FloatArray:
    """Apply the common displacement-to-element EVM operator."""

    displacement = np.asarray(displacement_mm, dtype=np.float64)
    if displacement.ndim != 3 or displacement.shape[-1] != 2:
        raise ValueError("displacement_mm must have shape (nx + 1, ny + 1, 2)")
    if not np.isfinite(displacement).all():
        raise ValueError("displacement_mm must contain only finite values")
    strain = strain_from_displacement(
        displacement[..., 0],
        displacement[..., 1],
        spacing_x=spacing_x_mm,
        spacing_y=spacing_y_mm,
    )
    nodal_evm = plane_stress_equivalent_strain(
        strain.epsilon_xx,
        strain.epsilon_yy,
        strain.gamma_xy,
        poisson_ratio=poisson_ratio,
        shear_convention="engineering",
    )
    return np.asarray(cell_average(nodal_evm), dtype=np.float64)


def _suffix_percent(value: float, *, prefix: str) -> str:
    return f"{prefix}_{f'{100.0 * value:g}'.replace('.', 'p')}pct"


def _padding_geometry(
    partition: Partition,
    *,
    global_shape: tuple[int, int],
) -> tuple[dict[str, int], int | None]:
    core_x0, core_x1, core_y0, core_y1 = partition.core_bounds
    solve_x0, solve_x1, solve_y0, solve_y1 = partition.solve_bounds
    nx, ny = global_shape
    sides = {
        "x_minus": core_x0 - solve_x0,
        "x_plus": solve_x1 - core_x1,
        "y_minus": core_y0 - solve_y0,
        "y_plus": solve_y1 - core_y1,
    }
    artificial = [
        sides["x_minus"] if core_x0 > 0 else None,
        sides["x_plus"] if core_x1 < nx else None,
        sides["y_minus"] if core_y0 > 0 else None,
        sides["y_plus"] if core_y1 < ny else None,
    ]
    finite = [value for value in artificial if value is not None]
    return sides, min(finite) if finite else None


def _boundary_classification(
    length: LengthScale,
    *,
    minimum_artificial_padding_pixels: int | None,
    pixel_size_mm: float,
    minimum_padding_length_ratio: float,
) -> tuple[float | None, str]:
    if length.length_mm == 0.0:
        return None, "baseline"
    if minimum_artificial_padding_pixels is None:
        return None, "physical_boundary_only"
    ratio = minimum_artificial_padding_pixels * pixel_size_mm / length.length_mm
    status = "usable" if ratio >= minimum_padding_length_ratio else "boundary_contaminated"
    return ratio, status


def _metric_record(
    *,
    field_name: str,
    length: LengthScale,
    filtered: FloatArray,
    raw: FloatArray,
    dic_reference: FloatArray,
    core_slice: tuple[slice, slice],
    padding_ratio: float | None,
    boundary_status: str,
    spacing_x_mm: float,
    spacing_y_mm: float,
    top_fractions: Sequence[float],
    dic_quantiles: Sequence[float],
    filter_residual_relative: float,
    filter_mean_drift_full: float,
) -> MetricRecord:
    raw_core = np.asarray(raw[core_slice], dtype=np.float64)
    filtered_core = np.asarray(filtered[core_slice], dtype=np.float64)
    dic_core = np.asarray(dic_reference[core_slice], dtype=np.float64)
    diffusivity = field_diffusivity_metrics(
        filtered_core,
        raw_field=raw_core,
        spacing_x_mm=spacing_x_mm,
        spacing_y_mm=spacing_y_mm,
    )
    record: MetricRecord = {
        "field": field_name,
        "length_mm": length.length_mm,
        "length_um": length.length_um,
        "length_pixels": length.length_pixels,
        "padding_to_length_ratio": padding_ratio,
        "boundary_status": boundary_status,
        "rmse": None,
        "mae": None,
        "relative_l2": None,
        "pearson": None,
        "mean_error": None,
        "max_abs_error": None,
        "gradient_rms": diffusivity.gradient_rms,
        "total_variation": diffusivity.total_variation,
        "mean": diffusivity.mean,
        "mean_drift": diffusivity.mean_drift,
        "std": diffusivity.standard_deviation,
        "std_ratio": diffusivity.standard_deviation_ratio,
        "minimum": diffusivity.minimum,
        "maximum": diffusivity.maximum,
        "peak_ratio": diffusivity.peak_ratio,
        "relative_change_norm": diffusivity.relative_change_norm,
        "filter_mean_drift_full": filter_mean_drift_full,
        "filter_residual_relative": filter_residual_relative,
    }
    if field_name == "EVM_HISTORICAL":
        errors = field_error_metrics(dic_core, filtered_core)
        record.update(
            {
                "rmse": errors.rmse,
                "mae": errors.mae,
                "relative_l2": errors.relative_l2_error,
                "pearson": errors.pearson_correlation,
                "mean_error": errors.signed_mean_error,
                "max_abs_error": errors.maximum_absolute_error,
            }
        )
    for fraction in top_fractions:
        overlap = localization_overlap_metrics(
            dic_core,
            filtered_core,
            top_fraction=fraction,
        )
        suffix = _suffix_percent(fraction, prefix="top")
        record.update(
            {
                f"iou_{suffix}": overlap.intersection_over_union,
                f"dice_{suffix}": overlap.dice_coefficient,
                f"precision_{suffix}": overlap.prediction_precision,
                f"recall_{suffix}": overlap.reference_recall,
            }
        )
    if field_name == "EVM_HISTORICAL":
        for quantile in dic_quantiles:
            absolute_overlap = absolute_threshold_overlap_metrics(
                dic_core,
                filtered_core,
                reference_quantile=quantile,
            )
            suffix = _suffix_percent(quantile, prefix="dic_q")
            record.update(
                {
                    f"threshold_{suffix}": absolute_overlap.absolute_threshold,
                    f"dic_active_fraction_{suffix}": absolute_overlap.reference_active_fraction,
                    f"predicted_active_fraction_{suffix}": (
                        absolute_overlap.prediction_active_fraction
                    ),
                    f"iou_{suffix}": absolute_overlap.intersection_over_union,
                    f"precision_{suffix}": absolute_overlap.prediction_precision,
                    f"recall_{suffix}": absolute_overlap.reference_recall,
                }
            )
    return record


def _pareto_front(records: Sequence[MetricRecord], *, iou_key: str) -> list[float]:
    nondominated: list[float] = []
    for candidate in records:
        dominated = False
        for other in records:
            if other is candidate:
                continue
            no_worse = (
                float(other["rmse"]) <= float(candidate["rmse"])
                and float(other["relative_l2"]) <= float(candidate["relative_l2"])
                and float(other["pearson"]) >= float(candidate["pearson"])
                and float(other[iou_key]) >= float(candidate[iou_key])
            )
            strictly_better = (
                float(other["rmse"]) < float(candidate["rmse"])
                or float(other["relative_l2"]) < float(candidate["relative_l2"])
                or float(other["pearson"]) > float(candidate["pearson"])
                or float(other[iou_key]) > float(candidate[iou_key])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            nondominated.append(float(candidate["length_mm"]))
    return sorted(nondominated)


def _select_candidates(
    records: Sequence[MetricRecord],
    *,
    selection_top_fraction: float,
    mode: DiagnosticMode,
    decision_thresholds: DecisionThresholds | None,
) -> dict[str, Any]:
    evm_records = [
        record
        for record in records
        if record["field"] == "EVM_HISTORICAL"
        and record["boundary_status"] in {"baseline", "usable", "physical_boundary_only"}
    ]
    if not evm_records:
        raise RuntimeError("no numerically usable EVM length is available")
    iou_key = f"iou_{_suffix_percent(selection_top_fraction, prefix='top')}"
    best_rmse = min(evm_records, key=lambda item: (float(item["rmse"]), item["length_mm"]))
    best_correlation = max(
        evm_records,
        key=lambda item: (float(item["pearson"]), -float(item["length_mm"])),
    )
    best_iou = max(
        evm_records,
        key=lambda item: (float(item[iou_key]), -float(item["length_mm"])),
    )
    selection: dict[str, Any] = {
        "selection_iou_top_fraction": selection_top_fraction,
        "best_by_rmse_mm": best_rmse["length_mm"],
        "best_by_correlation_mm": best_correlation["length_mm"],
        "best_by_iou_mm": best_iou["length_mm"],
        "pareto_length_scales_mm": _pareto_front(evm_records, iou_key=iou_key),
        "automatic_physical_conclusion": None,
    }
    if mode == "exploratory":
        selection["interpretation"] = (
            "Exploratory rankings only; no material length or nonlocal hypothesis is "
            "automatically validated."
        )
        return selection
    if decision_thresholds is None:
        raise ValueError("confirmatory mode requires pre-declared decision thresholds")

    baseline = next(record for record in evm_records if record["length_mm"] == 0.0)
    baseline_relative_l2 = float(baseline["relative_l2"])
    candidates: list[dict[str, Any]] = []
    for record in evm_records:
        if record["length_mm"] == 0.0:
            continue
        reduction = (
            (baseline_relative_l2 - float(record["relative_l2"])) / baseline_relative_l2
            if baseline_relative_l2
            else 0.0
        )
        raw_mean = float(record["mean"]) - float(record["mean_drift"])
        relative_mean_drift = abs(float(record["filter_mean_drift_full"])) / max(
            abs(raw_mean),
            np.finfo(float).eps,
        )
        gains = {
            "length_mm": record["length_mm"],
            "correlation_gain": float(record["pearson"]) - float(baseline["pearson"]),
            "relative_l2_reduction": reduction,
            "iou_gain": float(record[iou_key]) - float(baseline[iou_key]),
            "relative_mean_drift": relative_mean_drift,
        }
        gains["passed"] = (
            gains["correlation_gain"] >= decision_thresholds.minimum_correlation_gain
            and gains["relative_l2_reduction"]
            >= decision_thresholds.minimum_relative_l2_reduction
            and gains["iou_gain"] >= decision_thresholds.minimum_iou_gain
            and gains["relative_mean_drift"]
            <= decision_thresholds.maximum_relative_mean_drift
        )
        candidates.append(gains)
    selection["decision_thresholds"] = asdict(decision_thresholds)
    selection["confirmatory_candidates"] = candidates
    selection["criteria_met_on_this_partition"] = any(item["passed"] for item in candidates)
    selection["interpretation"] = (
        "Threshold result for this partition only. A material interpretation requires "
        "pre-selection on a different partition and held-out confirmation."
    )
    return selection


def run_field_sweep(
    *,
    dic_evm_reference: ArrayLike,
    fem_evm_raw: ArrayLike,
    lengths: Sequence[LengthScale],
    spacing_x_mm: float,
    spacing_y_mm: float,
    core_slice: tuple[slice, slice],
    minimum_artificial_padding_pixels: int | None,
    minimum_padding_length_ratio: float = 4.0,
    top_fractions: Sequence[float] = (0.05, 0.1, 0.2),
    dic_quantiles: Sequence[float] = (0.8, 0.9, 0.95),
    peeq_raw: ArrayLike | None = None,
    mode: DiagnosticMode = "exploratory",
    decision_thresholds: DecisionThresholds | None = None,
) -> NonlocalitySweep:
    """Filter complete padded fields, then evaluate metrics only on the core."""

    if not lengths or not any(length.length_mm == 0.0 for length in lengths):
        raise ValueError("lengths must contain the zero-length baseline")
    if not np.isfinite(minimum_padding_length_ratio) or minimum_padding_length_ratio <= 0:
        raise ValueError("minimum_padding_length_ratio must be finite and positive")
    for fraction in top_fractions:
        if not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("top_fractions must lie in (0, 1]")
    for quantile in dic_quantiles:
        if not np.isfinite(quantile) or not 0 <= quantile <= 1:
            raise ValueError("dic_quantiles must lie in [0, 1]")
    if mode not in {"exploratory", "confirmatory"}:
        raise ValueError("mode must be 'exploratory' or 'confirmatory'")
    if mode == "confirmatory" and decision_thresholds is None:
        raise ValueError("confirmatory mode requires pre-declared decision thresholds")

    dic = np.asarray(dic_evm_reference, dtype=np.float64)
    fem = np.asarray(fem_evm_raw, dtype=np.float64)
    if dic.ndim != 2 or fem.ndim != 2 or dic.shape != fem.shape:
        raise ValueError("DIC and FEM EVM fields must be matching two-dimensional arrays")
    if not np.isfinite(dic).all() or not np.isfinite(fem).all():
        raise ValueError("DIC and FEM EVM fields must contain only finite values")
    peeq = None if peeq_raw is None else np.asarray(peeq_raw, dtype=np.float64)
    if peeq is not None and (peeq.shape != fem.shape or not np.isfinite(peeq).all()):
        raise ValueError("PEEQ must be finite and have the same shape as FEM EVM")

    filtered_evm: dict[float, FloatArray] = {}
    filtered_peeq: dict[float, FloatArray] = {}
    records: list[MetricRecord] = []
    checks: list[dict[str, Any]] = []
    for length in lengths:
        padding_ratio, boundary_status = _boundary_classification(
            length,
            minimum_artificial_padding_pixels=minimum_artificial_padding_pixels,
            pixel_size_mm=spacing_x_mm,
            minimum_padding_length_ratio=minimum_padding_length_ratio,
        )
        evm_result = helmholtz_filter_element_field(
            fem,
            length_scale_mm=length.length_mm,
            spacing_x_mm=spacing_x_mm,
            spacing_y_mm=spacing_y_mm,
        )
        filtered_evm[length.length_mm] = evm_result.filtered_element_field
        records.append(
            _metric_record(
                field_name="EVM_HISTORICAL",
                length=length,
                filtered=evm_result.filtered_element_field,
                raw=fem,
                dic_reference=dic,
                core_slice=core_slice,
                padding_ratio=padding_ratio,
                boundary_status=boundary_status,
                spacing_x_mm=spacing_x_mm,
                spacing_y_mm=spacing_y_mm,
                top_fractions=top_fractions,
                dic_quantiles=dic_quantiles,
                filter_residual_relative=evm_result.residual_relative,
                filter_mean_drift_full=evm_result.mean_drift,
            )
        )
        checks.append(
            {
                "field": "EVM_HISTORICAL",
                "length_mm": length.length_mm,
                "mean_drift_full": evm_result.mean_drift,
                "residual_relative": evm_result.residual_relative,
            }
        )
        if peeq is not None:
            peeq_result = helmholtz_filter_element_field(
                peeq,
                length_scale_mm=length.length_mm,
                spacing_x_mm=spacing_x_mm,
                spacing_y_mm=spacing_y_mm,
            )
            filtered_peeq[length.length_mm] = peeq_result.filtered_element_field
            records.append(
                _metric_record(
                    field_name="PEEQ",
                    length=length,
                    filtered=peeq_result.filtered_element_field,
                    raw=peeq,
                    dic_reference=dic,
                    core_slice=core_slice,
                    padding_ratio=padding_ratio,
                    boundary_status=boundary_status,
                    spacing_x_mm=spacing_x_mm,
                    spacing_y_mm=spacing_y_mm,
                    top_fractions=top_fractions,
                    dic_quantiles=(),
                    filter_residual_relative=peeq_result.residual_relative,
                    filter_mean_drift_full=peeq_result.mean_drift,
                )
            )
            checks.append(
                {
                    "field": "PEEQ",
                    "length_mm": length.length_mm,
                    "mean_drift_full": peeq_result.mean_drift,
                    "residual_relative": peeq_result.residual_relative,
                }
            )

    selection_fraction = min(top_fractions, key=lambda value: abs(value - 0.1))
    selection = _select_candidates(
        records,
        selection_top_fraction=selection_fraction,
        mode=mode,
        decision_thresholds=decision_thresholds,
    )
    return NonlocalitySweep(
        lengths=tuple(lengths),
        filtered_evm=filtered_evm,
        filtered_peeq=filtered_peeq,
        metrics=tuple(records),
        filter_checks=tuple(checks),
        selection=selection,
    )


def _field_filename(field: str, length: LengthScale) -> str:
    return f"{field}_ell_{round(length.length_um * 100.0):06d}um.npy"


def _write_metrics_csv(path: Path, records: Sequence[MetricRecord]) -> None:
    required = [
        "field",
        "length_mm",
        "length_um",
        "length_pixels",
        "padding_to_length_ratio",
        "boundary_status",
        "rmse",
        "mae",
        "relative_l2",
        "pearson",
        "mean_error",
        "max_abs_error",
        "gradient_rms",
        "total_variation",
        "mean_drift",
        "std_ratio",
        "peak_ratio",
    ]
    extras = sorted(set().union(*(record.keys() for record in records)) - set(required))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*required, *extras])
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def _plot_field(
    axis: Any,
    values: FloatArray,
    *,
    title: str,
    extent_mm: tuple[float, float, float, float],
    color_limits: tuple[float, float],
    core_rectangle_mm: tuple[float, float, float, float],
    cmap: str,
) -> Any:
    from matplotlib.patches import Rectangle

    image = axis.imshow(
        values.T,
        origin="lower",
        extent=extent_mm,
        aspect="equal",
        cmap=cmap,
        vmin=color_limits[0],
        vmax=color_limits[1],
        interpolation="nearest",
    )
    x0, y0, width, height = core_rectangle_mm
    axis.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            fill=False,
            edgecolor="white",
            linewidth=1.0,
            linestyle="--",
        )
    )
    axis.set_title(title)
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    return image


def _write_figures(
    directory: Path,
    *,
    dic_evm: FloatArray,
    fem_evm: FloatArray,
    peeq: FloatArray | None,
    sweep: NonlocalitySweep,
    spacing_x_mm: float,
    spacing_y_mm: float,
    core_slice: tuple[slice, slice],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    nx, ny = dic_evm.shape
    extent = (0.0, nx * spacing_x_mm, 0.0, ny * spacing_y_mm)
    core_x, core_y = core_slice
    core_rectangle = (
        int(core_x.start or 0) * spacing_x_mm,
        int(core_y.start or 0) * spacing_y_mm,
        (int(core_x.stop or nx) - int(core_x.start or 0)) * spacing_x_mm,
        (int(core_y.stop or ny) - int(core_y.start or 0)) * spacing_y_mm,
    )
    evm_values = [dic_evm, fem_evm, *sweep.filtered_evm.values()]
    evm_limits = (
        min(float(np.min(values)) for values in evm_values),
        max(float(np.max(values)) for values in evm_values),
    )
    differences = [values - dic_evm for values in sweep.filtered_evm.values()]
    difference_limit = max(float(np.max(np.abs(values))) for values in differences)

    for length in sweep.lengths:
        filtered = sweep.filtered_evm[length.length_mm]
        figure, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
        images = [
            _plot_field(
                axes[0],
                dic_evm,
                title="DIC reference",
                extent_mm=extent,
                color_limits=evm_limits,
                core_rectangle_mm=core_rectangle,
                cmap="viridis",
            ),
            _plot_field(
                axes[1],
                fem_evm,
                title="Raw FEM",
                extent_mm=extent,
                color_limits=evm_limits,
                core_rectangle_mm=core_rectangle,
                cmap="viridis",
            ),
            _plot_field(
                axes[2],
                filtered,
                title=f"Filtered FEM, ell={length.length_um:g} um",
                extent_mm=extent,
                color_limits=evm_limits,
                core_rectangle_mm=core_rectangle,
                cmap="viridis",
            ),
            _plot_field(
                axes[3],
                filtered - dic_evm,
                title="FEM filtered - DIC",
                extent_mm=extent,
                color_limits=(-difference_limit, difference_limit),
                core_rectangle_mm=core_rectangle,
                cmap="coolwarm",
            ),
        ]
        figure.colorbar(images[0], ax=axes[:3], shrink=0.75, label="equivalent strain")
        figure.colorbar(images[3], ax=axes[3], shrink=0.75, label="strain difference")
        code = round(length.length_um * 100.0)
        figure.savefig(directory / f"evm_comparison_ell_{code:06d}um.png", dpi=180)
        if length.length_mm == 0.0:
            figure.savefig(directory / "evm_baseline.png", dpi=180)
        plt.close(figure)

    if peeq is not None:
        peeq_values = [peeq, *sweep.filtered_peeq.values()]
        peeq_limits = (
            min(float(np.min(values)) for values in peeq_values),
            max(float(np.max(values)) for values in peeq_values),
        )
        difference_limit = max(
            float(np.max(np.abs(values - peeq))) for values in sweep.filtered_peeq.values()
        )
        for length in sweep.lengths:
            filtered = sweep.filtered_peeq[length.length_mm]
            figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
            raw_image = _plot_field(
                axes[0],
                peeq,
                title="Raw PEEQ",
                extent_mm=extent,
                color_limits=peeq_limits,
                core_rectangle_mm=core_rectangle,
                cmap="magma",
            )
            _plot_field(
                axes[1],
                filtered,
                title=f"Filtered PEEQ, ell={length.length_um:g} um",
                extent_mm=extent,
                color_limits=peeq_limits,
                core_rectangle_mm=core_rectangle,
                cmap="magma",
            )
            difference_image = _plot_field(
                axes[2],
                filtered - peeq,
                title="Filtered - raw PEEQ",
                extent_mm=extent,
                color_limits=(-difference_limit, difference_limit),
                core_rectangle_mm=core_rectangle,
                cmap="coolwarm",
            )
            figure.colorbar(raw_image, ax=axes[:2], shrink=0.75, label="PEEQ")
            figure.colorbar(difference_image, ax=axes[2], shrink=0.75, label="PEEQ difference")
            code = round(length.length_um * 100.0)
            figure.savefig(directory / f"peeq_comparison_ell_{code:06d}um.png", dpi=180)
            plt.close(figure)

    evm_records = [record for record in sweep.metrics if record["field"] == "EVM_HISTORICAL"]
    lengths_um = [float(record["length_um"]) for record in evm_records]
    iou_key = f"iou_{_suffix_percent(sweep.selection['selection_iou_top_fraction'], prefix='top')}"
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    curves = (
        ("rmse", "RMSE"),
        ("relative_l2", "relative L2"),
        ("pearson", "Pearson correlation"),
        (iou_key, "localization IoU"),
    )
    for axis, (key, label) in zip(axes.ravel(), curves, strict=True):
        axis.plot(lengths_um, [record[key] for record in evm_records], marker="o")
        axis.set_xlabel("ell (um)")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    figure.savefig(directory / "metric_curves.svg")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    curves = (
        ("gradient_rms", "gradient RMS"),
        ("total_variation", "discrete total variation"),
        ("peak_ratio", "peak / raw peak"),
        ("std_ratio", "standard deviation / raw"),
    )
    for axis, (key, label) in zip(axes.ravel(), curves, strict=True):
        axis.plot(lengths_um, [record[key] for record in evm_records], marker="o")
        axis.set_xlabel("ell (um)")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    figure.savefig(directory / "diffusivity_curves.svg")
    plt.close(figure)

    candidate_lengths = list(
        dict.fromkeys(
            [
                float(sweep.selection["best_by_rmse_mm"]),
                float(sweep.selection["best_by_correlation_mm"]),
                float(sweep.selection["best_by_iou_mm"]),
            ]
        )
    )
    figure, axes = plt.subplots(
        1,
        len(candidate_lengths) + 1,
        figsize=(4 * (len(candidate_lengths) + 1), 4),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    image = _plot_field(
        axes_array[0],
        dic_evm,
        title="DIC reference",
        extent_mm=extent,
        color_limits=evm_limits,
        core_rectangle_mm=core_rectangle,
        cmap="viridis",
    )
    for axis, length_mm in zip(axes_array[1:], candidate_lengths, strict=True):
        _plot_field(
            axis,
            sweep.filtered_evm[length_mm],
            title=f"candidate ell={length_mm * 1000.0:g} um",
            extent_mm=extent,
            color_limits=evm_limits,
            core_rectangle_mm=core_rectangle,
            cmap="viridis",
        )
    figure.colorbar(image, ax=axes_array.tolist(), shrink=0.75, label="equivalent strain")
    figure.savefig(directory / "best_candidates.png", dpi=180)
    plt.close(figure)


def _selected_lengths(sweep: NonlocalitySweep, save_fields: SaveFieldsMode) -> set[float]:
    if save_fields == "all":
        return {length.length_mm for length in sweep.lengths}
    if save_fields == "none":
        return set()
    return {
        0.0,
        float(sweep.selection["best_by_rmse_mm"]),
        float(sweep.selection["best_by_correlation_mm"]),
        float(sweep.selection["best_by_iou_mm"]),
    }


def _prepare_destination(destination: Path, *, overwrite: bool) -> Path:
    resolved = destination.resolve()
    if resolved in {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}:
        raise ValueError(f"unsafe diagnostic output directory: {resolved}")
    if destination.is_symlink():
        raise ValueError("diagnostic output directory must not be a symbolic link")
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"diagnostic output directory is not empty; pass overwrite explicitly: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    temporary.mkdir()
    return temporary


def _promote_destination(temporary: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)


def _load_campaign_fields(
    *,
    input_path: Path,
    campaign_path: Path,
    layout: PartitionLayout,
    partition: Partition,
    status: dict[str, Any],
) -> tuple[FloatArray, FloatArray, Path, Path]:
    partition_path = campaign_path / "partitions" / f"{partition.partition_id:04d}"
    u_path = partition_path / "U.npy"
    peeq_path = partition_path / "PEEQ.npy"
    for name, path in (("U", u_path), ("PEEQ", peeq_path)):
        if not path.is_file():
            raise FileNotFoundError(f"missing saved partition field {name}: {path}")
        expected_hash = status.get("outputs", {}).get(name)
        if expected_hash is not None and fingerprint_file(path) != expected_hash:
            raise RuntimeError(f"saved partition field fails its status hash: {path}")
    fem_displacement = np.asarray(np.load(u_path, mmap_mode="r", allow_pickle=False))
    expected = (partition.solve_shape[0] + 1, partition.solve_shape[1] + 1, 2)
    if fem_displacement.shape != expected:
        raise ValueError(f"U has shape {fem_displacement.shape}, expected {expected}")
    global_x = np.load(
        input_path / "displacement_x_mm.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    global_y = np.load(
        input_path / "displacement_y_mm.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    dic_x = extract_partition_field(
        global_x,
        layout=layout,
        partition=partition,
        location="node",
    )
    dic_y = extract_partition_field(
        global_y,
        layout=layout,
        partition=partition,
        location="node",
    )
    return fem_displacement, np.stack((dic_x, dic_y), axis=-1), u_path, peeq_path


def run_nonlocality_diagnostic(
    *,
    input_directory: str | Path,
    campaign_directory: str | Path,
    partition_id: int,
    output_directory: str | Path,
    length_values: Sequence[float],
    length_unit: LengthUnit,
    include_peeq: bool = False,
    mode: DiagnosticMode = "exploratory",
    decision_thresholds: DecisionThresholds | None = None,
    top_fractions: Sequence[float] = (0.05, 0.1, 0.2),
    dic_quantiles: Sequence[float] = (0.8, 0.9, 0.95),
    minimum_padding_length_ratio: float = 4.0,
    save_fields: SaveFieldsMode = "all",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one immutable diagnostic from a saved padded partition."""

    if save_fields not in {"all", "best", "none"}:
        raise ValueError("save_fields must be 'all', 'best', or 'none'")
    if mode == "confirmatory" and decision_thresholds is None:
        raise ValueError("confirmatory mode requires pre-declared decision thresholds")
    input_path = Path(input_directory)
    campaign_path = Path(campaign_directory)
    destination = Path(output_directory)
    manifest_path = campaign_path / "manifest.json"
    campaign_manifest = _load_json(manifest_path)
    layout_data = campaign_manifest.get("layout")
    config_data = campaign_manifest.get("config")
    if not isinstance(layout_data, dict) or not isinstance(config_data, dict):
        raise ValueError("campaign manifest lacks layout or config metadata")
    global_shape = tuple(int(value) for value in layout_data["global_shape"])
    partition_shape = tuple(int(value) for value in layout_data["partition_shape"])
    if len(global_shape) != 2 or len(partition_shape) != 2:
        raise ValueError("campaign layout shapes must have two entries")
    layout = PartitionLayout(
        global_shape=(global_shape[0], global_shape[1]),
        partition_shape=(partition_shape[0], partition_shape[1]),
        padding=int(layout_data["padding"]),
    )
    partition = layout.get(partition_id)
    serialized = [
        item
        for item in layout_data.get("partitions", [])
        if int(item.get("partition_id", -1)) == partition_id
    ]
    if len(serialized) != 1:
        raise ValueError("campaign manifest does not identify exactly one requested partition")
    if (
        tuple(serialized[0]["core_bounds"]) != partition.core_bounds
        or tuple(serialized[0]["solve_bounds"]) != partition.solve_bounds
    ):
        raise ValueError("partition bounds disagree with the declared layout")

    mesh = config_data["mesh"]
    material = config_data["material"]
    spacing_x_mm = float(mesh["base_pixel_size_mm"]) * float(mesh["scale_factor"])
    spacing_y_mm = spacing_x_mm
    poisson_ratio = float(material["poisson_ratio"])
    lengths = normalize_length_scales(
        length_values,
        unit=length_unit,
        pixel_size_mm=spacing_x_mm,
    )
    sides_pixels, minimum_padding_pixels = _padding_geometry(
        partition,
        global_shape=layout.global_shape,
    )
    status_path = campaign_path / "partitions" / f"{partition_id:04d}" / "status.json"
    status = _load_json(status_path)
    if not status.get("complete"):
        raise RuntimeError(f"partition is not marked complete: {status_path}")
    fem_displacement, dic_displacement, u_path, peeq_path = _load_campaign_fields(
        input_path=input_path,
        campaign_path=campaign_path,
        layout=layout,
        partition=partition,
        status=status,
    )
    dic_evm = reconstruct_historical_evm(
        dic_displacement,
        spacing_x_mm=spacing_x_mm,
        spacing_y_mm=spacing_y_mm,
        poisson_ratio=poisson_ratio,
    )
    fem_evm = reconstruct_historical_evm(
        fem_displacement,
        spacing_x_mm=spacing_x_mm,
        spacing_y_mm=spacing_y_mm,
        poisson_ratio=poisson_ratio,
    )
    peeq = (
        np.asarray(np.load(peeq_path, mmap_mode="r", allow_pickle=False), dtype=np.float64)
        if include_peeq
        else None
    )
    sweep = run_field_sweep(
        dic_evm_reference=dic_evm,
        fem_evm_raw=fem_evm,
        peeq_raw=peeq,
        lengths=lengths,
        spacing_x_mm=spacing_x_mm,
        spacing_y_mm=spacing_y_mm,
        core_slice=partition.core_element_slice_local,
        minimum_artificial_padding_pixels=minimum_padding_pixels,
        minimum_padding_length_ratio=minimum_padding_length_ratio,
        top_fractions=top_fractions,
        dic_quantiles=dic_quantiles,
        mode=mode,
        decision_thresholds=decision_thresholds,
    )

    temporary = _prepare_destination(destination, overwrite=overwrite)
    try:
        _write_metrics_csv(temporary / "metrics.csv", sweep.metrics)
        selected = _selected_lengths(sweep, save_fields)
        saved_fields: dict[str, dict[str, Any]] = {}
        if selected:
            raw_fields = {
                "evm_fe_raw.npy": fem_evm,
                "evm_dic_reference.npy": dic_evm,
            }
            if peeq is not None:
                raw_fields["peeq_raw.npy"] = peeq
            for filename, values in raw_fields.items():
                path = temporary / "fields" / filename
                _atomic_save_array(path, values)
                saved_fields[filename] = {
                    "shape": list(np.shape(values)),
                    "sha256": fingerprint_file(path),
                }
            for length in sweep.lengths:
                if length.length_mm not in selected:
                    continue
                filename = _field_filename("evm_fe", length)
                path = temporary / "fields" / filename
                _atomic_save_array(path, sweep.filtered_evm[length.length_mm])
                saved_fields[filename] = {
                    "shape": list(sweep.filtered_evm[length.length_mm].shape),
                    "sha256": fingerprint_file(path),
                }
                if peeq is not None:
                    filename = _field_filename("peeq", length)
                    path = temporary / "fields" / filename
                    _atomic_save_array(path, sweep.filtered_peeq[length.length_mm])
                    saved_fields[filename] = {
                        "shape": list(sweep.filtered_peeq[length.length_mm].shape),
                        "sha256": fingerprint_file(path),
                    }
        _write_figures(
            temporary / "figures",
            dic_evm=dic_evm,
            fem_evm=fem_evm,
            peeq=peeq,
            sweep=sweep,
            spacing_x_mm=spacing_x_mm,
            spacing_y_mm=spacing_y_mm,
            core_slice=partition.core_element_slice_local,
        )

        repository = Path(__file__).resolve().parents[3]
        input_files = {
            "prepared_manifest": input_path / "manifest.json",
            "displacement_x_mm": input_path / "displacement_x_mm.npy",
            "displacement_y_mm": input_path / "displacement_y_mm.npy",
            "campaign_manifest": manifest_path,
            "partition_status": status_path,
            "partition_U": u_path,
            "partition_PEEQ": peeq_path,
        }
        input_metadata = {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": fingerprint_file(path),
            }
            for name, path in input_files.items()
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "software": {
                "name": "kinematics-driven-316l-strain-reconstruction",
                "version": __version__,
                "git": _git_state(repository),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "inputs": input_metadata,
            "partition": {
                **partition.as_dict(),
                "global_shape": list(layout.global_shape),
                "nominal_padding_pixels": layout.padding,
                "padding_sides": {
                    name: {
                        "pixels": pixels,
                        "um": pixels * spacing_x_mm * 1000.0,
                        "mm": pixels * spacing_x_mm,
                    }
                    for name, pixels in sides_pixels.items()
                },
                "minimum_artificial_padding_pixels": minimum_padding_pixels,
            },
            "spacing": {
                "spacing_x_mm": spacing_x_mm,
                "spacing_y_mm": spacing_y_mm,
                "pixel_size_um": spacing_x_mm * 1000.0,
            },
            "length_scales": [asdict(length) for length in lengths],
            "filter": {
                "equation": "(I + ell^2 L_h) q_bar = q",
                "operator": "positive cell-centred finite-difference -Laplacian",
                "solver": "orthonormal scipy.fft DCT-II",
                "boundary_condition": "homogeneous Neumann flux",
                "axis_convention": "axis 0 = x, axis 1 = y",
            },
            "observable": {
                "primary": "EVM_HISTORICAL",
                "construction": [
                    "strain_from_displacement",
                    "plane_stress_equivalent_strain(engineering shear)",
                    "cell_average",
                ],
                "dic_filtered": False,
                "secondary": "PEEQ" if include_peeq else None,
                "peeq_amplitude_compared_to_dic_evm": False,
            },
            "options": {
                "mode": mode,
                "top_fractions": list(top_fractions),
                "dic_quantiles": list(dic_quantiles),
                "minimum_padding_length_ratio": minimum_padding_length_ratio,
                "save_fields": save_fields,
                "include_peeq": include_peeq,
                "decision_thresholds": (
                    asdict(decision_thresholds) if decision_thresholds is not None else None
                ),
            },
            "saved_fields": saved_fields,
        }
        _atomic_write_text(
            temporary / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        report = {
            "schema_version": 1,
            "status": "completed",
            "scientific_question": (
                "Does a scalar spatial regularization length improve FEM-DIC agreement "
                "without changing the mechanical solution?"
            ),
            "facts": {
                "filter_checks": list(sweep.filter_checks),
                "metrics_csv_sha256": fingerprint_file(temporary / "metrics.csv"),
                "metric_row_count": len(sweep.metrics),
                "all_lengths": [asdict(length) for length in sweep.lengths],
                "boundary_contaminated_lengths_mm": sorted(
                    {
                        float(record["length_mm"])
                        for record in sweep.metrics
                        if record["boundary_status"] == "boundary_contaminated"
                    }
                ),
            },
            "diagnostic_result": sweep.selection,
            "physical_interpretation": {
                "identified_material_length": None,
                "mechanical_solution_modified": False,
                "statement": (
                    "This scalar post-processing experiment diagnoses spatial width only. "
                    "It does not identify an internal material length or validate a coupled "
                    "nonlocal constitutive model."
                ),
                "allowed_final_conclusions": [
                    "spatial_width_hypothesis_supported",
                    "spatial_width_hypothesis_partially_supported",
                    "spatial_width_hypothesis_insufficient",
                ],
                "automatic_conclusion": None,
            },
        }
        _atomic_write_text(
            temporary / "report.json",
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        _promote_destination(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report
