"""Comparative raw-field visualisation for coupling campaigns."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.data_preparation import fingerprint_file
from fem_inhouse.partitioning.stitch import extract_partition_field
from fem_inhouse.postprocessing.metrics import FieldErrorMetrics, field_error_metrics
from fem_inhouse.workflows.coupled_nonlocal_validation import (
    _load_json,
    _load_verified_field,
    _partition_from_manifest,
    _validate_campaign_pair,
)
from fem_inhouse.workflows.nonlocality_diagnostic import reconstruct_historical_evm
from fem_inhouse.workflows.partitioned import fingerprint_array

FloatArray = NDArray[np.float64]
ALPHAS = (0.0, 0.5, 1.0, 2.0)
OPTIONAL_FIELDS = ("PEEQ_NONLOCAL", "PEEQ_MISMATCH", "NONLOCAL_HARDENING_MPA")


@dataclass(frozen=True, slots=True)
class CoupledAlphaVisualizationData:
    """Validated core fields and derived quantities used by every figure."""

    partition_id: int
    core_bounds: tuple[int, int, int, int]
    core_shape: tuple[int, int]
    spacing_x_mm: float
    spacing_y_mm: float
    extent_mm: tuple[float, float, float, float]
    dic_evm: FloatArray
    evm_by_alpha: tuple[FloatArray, ...]
    peeq_by_alpha: tuple[FloatArray, ...]
    metrics_by_alpha: tuple[FieldErrorMetrics, ...]
    optional_fields: dict[str, tuple[FloatArray, ...]]
    manifest_hashes: dict[str, str]
    h_ref_mpa: float
    hchi_by_alpha_mpa: tuple[float, ...]
    actual_alpha_by_campaign: tuple[float, ...]


def _finite_2d(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional field")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_percentile(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value) or not 0.0 < value <= 100.0:
        raise ValueError(f"{name} must be finite and in (0, 100]")
    return float(value)


def common_color_limits(
    fields: Iterable[ArrayLike],
    *,
    percentile: float | None = None,
    lower: float = 0.0,
) -> tuple[float, float, float, float, float | None]:
    """Return real min/max and the common sequential colour limits."""

    percentile = _validate_percentile(percentile, name="percentile")
    arrays = [_finite_2d(field, name="field") for field in fields]
    if not arrays:
        raise ValueError("at least one field is required")
    values = np.concatenate([array.ravel() for array in arrays])
    real_min = float(np.min(values))
    real_max = float(np.max(values))
    used_max = real_max if percentile is None else float(np.percentile(values, percentile))
    if not np.isfinite(used_max) or used_max <= lower:
        used_max = max(real_max, lower + np.finfo(float).eps)
    return real_min, real_max, float(lower), used_max, percentile


def symmetric_color_limit(
    fields: Iterable[ArrayLike],
    *,
    percentile: float | None = None,
) -> tuple[float, float, float, float | None]:
    """Return real absolute extrema and a symmetric divergent colour limit."""

    percentile = _validate_percentile(percentile, name="percentile")
    arrays = [_finite_2d(field, name="difference") for field in fields]
    if not arrays:
        raise ValueError("at least one difference field is required")
    values = np.concatenate([np.abs(array.ravel()) for array in arrays])
    real_max = float(np.max(values))
    used_max = real_max if percentile is None else float(np.percentile(values, percentile))
    if not np.isfinite(used_max) or used_max <= 0.0:
        used_max = max(real_max, np.finfo(float).eps)
    return -used_max, used_max, real_max, percentile


def _manifest_hash(path: Path) -> str:
    return fingerprint_file(path)


def _load_verified_input(
    input_directory: Path,
    manifest: dict[str, Any],
    name: str,
) -> FloatArray:
    path = input_directory / f"{name}.npy"
    expected = manifest.get("inputs", {}).get(name)
    if expected is None:
        raise ValueError(f"manifest does not declare input {name}")
    values = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False), dtype=np.float64)
    if fingerprint_array(values) != expected:
        raise RuntimeError(f"input field fails its manifest hash: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"input field contains non-finite values: {path}")
    return values


def _status(campaign: Path, partition_id: int) -> dict[str, Any]:
    status_path = campaign / "partitions" / f"{partition_id:04d}" / "status.json"
    status = _load_json(status_path)
    if not status.get("complete"):
        raise RuntimeError(f"campaign partition is not complete: {status_path}")
    return status


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def prepare_coupled_alpha_fields(
    *,
    input_directory: str | Path,
    campaigns: Sequence[str | Path],
    partition_id: int,
    strain_vmax_percentile: float | None = None,
    peeq_vmax_percentile: float | None = None,
    difference_vmax_percentile: float | None = None,
    include_optional_fields: bool = False,
    alpha_values: Sequence[float] = ALPHAS,
) -> tuple[CoupledAlphaVisualizationData, dict[str, Any]]:
    """Load, validate, crop and derive four raw coupling states."""

    if len(campaigns) != 4 or len(alpha_values) != 4:
        raise ValueError("exactly four campaigns and four alpha values are required")
    alphas = tuple(float(value) for value in alpha_values)
    if not np.isfinite(alphas).all() or min(alphas) < 0.0 or len(set(alphas)) != 4:
        raise ValueError("alpha_values must contain four distinct finite nonnegative values")
    strain_vmax_percentile = _validate_percentile(
        strain_vmax_percentile,
        name="strain_vmax_percentile",
    )
    peeq_vmax_percentile = _validate_percentile(
        peeq_vmax_percentile,
        name="peeq_vmax_percentile",
    )
    difference_vmax_percentile = _validate_percentile(
        difference_vmax_percentile,
        name="difference_vmax_percentile",
    )
    inputs = Path(input_directory)
    campaign_paths = tuple(Path(path) for path in campaigns)
    manifests = tuple(_load_json(path / "manifest.json") for path in campaign_paths)
    local_manifest = manifests[0]
    layout, partition = _partition_from_manifest(local_manifest, partition_id)
    for manifest in manifests[1:]:
        _validate_campaign_pair(local_manifest, manifest)
        if manifest.get("layout") != local_manifest.get("layout"):
            raise ValueError("campaigns do not use identical layouts")
    statuses = tuple(_status(path, partition_id) for path in campaign_paths)
    core = partition.core_element_slice_local
    core_shape = (int(partition.core_shape[0]), int(partition.core_shape[1]))
    solve_shape = tuple(int(value) for value in partition.solve_shape)
    expected_u_shape = (solve_shape[0] + 1, solve_shape[1] + 1, 2)
    u_fields: list[FloatArray] = []
    peeq_fields: list[FloatArray] = []
    optional_fields: dict[str, list[FloatArray]] = {name: [] for name in OPTIONAL_FIELDS}
    optional_fallbacks: dict[str, str] = {}
    manifest_hashes: dict[str, str] = {}
    for index, (campaign, _manifest, status) in enumerate(
        zip(campaign_paths, manifests, statuses, strict=True)
    ):
        alpha_token = f"{alphas[index]:g}".replace(".", "p")
        manifest_hashes[f"alpha_{alpha_token}"] = _manifest_hash(campaign / "manifest.json")
        u = _load_verified_field(campaign, partition_id=partition_id, status=status, name="U")
        if u.shape != expected_u_shape:
            raise ValueError(f"U shape {u.shape} does not match {expected_u_shape}")
        u_fields.append(u)
        peeq = _load_verified_field(campaign, partition_id=partition_id, status=status, name="PEEQ")
        if peeq.ndim != 2 or peeq.shape != solve_shape:
            raise ValueError(f"PEEQ shape {peeq.shape} does not match {solve_shape}")
        peeq_fields.append(_finite_2d(peeq[core], name="PEEQ"))
        if include_optional_fields:
            for name in OPTIONAL_FIELDS:
                try:
                    values = _load_verified_field(
                        campaign,
                        partition_id=partition_id,
                        status=status,
                        name=name,
                    )
                except FileNotFoundError:
                    if index != 0:
                        raise
                    if name == "PEEQ_NONLOCAL":
                        values = peeq
                        optional_fallbacks[name] = "local PEEQ (Hchi=0 control)"
                    elif name == "PEEQ_MISMATCH":
                        values = np.zeros_like(peeq)
                        optional_fallbacks[name] = "zero (Hchi=0 control)"
                    else:
                        values = np.zeros_like(peeq)
                        optional_fallbacks[name] = "zero MPa (Hchi=0 control)"
                if values.ndim != 2 or values.shape != solve_shape:
                    raise ValueError(f"{name} shape {values.shape} does not match {solve_shape}")
                optional_fields[name].append(_finite_2d(values[core], name=name))

    base_pixel_mm = float(local_manifest["config"]["mesh"]["base_pixel_size_mm"])
    scale_factor = float(local_manifest["config"]["mesh"]["scale_factor"])
    spacing_x_mm = base_pixel_mm * scale_factor
    spacing_y_mm = spacing_x_mm
    if not np.isfinite(spacing_x_mm) or spacing_x_mm <= 0.0:
        raise ValueError("campaign mesh has an invalid physical pixel size")
    poisson_ratio = float(local_manifest["config"]["material"]["poisson_ratio"])
    dic_x = _load_verified_input(inputs, local_manifest, "displacement_x_mm")
    dic_y = _load_verified_input(inputs, local_manifest, "displacement_y_mm")
    expected_global_nodal_shape = tuple(value + 1 for value in layout.global_shape)
    if dic_x.shape != expected_global_nodal_shape or dic_y.shape != dic_x.shape:
        raise ValueError("DIC displacement fields do not match the manifest global nodal shape")
    dic_u = np.stack(
        (
            extract_partition_field(dic_x, layout=layout, partition=partition, location="node"),
            extract_partition_field(dic_y, layout=layout, partition=partition, location="node"),
        ),
        axis=-1,
    )
    dic_evm = reconstruct_historical_evm(
        dic_u,
        spacing_x_mm=spacing_x_mm,
        spacing_y_mm=spacing_y_mm,
        poisson_ratio=poisson_ratio,
    )
    dic_evm_core = _finite_2d(dic_evm[core], name="DIC EVM")
    evm_fields = []
    metrics = []
    for u in u_fields:
        evm = reconstruct_historical_evm(
            u,
            spacing_x_mm=spacing_x_mm,
            spacing_y_mm=spacing_y_mm,
            poisson_ratio=poisson_ratio,
        )
        cropped = _finite_2d(evm[core], name="FEM EVM")
        if cropped.shape != core_shape:
            raise ValueError(f"FEM EVM core has shape {cropped.shape}, expected {core_shape}")
        evm_fields.append(cropped)
        metrics.append(field_error_metrics(dic_evm_core, cropped))

    href_path = campaign_paths[0] / "HREF.json"
    if not href_path.is_file():
        raise FileNotFoundError(f"missing H_ref report: {href_path}")
    href_report = _load_json(href_path)
    h_ref_mpa = float(href_report["reference_hardening_modulus_mpa"])
    if not np.isfinite(h_ref_mpa) or h_ref_mpa <= 0.0:
        raise ValueError("H_ref must be finite and strictly positive")
    hchi_values: list[float] = []
    actual_alpha: list[float] = []
    for expected_alpha, manifest in zip(alphas, manifests, strict=True):
        nonlocal_config = manifest.get("config", {}).get("nonlocal_plasticity", {})
        hchi = float(nonlocal_config.get("coupling_modulus_mpa", 0.0))
        enabled = bool(nonlocal_config.get("enabled", False))
        if expected_alpha == 0.0:
            if enabled and not np.isclose(hchi, 0.0, atol=1e-12):
                raise ValueError("alpha=0 campaign is not the local reference")
        elif not enabled or not np.isclose(hchi / h_ref_mpa, expected_alpha, rtol=1e-8, atol=1e-10):
            raise ValueError(f"campaign does not match expected alpha={expected_alpha}")
        hchi_values.append(hchi)
        actual_alpha.append(hchi / h_ref_mpa)

    core_x0, core_x1, core_y0, core_y1 = partition.core_bounds
    extent_mm = (
        core_x0 * spacing_x_mm,
        core_x1 * spacing_x_mm,
        core_y0 * spacing_y_mm,
        core_y1 * spacing_y_mm,
    )
    data = CoupledAlphaVisualizationData(
        partition_id=partition_id,
        core_bounds=(core_x0, core_x1, core_y0, core_y1),
        core_shape=core_shape,
        spacing_x_mm=spacing_x_mm,
        spacing_y_mm=spacing_y_mm,
        extent_mm=extent_mm,
        dic_evm=dic_evm_core,
        evm_by_alpha=tuple(evm_fields),
        peeq_by_alpha=tuple(peeq_fields),
        metrics_by_alpha=tuple(metrics),
        optional_fields={name: tuple(values) for name, values in optional_fields.items() if values},
        manifest_hashes=manifest_hashes,
        h_ref_mpa=h_ref_mpa,
        hchi_by_alpha_mpa=tuple(hchi_values),
        actual_alpha_by_campaign=tuple(actual_alpha),
    )
    metadata = {
        "partition_id": partition_id,
        "core_bounds": list(data.core_bounds),
        "core_shape": list(data.core_shape),
        "solve_shape": list(solve_shape),
        "spacing_x_mm": spacing_x_mm,
        "spacing_y_mm": spacing_y_mm,
        "extent_mm": list(extent_mm),
        "alphas": list(alphas),
        "actual_alpha_by_campaign": list(actual_alpha),
        "h_ref_mpa": h_ref_mpa,
        "hchi_by_alpha_mpa": list(hchi_values),
        "campaign_manifest_sha256": manifest_hashes,
        "campaign_paths": [str(path) for path in campaign_paths],
        "input_directory": str(inputs),
        "evm_post_filter_applied": False,
        "evm_construction": "reconstruct_historical_evm from DIC/FEM nodal displacements",
        "peeq_interpretation": "internal plasticity variable; no experimental PEEQ comparison",
        "optional_field_fallbacks": optional_fallbacks,
        "git_commit": _git_commit(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "metrics": [
            {
                "alpha": alpha,
                "rmse": metric.rmse,
                "mae": metric.mae,
                "relative_l2_error": metric.relative_l2_error,
                "pearson_correlation": metric.pearson_correlation,
                "signed_mean_error": metric.signed_mean_error,
            }
            for alpha, metric in zip(alphas, metrics, strict=True)
        ],
        "peeq_statistics": [
            {
                "alpha": alpha,
                "minimum": float(np.min(field)),
                "maximum": float(np.max(field)),
                "mean": float(np.mean(field)),
                "standard_deviation": float(np.std(field)),
            }
            for alpha, field in zip(alphas, peeq_fields, strict=True)
        ],
        "color_limits": {},
    }
    return data, metadata


def _save_figure(figure: Any, output_stem: Path, formats: Sequence[str], dpi: int) -> list[str]:
    paths: list[str] = []
    for file_format in formats:
        path = output_stem.with_suffix(f".{file_format}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path))
    return paths


def _draw_field(
    axis: Any,
    field: FloatArray,
    *,
    title: str,
    extent: tuple[float, ...],
    limits: tuple[float, float],
    cmap: str,
) -> Any:
    image = axis.imshow(
        field.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
    )
    axis.set_title(title)
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    return image


def _add_common_colorbar(figure: Any, axes: Sequence[Any], image: Any, label: str) -> None:
    figure.colorbar(image, ax=list(axes), shrink=0.82, label=label)


def plot_coupled_alpha_fields(
    *,
    input_directory: str | Path,
    local_campaign: str | Path,
    coupled_campaigns: Sequence[tuple[float, str | Path]] | None = None,
    campaign_a050: str | Path | None = None,
    campaign_a100: str | Path | None = None,
    campaign_a200: str | Path | None = None,
    partition_id: int,
    output_directory: str | Path,
    dpi: int = 180,
    formats: Sequence[str] = ("png", "pdf", "svg"),
    strain_vmax_percentile: float | None = None,
    peeq_vmax_percentile: float | None = None,
    difference_vmax_percentile: float | None = None,
    include_optional_fields: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create comparative figures and write reproducibility metadata.

    ``coupled_campaigns`` is the generic interface and must contain three
    ``(alpha, path)`` pairs.  The three named campaign arguments remain
    available for compatibility with the original alpha=0.5,1,2 workflow.
    """

    if dpi <= 0:
        raise ValueError("dpi must be strictly positive")
    legacy_campaigns = (campaign_a050, campaign_a100, campaign_a200)
    if coupled_campaigns is not None and any(path is not None for path in legacy_campaigns):
        raise ValueError("use either coupled_campaigns or the legacy campaign arguments")
    if coupled_campaigns is None:
        if any(path is None for path in legacy_campaigns):
            raise ValueError(
                "provide exactly three coupled campaigns, "
                "either generically or via legacy arguments"
            )
        legacy_paths = tuple(path for path in legacy_campaigns if path is not None)
        selected_campaigns: tuple[tuple[float, str | Path], ...] = tuple(
            zip((0.5, 1.0, 2.0), legacy_paths, strict=True)
        )
    else:
        selected_campaigns = tuple(coupled_campaigns)
    if len(selected_campaigns) != 3:
        raise ValueError("exactly three coupled campaigns are required")
    coupled_alphas = tuple(float(alpha) for alpha, _path in selected_campaigns)
    coupled_paths = tuple(path for _alpha, path in selected_campaigns)
    alpha_values = (0.0, *coupled_alphas)
    normalized_formats = tuple(
        dict.fromkeys(format_value.lower().lstrip(".") for format_value in formats)
    )
    allowed_formats = {"png", "pdf", "svg"}
    if not normalized_formats or not set(normalized_formats) <= allowed_formats:
        raise ValueError("formats must be a non-empty subset of png, pdf, svg")
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data, metadata = prepare_coupled_alpha_fields(
        input_directory=input_directory,
        campaigns=(local_campaign, *coupled_paths),
        partition_id=partition_id,
        strain_vmax_percentile=strain_vmax_percentile,
        peeq_vmax_percentile=peeq_vmax_percentile,
        difference_vmax_percentile=difference_vmax_percentile,
        include_optional_fields=include_optional_fields,
        alpha_values=alpha_values,
    )
    alphas = tuple(float(value) for value in alpha_values)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    evm_fields = (data.dic_evm, *data.evm_by_alpha)
    evm_limits_raw = common_color_limits(evm_fields, percentile=strain_vmax_percentile)
    evm_limits = (evm_limits_raw[2], float(evm_limits_raw[3]))
    differences = tuple(field - data.dic_evm for field in data.evm_by_alpha)
    diff_limits_raw = symmetric_color_limit(differences, percentile=difference_vmax_percentile)
    diff_limits = (diff_limits_raw[0], diff_limits_raw[1])
    peeq_limits_raw = common_color_limits(data.peeq_by_alpha, percentile=peeq_vmax_percentile)
    peeq_limits = (peeq_limits_raw[2], float(peeq_limits_raw[3]))
    metadata["color_limits"] = {
        "evm": {
            "real_min": evm_limits_raw[0],
            "real_max": evm_limits_raw[1],
            "vmin": evm_limits[0],
            "vmax": evm_limits[1],
            "percentile": strain_vmax_percentile,
        },
        "difference": {
            "real_min": float(np.min(np.concatenate([field.ravel() for field in differences]))),
            "real_max": float(np.max(np.concatenate([field.ravel() for field in differences]))),
            "real_absolute_max": diff_limits_raw[2],
            "vmin": diff_limits[0],
            "vmax": diff_limits[1],
            "percentile": difference_vmax_percentile,
        },
        "peeq": {
            "real_min": peeq_limits_raw[0],
            "real_max": peeq_limits_raw[1],
            "vmin": peeq_limits[0],
            "vmax": peeq_limits[1],
            "percentile": peeq_vmax_percentile,
        },
    }
    metadata["plot_options"] = {
        "dpi": dpi,
        "formats": list(normalized_formats),
        "include_optional_fields": include_optional_fields,
        "helmholtz_filter_applied_to_evm": False,
    }
    figure_paths: dict[str, list[str]] = {}
    extent = data.extent_mm
    figure_prefix = f"p{data.partition_id:04d}"
    alpha_titles = ["DIC", *(f"alpha = {alpha:g}" for alpha in alphas)]

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False, constrained_layout=True)
    flat_axes = list(axes.ravel())
    for axis, field, title in zip(flat_axes[:-1], evm_fields, alpha_titles, strict=True):
        image = _draw_field(
            axis, field, title=title, extent=extent, limits=evm_limits, cmap="viridis"
        )
    flat_axes[-1].axis("off")
    _add_common_colorbar(figure, flat_axes[:-1], image, "Total equivalent strain, EVM")
    figure_paths["total_evm_comparison"] = _save_figure(
        figure, output / f"{figure_prefix}_total_evm_comparison", normalized_formats, dpi
    )
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False, constrained_layout=True)
    flat_axes = list(axes.ravel())
    for axis, alpha, field, metric in zip(
        flat_axes,
        alphas,
        differences,
        data.metrics_by_alpha,
        strict=True,
    ):
        image = _draw_field(
            axis,
            field,
            title=(
                f"alpha = {alpha:g}\nr = {metric.pearson_correlation:.3f}, "
                f"rel. L2 = {metric.relative_l2_error:.3f}, RMSE = {metric.rmse:.3g}"
            ),
            extent=extent,
            limits=diff_limits,
            cmap="coolwarm",
        )
    _add_common_colorbar(figure, flat_axes, image, "FEM EVM - DIC EVM")
    figure_paths["total_evm_difference"] = _save_figure(
        figure, output / f"{figure_prefix}_total_evm_difference", normalized_formats, dpi
    )
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False, constrained_layout=True)
    flat_axes = list(axes.ravel())
    for axis, alpha, field in zip(flat_axes, alphas, data.peeq_by_alpha, strict=True):
        image = _draw_field(
            axis,
            field,
            title=f"alpha = {alpha:g}\nmax = {np.max(field):.4g}, mean = {np.mean(field):.4g}",
            extent=extent,
            limits=peeq_limits,
            cmap="magma",
        )
    _add_common_colorbar(figure, flat_axes, image, "PEEQ (internal variable)")
    figure_paths["peeq_comparison"] = _save_figure(
        figure, output / f"{figure_prefix}_peeq_comparison", normalized_formats, dpi
    )
    plt.close(figure)

    figure, (hist_axis, cdf_axis) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    bins = np.linspace(0.0, peeq_limits[1], 51)
    for alpha, field in zip(alphas, data.peeq_by_alpha, strict=True):
        values = field.ravel()
        hist_axis.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=f"alpha = {alpha:g}",
        )
        sorted_values = np.sort(values)
        cdf_axis.plot(
            sorted_values,
            np.linspace(0.0, 1.0, sorted_values.size, endpoint=True),
            label=f"alpha = {alpha:g}",
        )
    hist_axis.set_xlabel("PEEQ")
    hist_axis.set_ylabel("Empirical density")
    hist_axis.set_title(f"PEEQ distributions on P{partition_id} core")
    cdf_axis.set_xlabel("PEEQ")
    cdf_axis.set_ylabel("Cumulative fraction")
    cdf_axis.set_title("PEEQ empirical CDF")
    hist_axis.legend()
    cdf_axis.legend()
    figure_paths["peeq_distributions"] = _save_figure(
        figure, output / f"{figure_prefix}_peeq_distributions", normalized_formats, dpi
    )
    plt.close(figure)

    figure, axes = plt.subplots(2, 5, figsize=(18, 7), squeeze=False, constrained_layout=True)
    top_axes = list(axes[0])
    bottom_axes = list(axes[1])
    for axis, field, title in zip(top_axes, evm_fields, alpha_titles, strict=True):
        image = _draw_field(
            axis, field, title=title, extent=extent, limits=evm_limits, cmap="viridis"
        )
    top_axes[-1].axis("off")
    _add_common_colorbar(figure, top_axes[:-1], image, "EVM")
    bottom_axes[0].axis("off")
    bottom_axes[0].text(
        0.05,
        0.5,
        "PEEQ is an internal\nplasticity variable.\nNo experimental PEEQ\ncomparison is made.",
        va="center",
        fontsize=11,
    )
    for axis, alpha, field in zip(bottom_axes[1:], alphas, data.peeq_by_alpha, strict=True):
        image_peeq = _draw_field(
            axis, field, title=f"alpha = {alpha:g}", extent=extent, limits=peeq_limits, cmap="magma"
        )
    _add_common_colorbar(figure, bottom_axes[1:], image_peeq, "PEEQ")
    figure_paths["alpha_summary"] = _save_figure(
        figure, output / f"{figure_prefix}_alpha_summary", normalized_formats, dpi
    )
    plt.close(figure)

    if include_optional_fields:
        for field_name, fields in data.optional_fields.items():
            figure, axes = plt.subplots(
                2, 2, figsize=(11, 8), squeeze=False, constrained_layout=True
            )
            flat_axes = list(axes.ravel())
            is_divergent = field_name in {"PEEQ_MISMATCH", "NONLOCAL_HARDENING_MPA"}
            if is_divergent:
                divergent_limits = symmetric_color_limit(fields, percentile=peeq_vmax_percentile)
                limits = (divergent_limits[0], divergent_limits[1])
                metadata["color_limits"][field_name] = {
                    "real_min": float(min(np.min(field) for field in fields)),
                    "real_max": float(max(np.max(field) for field in fields)),
                    "vmin": limits[0],
                    "vmax": limits[1],
                    "percentile": peeq_vmax_percentile,
                }
            else:
                sequential_limits = common_color_limits(fields, percentile=peeq_vmax_percentile)
                limits = (sequential_limits[2], float(sequential_limits[3]))
                metadata["color_limits"][field_name] = {
                    "real_min": sequential_limits[0],
                    "real_max": sequential_limits[1],
                    "vmin": limits[0],
                    "vmax": limits[1],
                    "percentile": peeq_vmax_percentile,
                }
            cmap = "coolwarm" if is_divergent else "viridis"
            for axis, alpha, field in zip(flat_axes, alphas, fields, strict=True):
                image = _draw_field(
                    axis,
                    field,
                    title=f"{field_name}, alpha = {alpha:g}",
                    extent=extent,
                    limits=limits,
                    cmap=cmap,
                )
            _add_common_colorbar(figure, flat_axes, image, field_name)
            key = field_name.lower()
            figure_paths[key] = _save_figure(
                figure, output / f"{figure_prefix}_{key}", normalized_formats, dpi
            )
            plt.close(figure)

    metadata["figure_paths"] = figure_paths
    metadata_path = output / "plot_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "output_directory": str(output),
        "metadata_path": str(metadata_path),
        "figures": figure_paths,
        "metadata": metadata,
    }
