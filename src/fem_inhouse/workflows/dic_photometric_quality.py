"""Relate DIC photometric consistency to local FEM/DIC EVM agreement."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.stats import pearsonr, spearmanr

from fem_inhouse.measurement import (
    canonical_to_image_flow,
    direct_photometric_residual,
)
from fem_inhouse.postprocessing.metrics import field_error_metrics
from fem_inhouse.workflows.dic_observation_replay import (
    PIXEL_SIZE_MM,
    RAW_CROP_COLUMN_START,
    RAW_CROP_ROW_START,
)

FloatArray = NDArray[np.float64]
BooleanArray = NDArray[np.bool_]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _prepare_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _verified_field(replay: Path, name: str, report: dict[str, Any]) -> FloatArray:
    path = replay / name
    if not path.is_file() or _sha256(path) != report["outputs"][name]:
        raise ValueError(f"{name} does not match immutable replay report: {replay}")
    field = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False), dtype=np.float64)
    if field.ndim != 2 or not np.isfinite(field).all():
        raise ValueError(f"{name} must be a finite two-dimensional field")
    return field


def _association(residual: FloatArray, error: FloatArray) -> dict[str, float]:
    return {
        "pearson": float(pearsonr(residual, error).statistic),
        "spearman": float(spearmanr(residual, error).statistic),
    }


def _decile_rows(
    residual: FloatArray,
    errors: dict[str, FloatArray],
) -> list[dict[str, Any]]:
    edges = np.quantile(residual, np.linspace(0.0, 1.0, 11))
    groups = np.searchsorted(edges[1:-1], residual, side="right")
    rows: list[dict[str, Any]] = []
    for decile in range(10):
        selected = groups == decile
        if not np.any(selected):
            continue
        for label, error in errors.items():
            rows.append(
                {
                    "case": label,
                    "decile": decile + 1,
                    "count": int(np.count_nonzero(selected)),
                    "residual_min_grey": float(np.min(residual[selected])),
                    "residual_mean_grey": float(np.mean(residual[selected])),
                    "residual_max_grey": float(np.max(residual[selected])),
                    "mean_absolute_evm_error": float(np.mean(error[selected])),
                    "rmse_evm": float(np.sqrt(np.mean(np.square(error[selected])))),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _quality_figure(
    path: Path,
    *,
    residual: FloatArray,
    valid: BooleanArray,
    dic: FloatArray,
    observed: FloatArray,
    threshold: float,
) -> None:
    error = np.abs(observed - dic)
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    residual_image = axes[0, 0].imshow(residual.T, origin="upper", cmap="viridis")
    axes[0, 0].set_title("Absolute photometric residual")
    figure.colorbar(residual_image, ax=axes[0, 0], label="Grey levels")
    error_image = axes[0, 1].imshow(error.T, origin="upper", cmap="magma")
    axes[0, 1].set_title("Absolute observed-FEM EVM error")
    figure.colorbar(error_image, ax=axes[0, 1], label="Equivalent strain")
    good = valid & (residual <= threshold)
    axes[0, 2].imshow(good.T, origin="upper", cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Primary support after q90 sensitivity mask")
    for axis in axes[0]:
        axis.set_xlabel("x element")
        axis.set_ylabel("y element")

    selected_residual = residual[valid]
    selected_error = error[valid]
    axes[1, 0].hexbin(
        selected_residual,
        selected_error,
        gridsize=60,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    axes[1, 0].axvline(threshold, color="red", linestyle="--", label="q90 residual")
    axes[1, 0].set_xlabel("Absolute photometric residual (grey levels)")
    axes[1, 0].set_ylabel("Absolute EVM error")
    axes[1, 0].legend()
    common_maximum = max(float(np.max(dic)), float(np.max(observed)))
    dic_image = axes[1, 1].imshow(
        dic.T, origin="upper", cmap="magma", vmin=0.0, vmax=common_maximum
    )
    axes[1, 1].set_title("DIC EVM")
    axes[1, 2].imshow(
        observed.T,
        origin="upper",
        cmap="magma",
        vmin=0.0,
        vmax=common_maximum,
    )
    axes[1, 2].set_title("Observed FEM EVM")
    figure.colorbar(dic_image, ax=axes[1, 1:], label="Equivalent total strain")
    for axis in axes[1, 1:]:
        axis.set_xlabel("x element")
        axis.set_ylabel("y element")
    figure.suptitle("P43 photometric consistency and local-model agreement")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _decile_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    labels = sorted({str(row["case"]) for row in rows})
    for label in labels:
        selected = [row for row in rows if row["case"] == label]
        axis.plot(
            [row["decile"] for row in selected],
            [row["mean_absolute_evm_error"] for row in selected],
            marker="o",
            label=label,
        )
    axis.set_xlabel("Photometric-residual decile")
    axis.set_ylabel("Mean absolute EVM error")
    axis.set_xticks(range(1, 11))
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_title("FEM/DIC error versus local image-matching quality")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def diagnose_dic_photometric_quality(
    *,
    reference_image: str | Path,
    final_image: str | Path,
    prepared_case: str | Path,
    replays: Sequence[tuple[str, float, str | Path]],
    output_directory: str | Path,
    figure_directory: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the preregistered P43 photometric-quality diagnostic."""

    if not replays:
        raise ValueError("at least one replay is required")
    reference_path = Path(reference_image)
    final_path = Path(final_image)
    prepared = Path(prepared_case)
    output = Path(output_directory)
    figures = Path(figure_directory)
    _prepare_directory(output, overwrite=overwrite)
    _prepare_directory(figures, overwrite=overwrite)

    reference_full = np.asarray(Image.open(reference_path).convert("L"), dtype=np.uint8)
    final_full = np.asarray(Image.open(final_path).convert("L"), dtype=np.uint8)
    if reference_full.shape != final_full.shape:
        raise ValueError("reference and final images must have the same shape")
    ux = np.load(prepared / "displacement_x_mm.npy", mmap_mode="r", allow_pickle=False)
    uy = np.load(prepared / "displacement_y_mm.npy", mmap_mode="r", allow_pickle=False)
    element_shape = (ux.shape[0] - 1, ux.shape[1] - 1)
    crop = (
        slice(RAW_CROP_ROW_START, RAW_CROP_ROW_START + element_shape[0]),
        slice(RAW_CROP_COLUMN_START, RAW_CROP_COLUMN_START + element_shape[1]),
    )
    reference = np.ascontiguousarray(reference_full[crop])
    final = np.ascontiguousarray(final_full[crop])
    if reference.shape != element_shape or uy.shape != ux.shape:
        raise ValueError("prepared displacement and image crop supports are incompatible")
    displacement = np.stack((ux[:-1, :-1], uy[:-1, :-1]), axis=-1)
    flow = canonical_to_image_flow(displacement, pixel_size_mm=PIXEL_SIZE_MM)
    photometric = direct_photometric_residual(reference, final, flow)

    loaded: list[
        tuple[str, float, Path, dict[str, Any], FloatArray, FloatArray]
    ] = []
    core_bounds: list[int] | None = None
    for label, alpha, replay_value in replays:
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError("replay alpha values must be finite and nonnegative")
        replay = Path(replay_value)
        report = _json(replay / "report.json")
        if report["status"] != "completed_symmetric_image_observation":
            raise ValueError(f"replay is incomplete: {replay}")
        if report["profile"]["name"] != "legacy_script_2021":
            raise ValueError("primary diagnostic requires legacy_script_2021 replays")
        declared_core = [int(value) for value in report["core_bounds"]]
        if core_bounds is None:
            core_bounds = declared_core
        elif core_bounds != declared_core:
            raise ValueError("all replays must use the same manifest-defined core")
        dic = _verified_field(replay, "dic_evm.npy", report)
        observed = _verified_field(replay, "fem_observed_evm.npy", report)
        if dic.shape != observed.shape:
            raise ValueError("DIC and observed FEM replay fields are incompatible")
        loaded.append((label, float(alpha), replay, report, dic, observed))
    assert core_bounds is not None
    x0, x1, y0, y1 = core_bounds
    residual_core = photometric.absolute_residual_grey_levels[x0:x1, y0:y1]
    valid_core = photometric.valid_mask[x0:x1, y0:y1]
    if residual_core.shape != loaded[0][4].shape or not np.any(valid_core):
        raise ValueError("photometric and V3 replay core supports are incompatible")
    threshold = float(np.quantile(residual_core[valid_core], 0.90))
    good = valid_core & (residual_core <= threshold)

    rows: list[dict[str, Any]] = []
    error_fields: dict[str, FloatArray] = {}
    for label, alpha, replay, _report, dic, observed in loaded:
        error = np.abs(observed - dic)
        error_fields[label] = error[valid_core]
        rows.append(
            {
                "case": label,
                "replay": str(replay.resolve()),
                "replay_report_sha256": _sha256(replay / "report.json"),
                "association": _association(residual_core[valid_core], error[valid_core]),
                "unmasked": asdict(field_error_metrics(dic[valid_core], observed[valid_core])),
                "q90_masked_sensitivity": asdict(
                    field_error_metrics(dic[good], observed[good])
                ),
                "retained_fraction": float(np.mean(good[valid_core])),
                "alpha": alpha,
            }
        )
    deciles = _decile_rows(residual_core[valid_core], error_fields)
    _write_csv(output / "decile_metrics.csv", deciles)
    np.save(output / "photometric_residual.npy", residual_core)
    np.save(output / "valid_mask.npy", valid_core)
    _quality_figure(
        figures / "photometric_quality_and_error.png",
        residual=residual_core,
        valid=valid_core,
        dic=loaded[0][4],
        observed=loaded[0][5],
        threshold=threshold,
    )
    _decile_figure(figures / "photometric_deciles.png", deciles)

    report = {
        "schema_version": 1,
        "status": "completed_baseline_no_acceptance_threshold",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "partition_id": loaded[0][3]["partition_id"],
        "core_bounds": core_bounds,
        "mechanics_rerun": False,
        "micromorphic_identification_run": False,
        "photometric_residual": {
            "definition": "abs(I40(x + u_DIC(x)) - I0(x))",
            "interpolation": "bilinear",
            "intensity_normalisation": "none",
            "unit": "8-bit grey levels",
            "q90_threshold": threshold,
            "valid_fraction": float(np.mean(valid_core)),
        },
        "sensitivity_mask": {
            "definition": "exclude residual strictly above core q90",
            "primary_metrics_remain_unmasked": True,
            "retained_fraction": float(np.mean(good[valid_core])),
        },
        "rows": rows,
        "provenance": {
            "reference_image": str(reference_path.resolve()),
            "reference_image_sha256": _sha256(reference_path),
            "final_image": str(final_path.resolve()),
            "final_image_sha256": _sha256(final_path),
            "prepared_case_manifest_sha256": _sha256(prepared / "manifest.json"),
            "pixel_size_mm": PIXEL_SIZE_MM,
            "axis_convention": (
                "image row -> canonical x/ux; image column -> canonical y/uy"
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    report["outputs"] = {
        path.name: _sha256(path)
        for path in sorted((*output.iterdir(), *figures.iterdir()))
        if path.is_file() and path.name != "report.json"
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
