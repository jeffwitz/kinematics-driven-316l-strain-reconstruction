"""Propagate the measured repeated-frame DIC residual over archived P43 fields."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from fem_inhouse.measurement import disflow_profile, image_flow_to_canonical, run_disflow
from fem_inhouse.postprocessing.metrics import (
    absolute_threshold_overlap_metrics,
    field_error_metrics,
    localization_overlap_metrics,
)
from fem_inhouse.workflows.dic_observation_replay import (
    PIXEL_SIZE_MM,
    RAW_CROP_COLUMN_START,
    RAW_CROP_ROW_START,
)
from fem_inhouse.workflows.nonlocality_diagnostic import reconstruct_historical_evm

FloatArray = NDArray[np.float64]

RAW_CROP_SHAPE = (3600, 3100)
METRIC_DIRECTIONS = {
    "rmse": "min",
    "relative_l2": "min",
    "pearson": "max",
    "top10_iou": "max",
    "absolute_q90_iou": "max",
    "absolute_q90_active_fraction_error": "min",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _prepare_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _verified_replay_field(replay: Path, name: str, report: Mapping[str, Any]) -> FloatArray:
    path = replay / name
    expected = report["outputs"][name]
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{name} does not match immutable replay report: {replay}")
    values = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False), dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite two-dimensional field")
    return values


def periodic_residual_on_support(
    centred_flow_pixels: NDArray[np.generic],
    *,
    solve_bounds: Sequence[int],
    shift_x: int,
    shift_y: int,
    sign: int,
) -> FloatArray:
    """Sample one signed periodic translation on a nodal solve support."""

    flow = np.asarray(centred_flow_pixels, dtype=np.float64)
    if flow.ndim != 3 or flow.shape[-1] != 2 or not np.isfinite(flow).all():
        raise ValueError("centred_flow_pixels must have finite shape (nx, ny, 2)")
    if len(solve_bounds) != 4:
        raise ValueError("solve_bounds must contain x0, x1, y0, y1")
    if sign not in {-1, 1}:
        raise ValueError("sign must be -1 or 1")
    x0, x1, y0, y1 = (int(value) for value in solve_bounds)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("solve bounds must be non-empty")
    x_indices = (np.arange(x0, x1 + 1) + int(shift_x)) % flow.shape[0]
    y_indices = (np.arange(y0, y1 + 1) + int(shift_y)) % flow.shape[1]
    return sign * np.asarray(flow[np.ix_(x_indices, y_indices)], dtype=np.float64)


def _metric_row(reference: FloatArray, prediction: FloatArray) -> dict[str, float]:
    errors = field_error_metrics(reference, prediction)
    relative = localization_overlap_metrics(reference, prediction, top_fraction=0.1)
    absolute = absolute_threshold_overlap_metrics(reference, prediction, reference_quantile=0.9)
    return {
        "rmse": errors.rmse,
        "relative_l2": errors.relative_l2_error,
        "pearson": errors.pearson_correlation,
        "top10_iou": relative.intersection_over_union,
        "absolute_q90_iou": absolute.intersection_over_union,
        "absolute_q90_prediction_active_fraction": absolute.prediction_active_fraction,
        "absolute_q90_active_fraction_error": abs(
            absolute.prediction_active_fraction - absolute.reference_active_fraction
        ),
    }


def _summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    labels = sorted({str(row["label"]) for row in rows})
    metric_names = tuple(METRIC_DIRECTIONS)
    for label in labels:
        selected = [row for row in rows if row["label"] == label]
        summary: dict[str, Any] = {
            "label": label,
            "alpha": float(selected[0]["alpha"]),
            "sample_count": len(selected),
            "metrics": {},
        }
        for metric in metric_names:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            summary["metrics"][metric] = {
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "q2_5": float(np.quantile(values, 0.025)),
                "q97_5": float(np.quantile(values, 0.975)),
            }
        summaries.append(summary)
    return summaries


def _ranking_probabilities(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    sample_ids = sorted({int(row["sample"]) for row in rows})
    labels = sorted({str(row["label"]) for row in rows})
    counts = {metric: {label: 0.0 for label in labels} for metric in METRIC_DIRECTIONS}
    for sample in sample_ids:
        selected = [row for row in rows if int(row["sample"]) == sample]
        for metric, direction in METRIC_DIRECTIONS.items():
            target = (
                min(float(row[metric]) for row in selected)
                if direction == "min"
                else max(float(row[metric]) for row in selected)
            )
            winners = [
                str(row["label"])
                for row in selected
                if np.isclose(float(row[metric]), target, rtol=1.0e-12, atol=1.0e-15)
            ]
            for winner in winners:
                counts[metric][winner] += 1 / len(winners)
    return {
        metric: {label: value / len(sample_ids) for label, value in values.items()}
        for metric, values in counts.items()
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _interval_figure(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    metrics = ("relative_l2", "pearson", "top10_iou", "absolute_q90_iou")
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    x = np.arange(len(summaries))
    labels = [f"alpha={entry['alpha']:g}" for entry in summaries]
    for axis, metric in zip(axes.flat, metrics, strict=True):
        medians = [entry["metrics"][metric]["median"] for entry in summaries]
        lower = [entry["metrics"][metric]["q2_5"] for entry in summaries]
        upper = [entry["metrics"][metric]["q97_5"] for entry in summaries]
        axis.errorbar(
            x,
            medians,
            yerr=[np.subtract(medians, lower), np.subtract(upper, medians)],
            fmt="o",
            capsize=4,
        )
        axis.set_xticks(x, labels)
        axis.set_title(metric.replace("_", " "))
        axis.grid(alpha=0.25)
    figure.suptitle("P43 surrogate sensitivity to measured repeat-frame DIC residual")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _ranking_figure(path: Path, probabilities: Mapping[str, Mapping[str, float]]) -> None:
    metrics = list(probabilities)
    labels = list(next(iter(probabilities.values())))
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    x = np.arange(len(metrics))
    width = 0.8 / len(labels)
    for index, label in enumerate(labels):
        axis.bar(
            x + (index - (len(labels) - 1) / 2) * width,
            [probabilities[metric][label] for metric in metrics],
            width,
            label=label,
        )
    axis.set_xticks(x, [metric.replace("_", "\n") for metric in metrics])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Fraction of surrogate samples ranked best")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def propagate_dic_uncertainty(
    *,
    final_image: str | Path,
    repeat_image: str | Path,
    prepared_case: str | Path,
    replays: Sequence[tuple[str, float, str | Path]],
    output_directory: str | Path,
    figure_directory: str | Path,
    sample_count: int = 256,
    seed: int = 20260729,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the preregistered light uncertainty propagation without mechanics."""

    if sample_count < 2:
        raise ValueError("sample_count must be at least two")
    if not replays:
        raise ValueError("at least one replay is required")
    output = Path(output_directory)
    figures = Path(figure_directory)
    prepared = Path(prepared_case)
    final_path = Path(final_image)
    repeat_path = Path(repeat_image)
    _prepare_directory(output, overwrite=overwrite)
    _prepare_directory(figures, overwrite=overwrite)

    final_full = np.asarray(Image.open(final_path).convert("L"), dtype=np.uint8)
    repeat_full = np.asarray(Image.open(repeat_path).convert("L"), dtype=np.uint8)
    if final_full.shape != repeat_full.shape:
        raise ValueError("final and repeat images must have the same shape")
    crop = (
        slice(RAW_CROP_ROW_START, RAW_CROP_ROW_START + RAW_CROP_SHAPE[0]),
        slice(RAW_CROP_COLUMN_START, RAW_CROP_COLUMN_START + RAW_CROP_SHAPE[1]),
    )
    final_crop = np.ascontiguousarray(final_full[crop])
    repeat_crop = np.ascontiguousarray(repeat_full[crop])
    if final_crop.shape != RAW_CROP_SHAPE:
        raise ValueError("raw images do not contain the canonical recorded crop")

    profile = disflow_profile("legacy_script_2021")
    repeat_flow = np.asarray(
        run_disflow(final_crop, repeat_crop, config=profile.config), dtype=np.float64
    )
    component_means = np.mean(repeat_flow, axis=(0, 1))
    centred_flow = repeat_flow - component_means
    centred_flow_path = output / "centred_repeat_flow_pixels.npy"
    np.save(centred_flow_path, np.asarray(centred_flow, dtype=np.float32))

    ux = np.load(prepared / "displacement_x_mm.npy", mmap_mode="r", allow_pickle=False)
    uy = np.load(prepared / "displacement_y_mm.npy", mmap_mode="r", allow_pickle=False)
    if ux.shape != uy.shape or ux.ndim != 2:
        raise ValueError("prepared displacement components must have equal 2D shapes")

    cases: list[dict[str, Any]] = []
    solve_bounds: tuple[int, int, int, int] | None = None
    core_bounds: tuple[int, int, int, int] | None = None
    dic_baseline: FloatArray | None = None
    for label, alpha, replay_value in replays:
        replay = Path(replay_value)
        report_path = replay / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "completed_symmetric_image_observation":
            raise ValueError(f"replay is not completed: {replay}")
        if report["profile"]["name"] != "legacy_script_2021":
            raise ValueError(f"replay does not use legacy_script_2021: {replay}")
        current_solve = tuple(int(value) for value in report["solve_bounds"])
        current_core = tuple(int(value) for value in report["core_bounds"])
        if len(current_solve) != 4 or len(current_core) != 4:
            raise ValueError("replay bounds must contain four entries")
        typed_solve = (
            current_solve[0],
            current_solve[1],
            current_solve[2],
            current_solve[3],
        )
        typed_core = (
            current_core[0],
            current_core[1],
            current_core[2],
            current_core[3],
        )
        if solve_bounds is None:
            solve_bounds, core_bounds = typed_solve, typed_core
        elif (typed_solve, typed_core) != (solve_bounds, core_bounds):
            raise ValueError("replays do not share solve and core bounds")
        dic = _verified_replay_field(replay, "dic_evm.npy", report)
        observed = _verified_replay_field(replay, "fem_observed_evm.npy", report)
        if dic_baseline is None:
            dic_baseline = dic
        elif not np.array_equal(dic, dic_baseline):
            raise ValueError("replays do not share the exact DIC reference")
        cases.append(
            {
                "label": str(label),
                "alpha": float(alpha),
                "replay": replay,
                "report_sha256": _sha256(report_path),
                "prediction": observed,
            }
        )
    assert solve_bounds is not None and core_bounds is not None and dic_baseline is not None
    sx0, sx1, sy0, sy1 = solve_bounds
    cx0, cx1, cy0, cy1 = core_bounds
    if sx1 >= ux.shape[0] or sy1 >= ux.shape[1]:
        raise ValueError("solve support exceeds prepared nodal displacement")
    prepared_solve = np.stack(
        (
            np.asarray(ux[sx0 : sx1 + 1, sy0 : sy1 + 1], dtype=np.float64),
            np.asarray(uy[sx0 : sx1 + 1, sy0 : sy1 + 1], dtype=np.float64),
        ),
        axis=-1,
    )
    core_slice = (
        slice(cx0 - sx0, cx1 - sx0),
        slice(cy0 - sy0, cy1 - sy0),
    )
    reconstructed_baseline = reconstruct_historical_evm(
        prepared_solve,
        spacing_x_mm=PIXEL_SIZE_MM,
        spacing_y_mm=PIXEL_SIZE_MM,
        poisson_ratio=0.3,
    )[core_slice]
    baseline_maximum_difference = float(np.max(np.abs(reconstructed_baseline - dic_baseline)))
    if baseline_maximum_difference > 1.0e-14:
        raise ValueError(
            "prepared DIC reconstruction does not match immutable replay reference "
            f"(max difference {baseline_maximum_difference:.3e})"
        )

    baseline_rows = [
        {
            "label": case["label"],
            "alpha": case["alpha"],
            **_metric_row(dic_baseline, case["prediction"]),
        }
        for case in cases
    ]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for sample in range(sample_count):
        shift_x = int(rng.integers(0, centred_flow.shape[0]))
        shift_y = int(rng.integers(0, centred_flow.shape[1]))
        sign = int(rng.choice((-1, 1)))
        residual_image = periodic_residual_on_support(
            centred_flow,
            solve_bounds=solve_bounds,
            shift_x=shift_x,
            shift_y=shift_y,
            sign=sign,
        )
        residual_canonical_mm = image_flow_to_canonical(residual_image, pixel_size_mm=PIXEL_SIZE_MM)
        perturbed_dic = reconstruct_historical_evm(
            prepared_solve + residual_canonical_mm,
            spacing_x_mm=PIXEL_SIZE_MM,
            spacing_y_mm=PIXEL_SIZE_MM,
            poisson_ratio=0.3,
        )[core_slice]
        for case in cases:
            rows.append(
                {
                    "sample": sample,
                    "shift_x": shift_x,
                    "shift_y": shift_y,
                    "sign": sign,
                    "label": case["label"],
                    "alpha": case["alpha"],
                    **_metric_row(perturbed_dic, case["prediction"]),
                }
            )

    summaries = _summaries(rows)
    rankings = _ranking_probabilities(rows)
    samples_path = output / "samples.csv"
    _write_csv(samples_path, rows)
    _interval_figure(figures / "metric_intervals.png", summaries)
    _ranking_figure(figures / "ranking_probabilities.png", rankings)
    report = {
        "schema_version": 1,
        "status": "completed_surrogate_sensitivity_no_acceptance_threshold",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "sample_count": sample_count,
        "seed": seed,
        "partition_id": 43,
        "solve_bounds": list(solve_bounds),
        "core_bounds": list(core_bounds),
        "profile": profile.manifest(),
        "residual_model": {
            "source": "centred repeat-frame DISFlow residual",
            "final_image": str(final_path.resolve()),
            "final_image_sha256": _sha256(final_path),
            "repeat_image": str(repeat_path.resolve()),
            "repeat_image_sha256": _sha256(repeat_path),
            "component_means_pixels": component_means.tolist(),
            "component_standard_deviations_pixels": np.std(centred_flow, axis=(0, 1)).tolist(),
            "translation": "uniform periodic shifts over the canonical crop",
            "sign": "equiprobable -1 or +1",
        },
        "prepared_case": {
            "path": str(prepared.resolve()),
            "manifest_sha256": _sha256(prepared / "manifest.json"),
            "baseline_reconstruction_max_abs_difference": baseline_maximum_difference,
        },
        "replays": [
            {
                "label": case["label"],
                "alpha": case["alpha"],
                "path": str(case["replay"].resolve()),
                "report_sha256": case["report_sha256"],
            }
            for case in cases
        ],
        "baseline_metrics": baseline_rows,
        "surrogate_sensitivity_summaries": summaries,
        "ranking_probabilities": rankings,
        "metric_directions": METRIC_DIRECTIONS,
        "peeq_max": {
            "status": "not_propagated_requires_mechanical_rerun",
            "reason": "PEEQ is a mechanical internal variable, not an observation output",
        },
        "interpretation_boundary": (
            "Intervals are surrogate sensitivity intervals under translated measured "
            "repeat-frame residuals; they are not confidence intervals."
        ),
        "mechanics_rerun": False,
        "micromorphic_identification_run": False,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "outputs": {
            "samples.csv": _sha256(samples_path),
            "centred_repeat_flow_pixels.npy": _sha256(centred_flow_path),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
