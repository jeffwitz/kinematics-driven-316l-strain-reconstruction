"""Compare final-state fields reached by different loading paths.

Both runs end on the same prescribed boundary displacement, so an interior
difference at the final state is path dependence of the elastoplastic solution
rather than a difference in what was imposed.

Registered in `validation/dic_multistep_p0043_path_dependence_preregistration.md`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from fem_inhouse.postprocessing.metrics import (
    field_error_metrics,
    localization_overlap_metrics,
)

FloatArray = NDArray[np.float64]

#: Relative L2 bands registered before the comparison was computed.
NEGLIGIBLE_RELATIVE_L2 = 0.05
MATERIAL_RELATIVE_L2 = 0.20

#: The discretisation control must be at least this much smaller to conclude.
CONTROL_SEPARATION_FACTOR = 3.0

#: Registered one-sided plastic bias implied by DIC noise on the measured path.
NOISE_RATCHET_FRACTION = 0.036


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


def core_slice(
    solve_bounds: tuple[int, int, int, int],
    core_bounds: tuple[int, int, int, int],
) -> tuple[slice, slice]:
    """Return the core window of a padded partition field."""

    sx0, sx1, sy0, sy1 = solve_bounds
    cx0, cx1, cy0, cy1 = core_bounds
    if not (sx0 <= cx0 < cx1 <= sx1 and sy0 <= cy0 < cy1 <= sy1):
        raise ValueError("core bounds must sit inside the solve bounds")
    return slice(cx0 - sx0, cx1 - sx0), slice(cy0 - sy0, cy1 - sy0)


def descriptive_statistics(field: NDArray[np.generic]) -> dict[str, float]:
    """Return the registered descriptive statistics of one field."""

    values = np.asarray(field, dtype=np.float64).ravel()
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "percentile_99": float(np.percentile(values, 99.0)),
        "maximum": float(np.max(values)),
    }


def band_structure_ratio(
    difference: NDArray[np.generic],
    reference: NDArray[np.generic],
    *,
    top_fraction: float = 0.1,
) -> float:
    """Compare the signed difference inside and outside the localisation bands.

    A value near one indicates a diffuse, noise-like excess; a value well above
    one indicates that the difference concentrates where plasticity localises.
    """

    delta = np.asarray(difference, dtype=np.float64).ravel()
    base = np.asarray(reference, dtype=np.float64).ravel()
    if delta.shape != base.shape:
        raise ValueError("difference and reference must have the same shape")
    if not 0.0 < top_fraction < 1.0:
        raise ValueError("top_fraction must lie strictly between zero and one")
    threshold = float(np.quantile(base, 1.0 - top_fraction))
    inside = delta[base >= threshold]
    outside = delta[base < threshold]
    if inside.size == 0 or outside.size == 0:
        return float("nan")
    outside_mean = float(np.mean(outside))
    if outside_mean == 0.0:
        return float("nan")
    return float(np.mean(inside) / outside_mean)


def disagreement_fraction(
    first: NDArray[np.generic],
    second: NDArray[np.generic],
    *,
    threshold: float,
) -> float:
    """Fraction of cells where the two fields disagree on being active."""

    left = np.asarray(first, dtype=np.float64) > threshold
    right = np.asarray(second, dtype=np.float64) > threshold
    return float(np.count_nonzero(left != right) / left.size)


def _pair_metrics(
    reference: FloatArray,
    prediction: FloatArray,
    *,
    top_fraction: float,
    activity_threshold: float,
) -> dict[str, Any]:
    errors = field_error_metrics(reference, prediction)
    overlap = localization_overlap_metrics(
        reference, prediction, top_fraction=top_fraction
    )
    return {
        "relative_l2": float(errors.relative_l2_error),
        "rmse": float(errors.rmse),
        "signed_mean_difference": float(errors.signed_mean_error),
        "maximum_absolute_difference": float(errors.maximum_absolute_error),
        "pearson_correlation": float(errors.pearson_correlation),
        "top_fraction_iou": float(overlap.intersection_over_union),
        "activity_disagreement_fraction": disagreement_fraction(
            reference, prediction, threshold=activity_threshold
        ),
    }


def _figure(
    path: Path,
    *,
    measured: FloatArray,
    proportional: FloatArray,
    difference: FloatArray,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))
    limit = float(max(np.max(measured), np.max(proportional)))
    for axis, field, title in (
        (axes[0], measured, "A: measured 40-state history"),
        (axes[1], proportional, "B: proportional ramp, 40 increments"),
    ):
        image = axis.imshow(field.T, origin="lower", vmin=0.0, vmax=limit, cmap="inferno")
        axis.set_title(f"{title}\nPEEQ")
        figure.colorbar(image, ax=axis, fraction=0.046)
    spread = float(np.max(np.abs(difference))) or 1.0
    image = axes[2].imshow(
        difference.T, origin="lower", vmin=-spread, vmax=spread, cmap="coolwarm"
    )
    axes[2].set_title("A - B\nsigned PEEQ difference")
    figure.colorbar(image, ax=axes[2], fraction=0.046)
    for axis in axes:
        axis.set_xlabel("x (elements)")
        axis.set_ylabel("y (elements)")
    figure.suptitle("P43 core: final-state PEEQ reached by two loading paths")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def export_run_as_observation_campaign(
    *,
    run_directory: str | Path,
    output_directory: str | Path,
    partition_id: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Present a multistep run in the campaign layout the DISFlow replay expects.

    `replay_dic_observation` reads `manifest.json` and
    `partitions/<id>/{U.npy,status.json}`. A multistep run stores the same
    displacement at its top level, so this writes the expected layout with
    hashes recomputed from the copied array rather than carried over.
    """

    run = Path(run_directory)
    output = Path(output_directory)
    _prepare_directory(output, overwrite=overwrite)

    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if int(report["partition_id"]) != partition_id:
        raise ValueError("the run does not hold the requested partition")
    source = run / "U.npy"
    if _sha256(source) != report["outputs"]["U.npy"]:
        raise ValueError("U.npy does not match the run report")

    partition_root = output / "partitions" / f"{partition_id:04d}"
    partition_root.mkdir(parents=True, exist_ok=True)
    destination = partition_root / "U.npy"
    destination.write_bytes(source.read_bytes())
    digest = _sha256(destination)

    manifest = {
        "schema_version": 1,
        "config": report["config"],
        "layout": {
            "partitions": [
                {
                    "partition_id": partition_id,
                    "solve_bounds": list(report["solve_bounds"]),
                    "core_bounds": list(report["core_bounds"]),
                }
            ]
        },
        "provenance": {
            "exported_from": str(run.resolve()),
            "source_status": report["status"],
            "source_mode": report["mode"],
            "source_u_sha256": report["outputs"]["U.npy"],
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status = {
        "partition_id": partition_id,
        "complete": True,
        "outputs": {"U": digest},
        "manifest_sha256": _sha256(output / "manifest.json"),
    }
    (partition_root / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"manifest": str((output / "manifest.json").resolve()), "u_sha256": digest}


def compare_multistep_path_dependence(
    *,
    measured_directory: str | Path,
    proportional_directory: str | Path,
    archived_field_path: str | Path,
    output_directory: str | Path,
    figure_directory: str | Path,
    field_name: str = "PEEQ",
    top_fraction: float = 0.1,
    activity_threshold: float = 1.0e-4,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compare final-state PEEQ between a measured history and proportional ramps."""

    measured_root = Path(measured_directory)
    proportional_root = Path(proportional_directory)
    archived_path = Path(archived_field_path)
    output = Path(output_directory)
    figures = Path(figure_directory)
    _prepare_directory(output, overwrite=overwrite)
    _prepare_directory(figures, overwrite=overwrite)

    measured_report = json.loads((measured_root / "report.json").read_text(encoding="utf-8"))
    proportional_report = json.loads(
        (proportional_root / "report.json").read_text(encoding="utf-8")
    )
    if measured_report["mode"] != "measured":
        raise ValueError("the measured directory does not hold a measured run")
    if proportional_report["mode"] != "proportional":
        raise ValueError("the proportional directory does not hold a proportional run")
    if measured_report["solve_bounds"] != proportional_report["solve_bounds"]:
        raise ValueError("the two runs do not share a support")
    if (
        measured_report["config"]["solver"]["increments"]
        != proportional_report["config"]["solver"]["increments"]
    ):
        raise ValueError("the control must use the same increment count as the measured run")

    solve_bounds = tuple(int(value) for value in measured_report["solve_bounds"])
    core_bounds = tuple(int(value) for value in measured_report["core_bounds"])
    window = core_slice(solve_bounds, core_bounds)  # type: ignore[arg-type]

    def _load(path: Path) -> FloatArray:
        values = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False), dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"{path} is not a scalar partition field")
        return np.ascontiguousarray(values[window])

    measured = _load(measured_root / f"{field_name}.npy")
    proportional = _load(proportional_root / f"{field_name}.npy")
    archived = _load(archived_path)
    if not (measured.shape == proportional.shape == archived.shape):
        raise ValueError("the three fields do not share a core window")

    difference = measured - proportional
    path_pair = _pair_metrics(
        proportional,
        measured,
        top_fraction=top_fraction,
        activity_threshold=activity_threshold,
    )
    control_pair = _pair_metrics(
        archived,
        proportional,
        top_fraction=top_fraction,
        activity_threshold=activity_threshold,
    )

    relative_l2 = path_pair["relative_l2"]
    control_l2 = control_pair["relative_l2"]
    control_separates = control_l2 * CONTROL_SEPARATION_FACTOR <= relative_l2
    if relative_l2 < NEGLIGIBLE_RELATIVE_L2:
        band = "negligible"
    elif relative_l2 <= MATERIAL_RELATIVE_L2:
        band = "present_not_dominant"
    else:
        band = "material"
    verdict = band if control_separates else "withdrawn_discretisation_not_separated"

    figure_name = f"p0043_path_dependence_{field_name.lower()}.png"
    _figure(
        output / figure_name,
        measured=measured,
        proportional=proportional,
        difference=difference,
    )
    (figures / figure_name).write_bytes((output / figure_name).read_bytes())

    rows = [
        {"pair": "A_measured_vs_B_proportional40", **path_pair},
        {"pair": "B_proportional40_vs_C_proportional20", **control_pair},
    ]
    with (output / "pair_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed_final_state_path_dependence",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "partition_id": int(measured_report["partition_id"]),
        "field": field_name,
        "preregistration": (
            "validation/dic_multistep_p0043_path_dependence_preregistration.md"
        ),
        "support": {
            "solve_bounds": list(solve_bounds),
            "core_bounds": list(core_bounds),
            "core_shape": list(measured.shape),
            "padding_excluded": True,
        },
        "sources": {
            "measured": str((measured_root / f"{field_name}.npy").resolve()),
            "measured_sha256": _sha256(measured_root / f"{field_name}.npy"),
            "proportional_40": str((proportional_root / f"{field_name}.npy").resolve()),
            "proportional_40_sha256": _sha256(proportional_root / f"{field_name}.npy"),
            "proportional_20_archived": str(archived_path.resolve()),
            "proportional_20_archived_sha256": _sha256(archived_path),
        },
        "descriptive": {
            "measured": descriptive_statistics(measured),
            "proportional_40": descriptive_statistics(proportional),
            "proportional_20": descriptive_statistics(archived),
        },
        "path_dependence": path_pair,
        "discretisation_control": control_pair,
        "thresholds": {
            "negligible_relative_l2": NEGLIGIBLE_RELATIVE_L2,
            "material_relative_l2": MATERIAL_RELATIVE_L2,
            "control_separation_factor": CONTROL_SEPARATION_FACTOR,
            "noise_ratchet_fraction": NOISE_RATCHET_FRACTION,
        },
        "discriminator": {
            "band_structure_ratio": band_structure_ratio(
                difference, proportional, top_fraction=top_fraction
            ),
            "interpretation": (
                "near 1 is a diffuse noise-like excess; well above 1 concentrates "
                "in the localisation bands"
            ),
        },
        "conclusion": {
            "relative_l2_band": band,
            "control_separates": bool(control_separates),
            "verdict": verdict,
            "exceeds_noise_ratchet": bool(relative_l2 > NOISE_RATCHET_FRACTION),
            "claim_boundary": (
                "compares two computed fields under one constitutive model and a "
                "shared endpoint; says nothing about which path the specimen followed"
            ),
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "outputs": {
            "pair_metrics.csv": _sha256(output / "pair_metrics.csv"),
            figure_name: _sha256(output / figure_name),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
