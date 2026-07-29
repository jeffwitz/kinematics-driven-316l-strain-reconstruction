"""Prepare direct-reference DIC histories and run measured-boundary diagnostics."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
from numpy.typing import NDArray
from PIL import Image

from fem_inhouse.measurement import disflow_profile, image_flow_to_canonical, run_disflow
from fem_inhouse.workflows.campaign_access import (
    load_json_object,
    partition_from_manifest,
)
from fem_inhouse.workflows.dic_observation_replay import (
    PIXEL_SIZE_MM,
    RAW_CROP_COLUMN_START,
    RAW_CROP_ROW_START,
)

FloatArray = NDArray[np.float64]
RAW_CROP_SHAPE = (3600, 3100)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def anchor_displacement_history(
    raw_history_mm: NDArray[np.generic],
    prepared_final_mm: NDArray[np.generic],
) -> FloatArray:
    """Linearly anchor a direct-reference history to an immutable endpoint."""

    history = np.asarray(raw_history_mm, dtype=np.float64)
    final = np.asarray(prepared_final_mm, dtype=np.float64)
    if history.ndim != 4 or history.shape[-1] != 2:
        raise ValueError("raw_history_mm must have shape (steps + 1, nx, ny, 2)")
    if final.shape != history.shape[1:]:
        raise ValueError("prepared_final_mm must match one history state")
    if not np.isfinite(history).all() or not np.isfinite(final).all():
        raise ValueError("history and endpoint must contain finite values")
    if not np.allclose(history[0], 0.0, rtol=0.0, atol=1.0e-14):
        raise ValueError("raw history must start from zero")
    fractions = np.linspace(0.0, 1.0, history.shape[0], dtype=np.float64)
    correction = final - history[-1]
    return history + fractions[:, None, None, None] * correction


def _history_figure(
    path: Path,
    *,
    raw: FloatArray,
    anchored: FloatArray,
    final: FloatArray,
) -> None:
    fractions = np.linspace(0.0, 1.0, raw.shape[0])
    proportional = fractions[:, None, None, None] * final
    raw_deviation = np.sqrt(np.mean(np.square(raw - proportional), axis=(1, 2, 3)))
    anchored_deviation = np.sqrt(np.mean(np.square(anchored - proportional), axis=(1, 2, 3)))
    endpoint_correction = np.sqrt(np.mean(np.square(anchored - raw), axis=(1, 2, 3)))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(fractions, raw_deviation * 1.0e3, label="direct OpenCV 4.14")
    axes[0].plot(fractions, anchored_deviation * 1.0e3, label="endpoint anchored")
    axes[0].set_xlabel("Ordered image fraction")
    axes[0].set_ylabel("RMS deviation from proportional path (um)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(fractions, endpoint_correction * 1.0e3)
    axes[1].set_xlabel("Ordered image fraction")
    axes[1].set_ylabel("RMS endpoint correction (um)")
    axes[1].grid(alpha=0.25)
    figure.suptitle("P43 direct-reference DIC boundary history")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def prepare_dic_multistep_history(
    *,
    image_directory: str | Path,
    prepared_case: str | Path,
    source_campaign: str | Path,
    partition_id: int,
    output_directory: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Reconstruct the 40 direct-reference P43 displacement states."""

    images = Path(image_directory)
    prepared = Path(prepared_case)
    campaign = Path(source_campaign)
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = campaign / "manifest.json"
    campaign_manifest = load_json_object(manifest_path)
    _layout, partition = partition_from_manifest(campaign_manifest, partition_id)
    sx0, sx1, sy0, sy1 = partition.solve_bounds
    support_shape = (sx1 - sx0 + 1, sy1 - sy0 + 1)
    history_shape = (41, *support_shape, 2)

    reference_path = images / "000294.tif"
    step_paths = [images / f"{index:06d}.tif" for index in range(295, 335)]
    for path in (reference_path, *step_paths):
        if not path.is_file():
            raise FileNotFoundError(f"missing DIC image: {path}")
    reference_full = np.asarray(Image.open(reference_path).convert("L"), dtype=np.uint8)
    crop = (
        slice(RAW_CROP_ROW_START, RAW_CROP_ROW_START + RAW_CROP_SHAPE[0]),
        slice(RAW_CROP_COLUMN_START, RAW_CROP_COLUMN_START + RAW_CROP_SHAPE[1]),
    )
    reference = np.ascontiguousarray(reference_full[crop])
    if reference.shape != RAW_CROP_SHAPE:
        raise ValueError("reference image does not contain the canonical crop")

    raw_path = output / "raw_direct_history_mm.npy"
    raw = open_memmap(raw_path, mode="w+", dtype=np.float32, shape=history_shape)
    raw[0] = 0.0
    profile = disflow_profile("legacy_script_2021")
    image_hashes = {reference_path.name: _sha256(reference_path)}
    for step, path in enumerate(step_paths, start=1):
        current_full = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        current = np.ascontiguousarray(current_full[crop])
        if current.shape != reference.shape:
            raise ValueError(f"unexpected image crop shape: {path}")
        flow = run_disflow(reference, current, config=profile.config)
        displacement = image_flow_to_canonical(flow, pixel_size_mm=PIXEL_SIZE_MM)
        raw[step] = displacement[sx0 : sx1 + 1, sy0 : sy1 + 1]
        raw.flush()
        image_hashes[path.name] = _sha256(path)
        (output / "progress.json").write_text(
            json.dumps(
                {"completed_steps": step, "last_image": path.name},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    raw_values = np.asarray(raw, dtype=np.float64)

    ux = np.load(prepared / "displacement_x_mm.npy", mmap_mode="r", allow_pickle=False)
    uy = np.load(prepared / "displacement_y_mm.npy", mmap_mode="r", allow_pickle=False)
    prepared_final = np.stack(
        (
            np.asarray(ux[sx0 : sx1 + 1, sy0 : sy1 + 1], dtype=np.float64),
            np.asarray(uy[sx0 : sx1 + 1, sy0 : sy1 + 1], dtype=np.float64),
        ),
        axis=-1,
    )
    anchored = anchor_displacement_history(raw_values, prepared_final)
    anchored_path = output / "anchored_history_mm.npy"
    correction_path = output / "endpoint_correction_mm.npy"
    np.save(anchored_path, np.asarray(anchored, dtype=np.float32))
    np.save(
        correction_path,
        np.asarray(prepared_final - raw_values[-1], dtype=np.float32),
    )
    endpoint_difference = raw_values[-1] - prepared_final
    fractions = np.linspace(0.0, 1.0, anchored.shape[0])
    proportional = fractions[:, None, None, None] * prepared_final
    history_deviation = anchored - proportional
    figure_path = output / "boundary_history_diagnostic.png"
    _history_figure(
        figure_path,
        raw=raw_values,
        anchored=anchored,
        final=prepared_final,
    )
    report = {
        "schema_version": 1,
        "status": "completed_direct_reference_history_endpoint_anchored",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "partition_id": partition_id,
        "solve_bounds": list(partition.solve_bounds),
        "core_bounds": list(partition.core_bounds),
        "step_count": 40,
        "ordered_pseudo_time_only": True,
        "load_cell_synchronisation_available": False,
        "mask": {
            "mode": "declared_all_valid",
            "historical_mask_reproduced": False,
        },
        "profile": profile.manifest(),
        "pixel_size_mm": PIXEL_SIZE_MM,
        "source": {
            "campaign_manifest": str(manifest_path.resolve()),
            "campaign_manifest_sha256": _sha256(manifest_path),
            "prepared_manifest_sha256": _sha256(prepared / "manifest.json"),
            "image_hashes": image_hashes,
        },
        "endpoint_compatibility": {
            "component_rms_difference_mm": float(np.sqrt(np.mean(np.square(endpoint_difference)))),
            "maximum_absolute_component_difference_mm": float(np.max(np.abs(endpoint_difference))),
            "relative_vector_norm": float(
                np.linalg.norm(endpoint_difference) / np.linalg.norm(prepared_final)
            ),
            "anchored_final_max_abs_difference_mm": float(
                np.max(np.abs(anchored[-1] - prepared_final))
            ),
        },
        "nonproportionality": {
            "maximum_rms_deviation_from_proportional_mm": float(
                np.max(np.sqrt(np.mean(np.square(history_deviation), axis=(1, 2, 3))))
            ),
            "final_rms_deviation_from_proportional_mm": float(
                np.sqrt(np.mean(np.square(history_deviation[-1])))
            ),
        },
        "outputs": {
            raw_path.name: _sha256(raw_path),
            anchored_path.name: _sha256(anchored_path),
            correction_path.name: _sha256(correction_path),
            figure_path.name: _sha256(figure_path),
        },
        "mechanics_rerun": False,
        "micromorphic_identification_run": False,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
