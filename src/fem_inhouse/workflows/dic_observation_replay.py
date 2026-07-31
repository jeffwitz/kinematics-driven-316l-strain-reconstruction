"""Replay archived FEM displacements through the declared image/DISFlow chain."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from fem_inhouse.identification.observation import DICObservationOperatorConfig
from fem_inhouse.measurement import (
    canonical_to_image_flow,
    declared_all_valid_mask,
    disflow_profile,
    image_flow_to_canonical,
    require_native_finest_scale,
    run_disflow,
    warp_forward_displacement,
)
from fem_inhouse.measurement.warp import WARP_BORDER_MODE, WARP_INTERPOLATION
from fem_inhouse.postprocessing.metrics import (
    absolute_threshold_overlap_metrics,
    field_diffusivity_metrics,
    field_error_metrics,
    localization_overlap_metrics,
)
from fem_inhouse.workflows.nonlocality_diagnostic import reconstruct_historical_evm

FloatArray = NDArray[np.float64]
PIXEL_SIZE_MM = 0.00184
RAW_CROP_ROW_START = 400
RAW_CROP_COLUMN_START = 1211


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _opencv_version() -> str:
    try:
        import cv2
    except ImportError:  # pragma: no cover - measurement extra is optional
        return "unavailable"
    return str(cv2.__version__)


def _repository_state() -> dict[str, Any]:
    """Record whether the working tree was clean when the replay ran."""

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True,
    ).strip()
    return {"git_sha": _git_sha(), "clean": not dirty}


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _partition(manifest: dict[str, Any], partition_id: int) -> dict[str, Any]:
    matches = [
        item
        for item in manifest["layout"]["partitions"]
        if int(item["partition_id"]) == partition_id
    ]
    if len(matches) != 1:
        raise ValueError(f"partition {partition_id} is not uniquely declared")
    return matches[0]


def _core_slice(partition: dict[str, Any]) -> tuple[slice, slice]:
    cx0, cx1, cy0, cy1 = (int(value) for value in partition["core_bounds"])
    sx0, _, sy0, _ = (int(value) for value in partition["solve_bounds"])
    return slice(cx0 - sx0, cx1 - sx0), slice(cy0 - sy0, cy1 - sy0)


def _statistics(field: FloatArray) -> dict[str, float]:
    return {
        "mean": float(np.mean(field)),
        "standard_deviation": float(np.std(field)),
        "minimum": float(np.min(field)),
        "q50": float(np.quantile(field, 0.50)),
        "q90": float(np.quantile(field, 0.90)),
        "q95": float(np.quantile(field, 0.95)),
        "q99": float(np.quantile(field, 0.99)),
        "maximum": float(np.max(field)),
    }


def _comparison(reference: FloatArray, prediction: FloatArray) -> dict[str, Any]:
    raw = np.asarray(prediction, dtype=np.float64)
    return {
        "errors": asdict(field_error_metrics(reference, raw)),
        "top10": asdict(localization_overlap_metrics(reference, raw, top_fraction=0.1)),
        "absolute_q90": asdict(
            absolute_threshold_overlap_metrics(
                reference,
                raw,
                reference_quantile=0.9,
            )
        ),
        "statistics": _statistics(raw),
        "diffusivity": asdict(
            field_diffusivity_metrics(
                raw,
                raw_field=raw,
                spacing_x_mm=PIXEL_SIZE_MM,
                spacing_y_mm=PIXEL_SIZE_MM,
            )
        ),
    }


def _figure(
    path: Path,
    *,
    dic: FloatArray,
    raw: FloatArray,
    observed: FloatArray,
    profile_name: str,
) -> None:
    fields = (dic, raw, observed)
    vmax = max(float(np.max(field)) for field in fields)
    errors = (raw - dic, observed - dic)
    error_limit = max(float(np.max(np.abs(field))) for field in errors)
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    titles = ("DIC EVM", "FEM raw EVM", f"FEM observed EVM\n{profile_name}")
    for axis, field, title in zip(axes[0], fields, titles, strict=True):
        image = axis.imshow(field.T, origin="upper", cmap="magma", vmin=0.0, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("x element")
        axis.set_ylabel("y element")
    figure.colorbar(image, ax=axes[0], label="Equivalent total strain")
    for axis, field, title in zip(
        axes[1, :2],
        errors,
        ("Raw FEM - DIC", "Observed FEM - DIC"),
        strict=True,
    ):
        difference = axis.imshow(
            field.T,
            origin="upper",
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit,
        )
        axis.set_title(title)
        axis.set_xlabel("x element")
        axis.set_ylabel("y element")
    figure.colorbar(difference, ax=axes[1, :2], label="Signed EVM error")
    coordinate = np.arange(dic.shape[0])
    y_index = dic.shape[1] // 2
    axes[1, 2].plot(coordinate, dic[:, y_index], label="DIC")
    axes[1, 2].plot(coordinate, raw[:, y_index], label="FEM raw")
    axes[1, 2].plot(coordinate, observed[:, y_index], label="FEM observed")
    axes[1, 2].set_title("Fixed central section")
    axes[1, 2].set_xlabel("x element")
    axes[1, 2].set_ylabel("EVM")
    axes[1, 2].legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def replay_dic_observation(
    *,
    campaign: str | Path,
    prepared_case: str | Path,
    reference_image: str | Path,
    partition_id: int,
    profile_name: str,
    output_directory: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Observe one immutable archived FEM displacement through DISFlow."""

    campaign_path = Path(campaign)
    prepared_path = Path(prepared_case)
    image_path = Path(reference_image)
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = campaign_path / "manifest.json"
    status_path = campaign_path / "partitions" / f"{partition_id:04d}" / "status.json"
    displacement_path = campaign_path / "partitions" / f"{partition_id:04d}" / "U.npy"
    manifest = _json(manifest_path)
    status = _json(status_path)
    if not status.get("complete"):
        raise ValueError("source campaign partition is incomplete")
    if _sha256(displacement_path) != status["outputs"]["U"]:
        raise ValueError("source U.npy does not match the immutable campaign status")
    partition = _partition(manifest, partition_id)
    solve_x0, solve_x1, solve_y0, solve_y1 = (
        int(value) for value in partition["solve_bounds"]
    )
    displacement = np.load(displacement_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (solve_x1 - solve_x0 + 1, solve_y1 - solve_y0 + 1, 2)
    if displacement.shape != expected_shape or not np.isfinite(displacement).all():
        raise ValueError(f"source displacement has incompatible shape; expected {expected_shape}")

    reference_full = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    reference = np.ascontiguousarray(
        reference_full[
            RAW_CROP_ROW_START + solve_x0 : RAW_CROP_ROW_START + solve_x1 + 1,
            RAW_CROP_COLUMN_START + solve_y0 : RAW_CROP_COLUMN_START + solve_y1 + 1,
        ]
    )
    if reference.shape != displacement.shape[:2]:
        raise ValueError("reference-image crop and FEM nodal support are incompatible")

    declared_mask = declared_all_valid_mask(reference.shape)
    flow_imposed = canonical_to_image_flow(displacement, pixel_size_mm=PIXEL_SIZE_MM)
    warp = warp_forward_displacement(
        reference,
        flow_imposed,
        mode="iterative_forward_inverse",
    )
    profile = disflow_profile(profile_name)
    # The symmetric replay is metrological use: a coarse finest scale skips
    # full-resolution refinement and silently changes the measured transfer.
    require_native_finest_scale(profile.config)
    flow_observed = run_disflow(reference, warp.image, config=profile.config)
    displacement_observed = image_flow_to_canonical(
        flow_observed,
        pixel_size_mm=PIXEL_SIZE_MM,
    )

    prepared_ux = np.load(
        prepared_path / "displacement_x_mm.npy", mmap_mode="r", allow_pickle=False
    )
    prepared_uy = np.load(
        prepared_path / "displacement_y_mm.npy", mmap_mode="r", allow_pickle=False
    )
    dic_displacement = np.stack(
        (
            prepared_ux[solve_x0 : solve_x1 + 1, solve_y0 : solve_y1 + 1],
            prepared_uy[solve_x0 : solve_x1 + 1, solve_y0 : solve_y1 + 1],
        ),
        axis=-1,
    )
    core = _core_slice(partition)
    dic_evm = reconstruct_historical_evm(
        dic_displacement,
        spacing_x_mm=PIXEL_SIZE_MM,
        spacing_y_mm=PIXEL_SIZE_MM,
        poisson_ratio=0.3,
    )[core]
    raw_evm = reconstruct_historical_evm(
        displacement,
        spacing_x_mm=PIXEL_SIZE_MM,
        spacing_y_mm=PIXEL_SIZE_MM,
        poisson_ratio=0.3,
    )[core]
    observed_evm = reconstruct_historical_evm(
        displacement_observed,
        spacing_x_mm=PIXEL_SIZE_MM,
        spacing_y_mm=PIXEL_SIZE_MM,
        poisson_ratio=0.3,
    )[core]
    if not (dic_evm.shape == raw_evm.shape == observed_evm.shape):
        raise ValueError("DIC, raw FEM and observed FEM core supports differ")

    config = DICObservationOperatorConfig(
        mode="synthetic_disflow",
        disflow_profile=profile_name,
        warp_mode="iterative_forward_inverse",
        mask_mode="declared_all_valid",
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "status": "completed_symmetric_image_observation",
        "partition_id": partition_id,
        "core_bounds": partition["core_bounds"],
        "solve_bounds": partition["solve_bounds"],
        "profile": profile.manifest(),
        "observation_operator": config.as_dict(),
        "observation_operator_sha256": config.fingerprint(),
        "warp": {
            "mode": warp.mode,
            "iterations": warp.iterations,
            "residual_pixels": warp.residual_pixels,
            "minimum_forward_jacobian": warp.minimum_forward_jacobian,
            "interpolation": WARP_INTERPOLATION,
            "border_mode": WARP_BORDER_MODE,
        },
        "mask": {
            "mode": "declared_all_valid",
            "dtype": str(declared_mask.dtype),
            "unique_values": [True],
            "sha256": hashlib.sha256(declared_mask.tobytes()).hexdigest(),
        },
        "provenance": {
            "campaign_manifest": str(manifest_path.resolve()),
            "campaign_manifest_sha256": _sha256(manifest_path),
            "campaign_status_sha256": _sha256(status_path),
            "source_displacement_sha256": _sha256(displacement_path),
            "reference_image": str(image_path.resolve()),
            "reference_image_sha256": _sha256(image_path),
            "prepared_case_manifest_sha256": _sha256(prepared_path / "manifest.json"),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": _opencv_version(),
        },
        "repository_state": _repository_state(),
        "axis_convention": (
            "image row -> canonical x/ux via drow; "
            "image column -> canonical y/uy via dcolumn"
        ),
        "pixel_size_mm": PIXEL_SIZE_MM,
        "grid_contract": {
            "fem_support": "nodal",
            "image_crop_shape": list(reference.shape),
            "fem_nodal_shape": list(displacement.shape[:2]),
            "interpolation_to_image_grid": "identity (one node is one pixel)",
            "evm_support": "element-centred",
            "note": (
                "the nodal and element-centred lattices differ by half a pixel; "
                "identical for DIC and FEM, so it cancels in this comparison"
            ),
        },
        "evm_operator": "reconstruct_historical_evm",
        "evm_differentiation": {
            "scheme": "historical plane-stress EVM from nodal displacement",
            "poisson_ratio": 0.3,
            "spacing_x_mm": PIXEL_SIZE_MM,
            "spacing_y_mm": PIXEL_SIZE_MM,
        },
        "evm_post_filter_applied": False,
        "metrics": {
            "raw": _comparison(dic_evm, raw_evm),
            "observed": _comparison(dic_evm, observed_evm),
            "dic_statistics": _statistics(dic_evm),
        },
    }
    np.save(output / "dic_evm.npy", dic_evm)
    np.save(output / "fem_raw_evm.npy", raw_evm)
    np.save(output / "fem_observed_evm.npy", observed_evm)
    np.save(output / "observed_flow_pixels.npy", flow_observed)
    # Audit artefacts required by the observed-EVM comparison specification.
    # The names above are kept because archived reports hash them; these are
    # additions, not renames.
    np.save(output / "fem_displacement_image_grid.npy", flow_imposed)
    np.save(output / "recovered_displacement_column.npy", flow_observed[..., 0])
    np.save(output / "recovered_displacement_row.npy", flow_observed[..., 1])
    np.save(output / "valid_mask.npy", declared_mask)
    Image.fromarray(warp.image).save(output / "synthetic_deformed_image.tif")
    _figure(
        output / "comparison.png",
        dic=dic_evm,
        raw=raw_evm,
        observed=observed_evm,
        profile_name=profile_name,
    )
    report["outputs"] = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "report.json"
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
