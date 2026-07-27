"""Validate homogeneous and spatially translated material-map controls."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from fem_inhouse.data_preparation import fingerprint_file
from fem_inhouse.partitioning.stitch import extract_partition_field
from fem_inhouse.postprocessing import (
    absolute_threshold_overlap_metrics,
    field_error_metrics,
    localization_overlap_metrics,
)
from fem_inhouse.workflows.campaign_access import (
    load_json_object,
    load_partition_status,
    load_verified_partition_field,
    partition_from_manifest,
)
from fem_inhouse.workflows.nonlocality_diagnostic import reconstruct_historical_evm


def _verified_input(path: Path, name: str) -> np.ndarray:
    manifest = load_json_object(path / "manifest.json")
    declaration = manifest.get("outputs", {}).get(name)
    source = path / f"{name}.npy"
    if not isinstance(declaration, dict) or fingerprint_file(source) != declaration.get("sha256"):
        raise RuntimeError(f"canonical input fails its manifest hash: {source}")
    values = np.load(source, mmap_mode="r", allow_pickle=False)
    if not np.isfinite(values).all():
        raise ValueError(f"canonical input contains non-finite values: {source}")
    return values


def validate_material_map_controls(
    *,
    input_directory: str | Path,
    campaigns: Sequence[tuple[str, str | Path]],
    partition_id: int,
    output_directory: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compare mapped, homogeneous and translated-map local campaigns with DIC."""

    if len(campaigns) != 3 or [label for label, _path in campaigns] != [
        "mapped",
        "homogeneous",
        "translated",
    ]:
        raise ValueError("campaigns must be ordered as mapped, homogeneous, translated")
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    paths = tuple(Path(path) for _label, path in campaigns)
    manifests = tuple(load_json_object(path / "manifest.json") for path in paths)
    layout, partition = partition_from_manifest(manifests[0], partition_id)
    reference_layout = manifests[0]["layout"]
    reference_mesh = manifests[0]["config"]["mesh"]
    reference_material = manifests[0]["config"]["material"]
    reference_solver = manifests[0]["config"]["solver"]
    for manifest in manifests[1:]:
        if manifest["layout"] != reference_layout:
            raise ValueError("control campaigns do not use the same partition layout")
        if (
            manifest["config"]["mesh"] != reference_mesh
            or manifest["config"]["material"] != reference_material
            or manifest["config"]["solver"] != reference_solver
        ):
            raise ValueError("control campaigns do not use identical mechanical settings")

    spacing = float(reference_mesh["base_pixel_size_mm"]) * float(
        reference_mesh["scale_factor"]
    )
    poisson = float(reference_material["poisson_ratio"])
    inputs = Path(input_directory)
    dic_x = _verified_input(inputs, "displacement_x_mm")
    dic_y = _verified_input(inputs, "displacement_y_mm")
    dic_u = np.stack(
        (
            extract_partition_field(dic_x, layout=layout, partition=partition, location="node"),
            extract_partition_field(dic_y, layout=layout, partition=partition, location="node"),
        ),
        axis=-1,
    )
    core = partition.core_element_slice_local
    dic_evm = reconstruct_historical_evm(
        dic_u,
        spacing_x_mm=spacing,
        spacing_y_mm=spacing,
        poisson_ratio=poisson,
    )[core]

    rows: list[dict[str, Any]] = []
    evm_fields: list[np.ndarray] = []
    peeq_fields: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    for (label, _campaign_value), path in zip(campaigns, paths, strict=True):
        manifest_hash = fingerprint_file(path / "manifest.json")
        status = load_partition_status(
            path,
            partition_id=partition_id,
            manifest_sha256=manifest_hash,
        )
        displacement = load_verified_partition_field(
            path, partition_id=partition_id, status=status, name="U"
        )
        peeq = load_verified_partition_field(
            path, partition_id=partition_id, status=status, name="PEEQ"
        )[core]
        evm = reconstruct_historical_evm(
            displacement,
            spacing_x_mm=spacing,
            spacing_y_mm=spacing,
            poisson_ratio=poisson,
        )[core]
        error = field_error_metrics(dic_evm, evm)
        relative = localization_overlap_metrics(dic_evm, evm, top_fraction=0.1)
        absolute = absolute_threshold_overlap_metrics(
            dic_evm,
            evm,
            reference_quantile=0.9,
        )
        rows.append(
            {
                "label": label,
                "error": asdict(error),
                "relative_top10": asdict(relative),
                "absolute_dic_q90": asdict(absolute),
                "peeq": {
                    "mean": float(np.mean(peeq)),
                    "standard_deviation": float(np.std(peeq)),
                    "maximum": float(np.max(peeq)),
                    "plastic_fraction": float(np.mean(peeq > 1.0e-6)),
                },
                "convergence": status["diagnostics"],
            }
        )
        evm_fields.append(np.asarray(evm))
        peeq_fields.append(np.asarray(peeq))
        sources.append(
            {
                "label": label,
                "campaign": str(path),
                "manifest_sha256": manifest_hash,
                "U_sha256": status["outputs"]["U"],
                "PEEQ_sha256": status["outputs"]["PEEQ"],
            }
        )

    all_evm = np.concatenate([dic_evm.ravel(), *(field.ravel() for field in evm_fields)])
    all_peeq = np.concatenate([field.ravel() for field in peeq_fields])
    core_x0, core_x1, core_y0, core_y1 = partition.core_bounds
    extent = (
        core_x0 * spacing,
        core_x1 * spacing,
        core_y0 * spacing,
        core_y1 * spacing,
    )
    figure, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
    evm_images = []
    for axis, title, field in zip(
        axes[0],
        ("DIC EVM", "Mapped local", "Homogeneous", "Translated maps"),
        (dic_evm, *evm_fields),
        strict=True,
    ):
        image = axis.imshow(
            field.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            vmin=0.0,
            vmax=float(np.max(all_evm)),
            cmap="viridis",
        )
        evm_images.append(image)
        axis.set_title(title)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    figure.colorbar(evm_images[0], ax=axes[0].tolist(), label="Total equivalent strain, EVM")
    axes[1, 0].axis("off")
    peeq_images = []
    for axis, title, field in zip(
        axes[1, 1:],
        ("Mapped PEEQ", "Homogeneous PEEQ", "Translated-map PEEQ"),
        peeq_fields,
        strict=True,
    ):
        image = axis.imshow(
            field.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            vmin=0.0,
            vmax=float(np.max(all_peeq)),
            cmap="magma",
        )
        peeq_images.append(image)
        axis.set_title(title)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    figure.colorbar(peeq_images[0], ax=axes[1, 1:].tolist(), label="PEEQ (model output)")
    figure.savefig(output / "material_map_controls.png", dpi=180)
    figure.savefig(output / "material_map_controls.pdf")
    plt.close(figure)

    report = {
        "schema_version": 1,
        "status": "completed_no_acceptance_threshold",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "partition_id": partition_id,
        "core_bounds": list(partition.core_bounds),
        "spacing_mm": spacing,
        "evm_post_filter_applied": False,
        "observation": "historical EVM reconstructed from DIC and FEM nodal displacement",
        "sources": sources,
        "rows": rows,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
