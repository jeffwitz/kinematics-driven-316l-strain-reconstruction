"""Raw-field validation for local and coupled nonlocal partition campaigns."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.data_preparation import fingerprint_file
from fem_inhouse.partitioning import PartitionLayout
from fem_inhouse.partitioning.stitch import extract_partition_field
from fem_inhouse.postprocessing.metrics import (
    absolute_threshold_overlap_metrics,
    field_error_metrics,
    localization_overlap_metrics,
)
from fem_inhouse.postprocessing.tensor_measures import von_mises_from_stress_tensor
from fem_inhouse.workflows.nonlocality_diagnostic import reconstruct_historical_evm

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CoupledValidationThresholds:
    """Pre-registered acceptance thresholds for the P154 coupling campaign."""

    minimum_correlation_gain: float = 0.05
    minimum_relative_l2_reduction: float = 0.05
    minimum_top10_iou_gain: float = 0.02
    minimum_dic_q90_iou_gain: float = 0.02
    minimum_dic_q90_active_fraction: float = 0.05
    maximum_dic_q90_active_fraction: float = 0.20
    maximum_displacement_error_degradation: float = 0.05
    plane_stress_relative_tolerance: float = 1.0e-6


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _partition_from_manifest(
    manifest: dict[str, Any],
    partition_id: int,
) -> tuple[PartitionLayout, Any]:
    layout_data = manifest.get("layout")
    if not isinstance(layout_data, dict):
        raise ValueError("campaign manifest lacks layout metadata")
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
    declared = [
        item
        for item in layout_data.get("partitions", [])
        if int(item.get("partition_id", -1)) == partition_id
    ]
    if len(declared) != 1:
        raise ValueError(f"campaign manifest does not identify partition {partition_id}")
    if (
        tuple(declared[0]["core_bounds"]) != partition.core_bounds
        or tuple(declared[0]["solve_bounds"]) != partition.solve_bounds
    ):
        raise ValueError("partition bounds disagree with the declared layout")
    return layout, partition


def _load_verified_field(
    campaign: Path,
    *,
    partition_id: int,
    status: dict[str, Any],
    name: str,
) -> FloatArray:
    path = campaign / "partitions" / f"{partition_id:04d}" / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"missing saved partition field {name}: {path}")
    expected_hash = status.get("outputs", {}).get(name)
    if expected_hash is None:
        raise ValueError(f"partition status does not declare output {name}")
    if fingerprint_file(path) != expected_hash:
        raise RuntimeError(f"saved partition field fails its status hash: {path}")
    values = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"saved partition field {name} contains non-finite values")
    return values


def _metric_record(reference: FloatArray, prediction: FloatArray) -> dict[str, float | int]:
    errors = field_error_metrics(reference, prediction)
    top10 = localization_overlap_metrics(reference, prediction, top_fraction=0.10)
    q90 = absolute_threshold_overlap_metrics(
        reference,
        prediction,
        reference_quantile=0.90,
    )
    return {
        **asdict(errors),
        "top10_iou": top10.intersection_over_union,
        "top10_dice": top10.dice_coefficient,
        "top10_precision": top10.prediction_precision,
        "top10_recall": top10.reference_recall,
        "dic_q90_threshold": q90.absolute_threshold,
        "dic_q90_iou": q90.intersection_over_union,
        "dic_q90_precision": q90.prediction_precision,
        "dic_q90_recall": q90.reference_recall,
        "dic_q90_reference_active_fraction": q90.reference_active_fraction,
        "dic_q90_prediction_active_fraction": q90.prediction_active_fraction,
    }


def _relative_change(new: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if new == 0.0 else float("inf")
    return (new - reference) / reference


def _validate_campaign_pair(
    local_manifest: dict[str, Any],
    coupled_manifest: dict[str, Any],
) -> None:
    if local_manifest.get("inputs") != coupled_manifest.get("inputs"):
        raise ValueError("local and coupled campaigns do not use identical input fields")
    if local_manifest.get("layout") != coupled_manifest.get("layout"):
        raise ValueError("local and coupled campaigns do not use the same partition layout")
    local_config = local_manifest["config"]
    coupled_config = coupled_manifest["config"]
    for section in ("mesh", "material"):
        if local_config.get(section) != coupled_config.get(section):
            raise ValueError(f"local and coupled campaigns differ in {section} configuration")
    local_solver = dict(local_config["solver"])
    coupled_solver = dict(coupled_config["solver"])
    local_solver.pop("mfront_threads", None)
    coupled_solver.pop("mfront_threads", None)
    if local_solver != coupled_solver:
        raise ValueError("local and coupled campaigns differ in mechanical solver configuration")
    if bool(local_config.get("nonlocal_plasticity", {}).get("enabled", False)):
        raise ValueError("the reference campaign must use the local constitutive model")
    if not bool(coupled_config.get("nonlocal_plasticity", {}).get("enabled", False)):
        raise ValueError("the candidate campaign must enable nonlocal plasticity")


def validate_coupled_nonlocal_campaign(
    *,
    input_directory: str | Path,
    local_campaign_directory: str | Path,
    coupled_campaign_directory: str | Path,
    partition_id: int,
    output_path: str | Path,
    thresholds: CoupledValidationThresholds | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compare raw local and coupled fields with DIC on one partition core."""

    inputs = Path(input_directory)
    local_campaign = Path(local_campaign_directory)
    coupled_campaign = Path(coupled_campaign_directory)
    destination = Path(output_path)
    thresholds = thresholds or CoupledValidationThresholds()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing report: {destination}")

    local_manifest_path = local_campaign / "manifest.json"
    coupled_manifest_path = coupled_campaign / "manifest.json"
    local_manifest = _load_json(local_manifest_path)
    coupled_manifest = _load_json(coupled_manifest_path)
    _validate_campaign_pair(local_manifest, coupled_manifest)
    layout, partition = _partition_from_manifest(local_manifest, partition_id)

    statuses: dict[str, dict[str, Any]] = {}
    for label, campaign in (
        ("local", local_campaign),
        ("coupled", coupled_campaign),
    ):
        status_path = campaign / "partitions" / f"{partition_id:04d}" / "status.json"
        status = _load_json(status_path)
        if not status.get("complete"):
            raise RuntimeError(f"{label} partition is not complete: {status_path}")
        statuses[label] = status

    local_u = _load_verified_field(
        local_campaign,
        partition_id=partition_id,
        status=statuses["local"],
        name="U",
    )
    coupled_u = _load_verified_field(
        coupled_campaign,
        partition_id=partition_id,
        status=statuses["coupled"],
        name="U",
    )
    expected_u_shape = (
        partition.solve_shape[0] + 1,
        partition.solve_shape[1] + 1,
        2,
    )
    if local_u.shape != expected_u_shape or coupled_u.shape != expected_u_shape:
        raise ValueError(f"saved U fields must have shape {expected_u_shape}")

    global_x = np.load(
        inputs / "displacement_x_mm.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    global_y = np.load(
        inputs / "displacement_y_mm.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    dic_u = np.stack(
        (
            extract_partition_field(
                global_x,
                layout=layout,
                partition=partition,
                location="node",
            ),
            extract_partition_field(
                global_y,
                layout=layout,
                partition=partition,
                location="node",
            ),
        ),
        axis=-1,
    )
    if not np.isfinite(dic_u).all():
        raise ValueError("DIC displacement contains non-finite values")

    mesh = local_manifest["config"]["mesh"]
    material = local_manifest["config"]["material"]
    spacing_mm = float(mesh["base_pixel_size_mm"]) * float(mesh["scale_factor"])
    poisson_ratio = float(material["poisson_ratio"])
    dic_evm = reconstruct_historical_evm(
        dic_u,
        spacing_x_mm=spacing_mm,
        spacing_y_mm=spacing_mm,
        poisson_ratio=poisson_ratio,
    )
    local_evm = reconstruct_historical_evm(
        local_u,
        spacing_x_mm=spacing_mm,
        spacing_y_mm=spacing_mm,
        poisson_ratio=poisson_ratio,
    )
    coupled_evm = reconstruct_historical_evm(
        coupled_u,
        spacing_x_mm=spacing_mm,
        spacing_y_mm=spacing_mm,
        poisson_ratio=poisson_ratio,
    )
    core = partition.core_element_slice_local
    dic_evm_core = dic_evm[core]
    local_evm_core = local_evm[core]
    coupled_evm_core = coupled_evm[core]
    local_metrics = _metric_record(dic_evm_core, local_evm_core)
    coupled_metrics = _metric_record(dic_evm_core, coupled_evm_core)

    core_x, core_y = core
    core_nodes = (
        slice(core_x.start, core_x.stop + 1),
        slice(core_y.start, core_y.stop + 1),
    )
    local_u_metrics = field_error_metrics(dic_u[core_nodes], local_u[core_nodes])
    coupled_u_metrics = field_error_metrics(dic_u[core_nodes], coupled_u[core_nodes])
    displacement_degradation = _relative_change(
        coupled_u_metrics.relative_l2_error,
        local_u_metrics.relative_l2_error,
    )

    stress_3d = _load_verified_field(
        coupled_campaign,
        partition_id=partition_id,
        status=statuses["coupled"],
        name="S_3D",
    )
    plane_stress_residual = _load_verified_field(
        coupled_campaign,
        partition_id=partition_id,
        status=statuses["coupled"],
        name="PLANE_STRESS_RESIDUAL_MPA",
    )
    vm_core = von_mises_from_stress_tensor(stress_3d[core])
    residual_core = plane_stress_residual[core]
    maximum_plane_stress_residual_mpa = float(np.max(np.abs(residual_core)))
    plane_stress_limit_mpa = thresholds.plane_stress_relative_tolerance * max(
        1.0,
        float(np.max(vm_core)),
    )

    for name in (
        "PEEQ",
        "PEEQ_NONLOCAL",
        "PEEQ_MISMATCH",
        "NONLOCAL_HARDENING_MPA",
        "YIELD_SURFACE_RADIUS_MPA",
        "NONLOCAL_RESIDUAL",
    ):
        _load_verified_field(
            coupled_campaign,
            partition_id=partition_id,
            status=statuses["coupled"],
            name=name,
        )

    gains = {
        "correlation": (
            float(coupled_metrics["pearson_correlation"])
            - float(local_metrics["pearson_correlation"])
        ),
        "relative_l2_reduction": (
            float(local_metrics["relative_l2_error"])
            - float(coupled_metrics["relative_l2_error"])
        )
        / float(local_metrics["relative_l2_error"]),
        "top10_iou": float(coupled_metrics["top10_iou"])
        - float(local_metrics["top10_iou"]),
        "dic_q90_iou": float(coupled_metrics["dic_q90_iou"])
        - float(local_metrics["dic_q90_iou"]),
        "displacement_relative_l2_degradation": displacement_degradation,
    }
    checks = {
        "minimum_correlation_gain": gains["correlation"]
        >= thresholds.minimum_correlation_gain,
        "minimum_relative_l2_reduction": gains["relative_l2_reduction"]
        >= thresholds.minimum_relative_l2_reduction,
        "minimum_top10_iou_gain": gains["top10_iou"]
        >= thresholds.minimum_top10_iou_gain,
        "minimum_dic_q90_iou_gain": gains["dic_q90_iou"]
        >= thresholds.minimum_dic_q90_iou_gain,
        "dic_q90_active_fraction": (
            thresholds.minimum_dic_q90_active_fraction
            <= float(coupled_metrics["dic_q90_prediction_active_fraction"])
            <= thresholds.maximum_dic_q90_active_fraction
        ),
        "maximum_displacement_error_degradation": displacement_degradation
        <= thresholds.maximum_displacement_error_degradation,
        "plane_stress": maximum_plane_stress_residual_mpa <= plane_stress_limit_mpa,
        "finite_required_fields": True,
    }
    coupled_config = coupled_manifest["config"]["nonlocal_plasticity"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "partition_id": partition_id,
        "scientific_question": (
            "Does energetic micromorphic coupling improve raw FEM-DIC agreement "
            "without post-filtering the computed solution?"
        ),
        "comparison_contract": {
            "observable": "EVM_HISTORICAL",
            "construction": [
                "strain_from_displacement",
                "plane_stress_equivalent_strain(engineering shear)",
                "cell_average",
            ],
            "metrics_domain": "partition core only",
            "post_filter_applied": False,
            "mechanical_solution_modified_by_candidate": True,
        },
        "nonlocal_parameters": coupled_config,
        "thresholds": asdict(thresholds),
        "metrics": {
            "local": local_metrics,
            "coupled": coupled_metrics,
            "local_displacement": asdict(local_u_metrics),
            "coupled_displacement": asdict(coupled_u_metrics),
        },
        "gains": gains,
        "mechanical_checks": {
            "maximum_plane_stress_residual_mpa": maximum_plane_stress_residual_mpa,
            "plane_stress_limit_mpa": plane_stress_limit_mpa,
            "maximum_von_mises_mpa": float(np.max(vm_core)),
        },
        "solver_diagnostics": {
            "local": statuses["local"].get("diagnostics", {}),
            "coupled": statuses["coupled"].get("diagnostics", {}),
        },
        "acceptance_checks": checks,
        "passed": all(checks.values()),
        "inputs": {
            "prepared_case": str(inputs),
            "local_campaign": str(local_campaign),
            "coupled_campaign": str(coupled_campaign),
            "local_manifest_sha256": fingerprint_file(local_manifest_path),
            "coupled_manifest_sha256": fingerprint_file(coupled_manifest_path),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return report
