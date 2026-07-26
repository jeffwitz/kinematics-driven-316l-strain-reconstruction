"""Verified, read-only access to saved partition campaigns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.data_preparation import fingerprint_file
from fem_inhouse.partitioning import Partition, PartitionLayout

FloatArray = NDArray[np.float64]


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load one JSON object, rejecting missing files and non-object roots."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing JSON file: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {source}")
    return value


def partition_from_manifest(
    manifest: dict[str, Any],
    partition_id: int,
) -> tuple[PartitionLayout, Partition]:
    """Reconstruct and verify one partition from immutable manifest metadata."""

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


def load_partition_status(
    campaign: str | Path,
    *,
    partition_id: int,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Load a complete status whose manifest digest matches the campaign."""

    campaign_path = Path(campaign)
    status_path = campaign_path / "partitions" / f"{partition_id:04d}" / "status.json"
    status = load_json_object(status_path)
    if not bool(status.get("complete", False)):
        raise RuntimeError(f"partition is not complete: {status_path}")
    if int(status.get("partition_id", -1)) != partition_id:
        raise ValueError(f"partition status identifies another partition: {status_path}")
    if status.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError(f"partition status does not match campaign manifest: {status_path}")
    return status


def load_verified_partition_field(
    campaign: str | Path,
    *,
    partition_id: int,
    status: dict[str, Any],
    name: str,
    mmap_mode: Literal["r+", "r", "w+", "c"] | None = "r",
) -> FloatArray:
    """Load one finite field after checking its hash against partition status."""

    campaign_path = Path(campaign)
    path = campaign_path / "partitions" / f"{partition_id:04d}" / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"missing saved partition field {name}: {path}")
    expected_hash = status.get("outputs", {}).get(name)
    if expected_hash is None:
        raise ValueError(f"partition status does not declare output {name}")
    if fingerprint_file(path) != expected_hash:
        raise RuntimeError(f"saved partition field fails its status hash: {path}")
    values = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"saved partition field {name} is not numeric")
    if not np.isfinite(values).all():
        raise ValueError(f"saved partition field {name} contains non-finite values")
    return np.asarray(values, dtype=np.float64)


def validate_mechanical_campaign_pair(
    local_manifest: dict[str, Any],
    coupled_manifest: dict[str, Any],
) -> None:
    """Require identical mechanics apart from the nonlocal configuration."""

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
    local_nonlocal = local_config.get("nonlocal_plasticity", {})
    if bool(local_nonlocal.get("enabled", False)) and float(
        local_nonlocal.get("coupling_modulus_mpa", 0.0)
    ) != 0.0:
        raise ValueError("the reference campaign must be local or use H_chi=0")
    if not bool(coupled_config.get("nonlocal_plasticity", {}).get("enabled", False)):
        raise ValueError("the candidate campaign must enable nonlocal plasticity")
