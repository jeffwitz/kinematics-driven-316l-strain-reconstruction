"""Pre-analysis utilities for the coupled micromorphic P154 campaign."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.workflows.partitioned import fingerprint_array

FloatArray = NDArray[np.float64]
DEFAULT_ALPHA_VALUES = (0.0, 0.25, 0.5, 1.0, 2.0)


@dataclass(frozen=True, slots=True)
class ReferenceHardeningReport:
    """Reference tangent hardening modulus evaluated on a local plastic core."""

    partition_id: int
    core_element_count: int
    plastic_element_count: int
    plastic_element_fraction: float
    first_positive_plastic_strain: float
    hardening_exponent: float
    reference_hardening_modulus_mpa: float
    derivative_minimum_mpa: float
    derivative_maximum_mpa: float
    derivative_q25_mpa: float
    derivative_q75_mpa: float
    alpha_values: tuple[float, ...]
    coupling_moduli_mpa: tuple[float, ...]
    source_campaign_manifest_sha256: str
    source_peeq_sha256: str
    source_hardening_map_sha256: str


def compute_reference_hardening_modulus(
    equivalent_plastic_strain: ArrayLike,
    hardening_coefficient_mpa: ArrayLike,
    *,
    core_slice: tuple[slice, slice],
    hardening_exponent: float,
    first_positive_plastic_strain: float,
) -> tuple[float, FloatArray]:
    """Return median ``K n p**(n-1)`` and samples on plastic core elements."""

    peeq = np.asarray(equivalent_plastic_strain, dtype=np.float64)
    hardening = np.asarray(hardening_coefficient_mpa, dtype=np.float64)
    if peeq.ndim != 2 or hardening.shape != peeq.shape:
        raise ValueError("PEEQ and K must be finite 2D fields with identical shapes")
    if not np.isfinite(peeq).all() or not np.isfinite(hardening).all():
        raise ValueError("PEEQ and K must be finite 2D fields with identical shapes")
    if np.any(peeq < 0) or np.any(hardening < 0):
        raise ValueError("PEEQ and K must be nonnegative")
    if not 0 < hardening_exponent < 1:
        raise ValueError("hardening_exponent must lie in (0, 1)")
    if first_positive_plastic_strain <= 0:
        raise ValueError("first_positive_plastic_strain must be positive")

    core_peeq = peeq[core_slice]
    core_hardening = hardening[core_slice]
    plastic = core_peeq > first_positive_plastic_strain
    if not np.any(plastic):
        raise ValueError("the selected core contains no elements above the plastic threshold")
    derivative = (
        core_hardening[plastic]
        * hardening_exponent
        * np.power(core_peeq[plastic], hardening_exponent - 1.0)
    )
    if not np.isfinite(derivative).all() or np.any(derivative < 0):
        raise ValueError("computed hardening derivatives are invalid")
    return float(np.median(derivative)), derivative


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _partition_metadata(manifest: dict[str, Any], partition_id: int) -> dict[str, Any]:
    for partition in manifest["layout"]["partitions"]:
        if int(partition["partition_id"]) == partition_id:
            return partition
    raise ValueError(f"partition {partition_id} is absent from the campaign manifest")


def estimate_reference_hardening_from_campaign(
    *,
    input_directory: str | Path,
    campaign_directory: str | Path,
    partition_id: int,
    output_path: str | Path,
    alpha_values: tuple[float, ...] = DEFAULT_ALPHA_VALUES,
    overwrite: bool = False,
) -> ReferenceHardeningReport:
    """Estimate and persist ``H_ref`` from a completed local partition campaign."""

    inputs = Path(input_directory)
    campaign = Path(campaign_directory)
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing report: {destination}")
    manifest_path = campaign / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nonlocal_config = manifest["config"].get("nonlocal_plasticity", {})
    if nonlocal_config.get("enabled", False) and nonlocal_config.get(
        "coupling_modulus_mpa", 0.0
    ) != 0.0:
        raise ValueError("H_ref must be estimated from a local or H_chi=0 campaign")
    partition = _partition_metadata(manifest, partition_id)
    status_path = campaign / "partitions" / f"{partition_id:04d}" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not status.get("complete"):
        raise ValueError(f"partition {partition_id} is not complete")
    peeq_path = campaign / "partitions" / f"{partition_id:04d}" / "PEEQ.npy"
    peeq = np.load(peeq_path, mmap_mode="r", allow_pickle=False)
    hardening_global = np.load(
        inputs / "hardening_coefficient_mpa.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    expected_hardening_hash = manifest["inputs"]["hardening_coefficient_mpa"]
    if fingerprint_array(hardening_global) != expected_hardening_hash:
        raise ValueError("input hardening map does not match the campaign manifest")

    sx0, sx1, sy0, sy1 = (int(value) for value in partition["solve_bounds"])
    cx0, cx1, cy0, cy1 = (int(value) for value in partition["core_bounds"])
    hardening = hardening_global[sx0:sx1, sy0:sy1]
    if hardening.shape != peeq.shape:
        raise ValueError(
            f"saved PEEQ shape {peeq.shape} does not match solve bounds {hardening.shape}"
        )
    core_slice = (slice(cx0 - sx0, cx1 - sx0), slice(cy0 - sy0, cy1 - sy0))
    material = manifest["config"]["material"]
    hardening_exponent = float(material["hardening_exponent"])
    plastic_threshold = float(material["first_positive_plastic_strain"])
    reference, derivative = compute_reference_hardening_modulus(
        peeq,
        hardening,
        core_slice=core_slice,
        hardening_exponent=hardening_exponent,
        first_positive_plastic_strain=plastic_threshold,
    )
    alphas = tuple(sorted(set(float(value) for value in alpha_values)))
    if not alphas or not np.isfinite(alphas).all() or min(alphas) < 0:
        raise ValueError("alpha_values must contain finite nonnegative values")
    plastic_count = int(derivative.size)
    core_count = int((cx1 - cx0) * (cy1 - cy0))
    report = ReferenceHardeningReport(
        partition_id=partition_id,
        core_element_count=core_count,
        plastic_element_count=plastic_count,
        plastic_element_fraction=plastic_count / core_count,
        first_positive_plastic_strain=plastic_threshold,
        hardening_exponent=hardening_exponent,
        reference_hardening_modulus_mpa=reference,
        derivative_minimum_mpa=float(np.min(derivative)),
        derivative_maximum_mpa=float(np.max(derivative)),
        derivative_q25_mpa=float(np.quantile(derivative, 0.25)),
        derivative_q75_mpa=float(np.quantile(derivative, 0.75)),
        alpha_values=alphas,
        coupling_moduli_mpa=tuple(reference * alpha for alpha in alphas),
        source_campaign_manifest_sha256=_file_sha256(manifest_path),
        source_peeq_sha256=_file_sha256(peeq_path),
        source_hardening_map_sha256=_file_sha256(
            inputs / "hardening_coefficient_mpa.npy"
        ),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return report
