"""Staged, cache-safe identification of ``ell`` and ``H_chi``."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import scipy
import yaml
from numpy.typing import NDArray
from scipy.stats import spearmanr

from fem_inhouse import __version__
from fem_inhouse.config import (
    CaseStudyConfig,
    MaterialConfig,
    MeshConfig,
    NonlocalPlasticityConfig,
    SolverConfig,
)
from fem_inhouse.data_preparation import fingerprint_file
from fem_inhouse.identification.metrics import (
    AmplitudeMetricConfig,
    evaluate_identification_metrics,
    peeq_diagnostic_metrics,
    radial_power_spectrum,
)
from fem_inhouse.identification.observation import (
    DICObservationOperator,
    DICObservationOperatorConfig,
)
from fem_inhouse.identification.parameters import NonlocalIdentificationPoint
from fem_inhouse.partitioning import PartitionLayout, extract_partition_field
from fem_inhouse.postprocessing.helmholtz import helmholtz_filter_element_field
from fem_inhouse.postprocessing.metrics import field_diffusivity_metrics
from fem_inhouse.workflows.campaign_access import (
    load_json_object,
    load_partition_status,
    load_verified_partition_field,
    partition_from_manifest,
    validate_mechanical_campaign_pair,
)
from fem_inhouse.workflows.partitioned import PartitionWorkflow, fingerprint_array

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ExistingHighFidelityPoint:
    """One already completed F2 campaign and its DIC validation report."""

    alpha: float
    campaign: Path
    validation_report: Path


@dataclass(frozen=True, slots=True)
class JointIdentificationConfig:
    """Validated campaign configuration used by all identification actions."""

    source_path: Path
    name: str
    input_directory: Path
    output_directory: Path
    local_campaign: Path
    partition_id: int
    h_ref_path: Path
    maximum_new_high_fidelity_runs: int
    ell_min_um: float
    ell_max_um: float
    ell_samples: int
    ell_anchors_um: tuple[float, ...]
    alpha_min: float
    alpha_max: float
    alpha_samples: int
    alpha_anchors: tuple[float, ...]
    low_spatial_reduction: int
    low_temporal_increments: int
    low_minimum_elements_per_ell: float
    low_residual_tolerance: float
    existing_high_fidelity: tuple[ExistingHighFidelityPoint, ...]
    observation: DICObservationOperatorConfig
    raw: dict[str, Any]

    @property
    def source_sha256(self) -> str:
        return fingerprint_file(self.source_path)

    def ell_values_um(self) -> tuple[float, ...]:
        values = np.linspace(self.ell_min_um, self.ell_max_um, self.ell_samples)
        return tuple(
            sorted({float(value) for value in values} | set(self.ell_anchors_um))
        )

    def alpha_values(self) -> tuple[float, ...]:
        values = np.linspace(self.alpha_min, self.alpha_max, self.alpha_samples)
        return tuple(sorted({float(value) for value in values} | set(self.alpha_anchors)))


def _repository_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError(f"cannot locate repository root from {config_path}")


def _resolve(root: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_range(
    data: dict[str, Any],
    *,
    name: str,
    minimum_allowed: float,
) -> tuple[float, float]:
    minimum = float(data["min"])
    maximum = float(data["max"])
    if (
        not np.isfinite(minimum)
        or not np.isfinite(maximum)
        or minimum < minimum_allowed
        or maximum < minimum
    ):
        raise ValueError(f"{name} range is invalid")
    return minimum, maximum


def load_joint_identification_config(
    path: str | Path,
) -> JointIdentificationConfig:
    """Load and validate the versioned joint-identification YAML."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"missing identification configuration: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw = _mapping(loaded, name="configuration")
    root = _repository_root(source)
    campaign = _mapping(raw.get("campaign"), name="campaign")
    parameters = _mapping(raw.get("parameters"), name="parameters")
    ell = _mapping(parameters.get("ell_um"), name="parameters.ell_um")
    alpha = _mapping(parameters.get("alpha"), name="parameters.alpha")
    ell_min, ell_max = _finite_range(ell, name="ell_um", minimum_allowed=0.0)
    alpha_min, alpha_max = _finite_range(alpha, name="alpha", minimum_allowed=0.0)
    if alpha_min <= 0.0:
        raise ValueError("positive-alpha exploration must start above zero")
    h_ref_source = parameters.get("h_ref_source")
    if h_ref_source != "campaign_metadata":
        raise ValueError("h_ref_source must be 'campaign_metadata'")

    existing_data = campaign.get("existing_high_fidelity", [])
    if not isinstance(existing_data, list):
        raise ValueError("campaign.existing_high_fidelity must be a list")
    existing = tuple(
        ExistingHighFidelityPoint(
            alpha=float(_mapping(item, name="F2 point")["alpha"]),
            campaign=_resolve(
                root,
                _mapping(item, name="F2 point")["campaign"],
                name="F2 campaign",
            ),
            validation_report=_resolve(
                root,
                _mapping(item, name="F2 point")["validation_report"],
                name="F2 validation report",
            ),
        )
        for item in existing_data
    )
    if len({point.alpha for point in existing}) != len(existing):
        raise ValueError("existing F2 alpha values must be unique")
    if any(point.alpha <= 0.0 for point in existing):
        raise ValueError("the local point must not be duplicated in existing_high_fidelity")

    observation_data = _mapping(raw.get("observation"), name="observation")
    observation = DICObservationOperatorConfig(
        grid_mapping=str(observation_data.get("grid_mapping", "identity")),  # type: ignore[arg-type]
        grid_reduction=int(observation_data.get("grid_reduction", 1)),
        spatial_filter=str(observation_data.get("spatial_filter", "none")),  # type: ignore[arg-type]
        use_core_only=bool(observation_data.get("use_core_mask_only", True)),
    )
    ell_samples = int(ell.get("samples", 21))
    alpha_samples = int(alpha.get("samples", 21))
    if ell_samples < 2 or alpha_samples < 2:
        raise ValueError("parameter ranges require at least two samples")
    maximum_f2 = int(campaign.get("max_new_high_fidelity_runs", 5))
    if not 1 <= maximum_f2 <= 5:
        raise ValueError("max_new_high_fidelity_runs must lie in [1, 5]")
    fidelity = _mapping(raw.get("fidelity"), name="fidelity")
    low = _mapping(fidelity.get("low"), name="fidelity.low")
    low_spatial_reduction = int(low.get("spatial_reduction", 2))
    low_temporal_increments = int(low.get("temporal_increments", 10))
    low_minimum_elements_per_ell = float(low.get("minimum_elements_per_ell", 3.0))
    low_residual_tolerance = float(low.get("residual_tolerance", 3.0e-6))
    if low_spatial_reduction < 1:
        raise ValueError("fidelity.low.spatial_reduction must be positive")
    if low_temporal_increments < 1:
        raise ValueError("fidelity.low.temporal_increments must be positive")
    if (
        not np.isfinite(low_minimum_elements_per_ell)
        or low_minimum_elements_per_ell <= 0.0
    ):
        raise ValueError("fidelity.low.minimum_elements_per_ell must be positive")
    if not 0.0 < low_residual_tolerance < 1.0:
        raise ValueError("fidelity.low.residual_tolerance must lie in (0, 1)")
    return JointIdentificationConfig(
        source_path=source,
        name=str(campaign["name"]),
        input_directory=_resolve(root, campaign["input"], name="campaign.input"),
        output_directory=_resolve(root, campaign["output"], name="campaign.output"),
        local_campaign=_resolve(
            root,
            campaign["local_campaign"],
            name="campaign.local_campaign",
        ),
        partition_id=int(campaign["partition_id"]),
        h_ref_path=_resolve(root, campaign["h_ref"], name="campaign.h_ref"),
        maximum_new_high_fidelity_runs=maximum_f2,
        ell_min_um=ell_min,
        ell_max_um=ell_max,
        ell_samples=ell_samples,
        ell_anchors_um=tuple(float(value) for value in ell.get("anchors", ())),
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_samples=alpha_samples,
        alpha_anchors=tuple(float(value) for value in alpha.get("anchors", ())),
        low_spatial_reduction=low_spatial_reduction,
        low_temporal_increments=low_temporal_increments,
        low_minimum_elements_per_ell=low_minimum_elements_per_ell,
        low_residual_tolerance=low_residual_tolerance,
        existing_high_fidelity=existing,
        observation=observation,
        raw=raw,
    )


def _git_state(repository: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_local_context(
    config: JointIdentificationConfig,
) -> tuple[dict[str, Any], dict[str, Any], Any, float, dict[str, Any]]:
    manifest_path = config.local_campaign / "manifest.json"
    manifest = load_json_object(manifest_path)
    _, partition = partition_from_manifest(manifest, config.partition_id)
    manifest_sha = _manifest_sha256(manifest_path)
    status = load_partition_status(
        config.local_campaign,
        partition_id=config.partition_id,
        manifest_sha256=manifest_sha,
    )
    h_ref_data = load_json_object(config.h_ref_path)
    if int(h_ref_data.get("partition_id", -1)) != config.partition_id:
        raise ValueError("H_ref metadata identifies another partition")
    if h_ref_data.get("source_campaign_manifest_sha256") != manifest_sha:
        raise RuntimeError("H_ref metadata does not match the local campaign manifest")
    h_ref = float(h_ref_data["reference_hardening_modulus_mpa"])
    if not np.isfinite(h_ref) or h_ref <= 0.0:
        raise ValueError("H_ref metadata must contain a positive finite modulus")
    return manifest, status, partition, h_ref, h_ref_data


def inspect_joint_identification(
    config: JointIdentificationConfig,
) -> dict[str, Any]:
    """Inspect inputs and report the planned F0/F1/F2 scope without writing."""

    manifest, status, partition, h_ref, h_ref_data = _load_local_context(config)
    existing: list[dict[str, Any]] = []
    for point in config.existing_high_fidelity:
        candidate_manifest = load_json_object(point.campaign / "manifest.json")
        validate_mechanical_campaign_pair(manifest, candidate_manifest)
        candidate_h = float(
            candidate_manifest["config"]["nonlocal_plasticity"]["coupling_modulus_mpa"]
        )
        if not np.isclose(candidate_h / h_ref, point.alpha, rtol=1e-12, atol=1e-12):
            raise ValueError(f"F2 campaign alpha mismatch: {point.campaign}")
        if not point.validation_report.is_file():
            raise FileNotFoundError(f"missing F2 validation report: {point.validation_report}")
        existing.append(
            {
                "alpha": point.alpha,
                "campaign": str(point.campaign),
                "validation_report": str(point.validation_report),
                "wall_time_seconds": candidate_manifest.get("diagnostics", {}).get(
                    "elapsed_seconds"
                ),
            }
        )
    return {
        "schema_version": 1,
        "campaign": config.name,
        "configuration": str(config.source_path),
        "configuration_sha256": config.source_sha256,
        "local_campaign": str(config.local_campaign),
        "local_campaign_manifest_sha256": _manifest_sha256(
            config.local_campaign / "manifest.json"
        ),
        "partition_id": config.partition_id,
        "core_shape": list(partition.core_shape),
        "solve_shape": list(partition.solve_shape),
        "spacing_mm": manifest["config"]["mesh"]["base_pixel_size_mm"]
        * manifest["config"]["mesh"]["scale_factor"],
        "h_ref_mpa": h_ref,
        "h_ref_source": {
            "path": str(config.h_ref_path),
            "sha256": fingerprint_file(config.h_ref_path),
            "source_peeq_sha256": h_ref_data.get("source_peeq_sha256"),
        },
        "f0": {
            "length_count": len(config.ell_values_um()),
            "alpha_count": len(config.alpha_values()),
            "mechanical_solves": 0,
            "helmholtz_solves": len(config.ell_values_um()),
            "parameter_pair_count": len(config.ell_values_um())
            * len(config.alpha_values())
            + 1,
        },
        "f1": {
            "spatial_reduction": config.low_spatial_reduction,
            "temporal_increments": config.low_temporal_increments,
            "residual_tolerance": config.low_residual_tolerance,
            "default_validation_point_count": len(config.existing_high_fidelity) + 1,
            "default_validation_points": [
                {"alpha": 0.0, "length_scale_um": None},
                *[
                    {
                        "alpha": point.alpha,
                        "length_scale_um": _existing_length_um(point),
                    }
                    for point in config.existing_high_fidelity
                ],
            ],
        },
        "existing_high_fidelity": existing,
        "local_diagnostics": status.get("diagnostics", {}),
        "high_fidelity_auto_execution": False,
    }


def _gradient_energy_integral(
    field: FloatArray,
    *,
    spacing_x_mm: float,
    spacing_y_mm: float,
) -> float:
    gradient_x, gradient_y = np.gradient(
        field,
        spacing_x_mm,
        spacing_y_mm,
        edge_order=1,
    )
    cell_area = spacing_x_mm * spacing_y_mm
    return float(np.sum(gradient_x**2 + gradient_y**2) * cell_area)


def _frozen_cache_manifest(
    config: JointIdentificationConfig,
    *,
    local_manifest_sha256: str,
    peeq_sha256: str,
    h_ref_sha256: str,
    h_ref_mpa: float,
) -> dict[str, Any]:
    repository = _repository_root(config.source_path)
    git = _git_state(repository)
    data = {
        "schema_version": 1,
        "fidelity": "F0_frozen_field",
        "campaign": config.name,
        "created_with": {
            "software_version": __version__,
            "git": git,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "configuration_sha256": config.source_sha256,
        "local_campaign_manifest_sha256": local_manifest_sha256,
        "source_peeq_sha256": peeq_sha256,
        "h_ref_sha256": h_ref_sha256,
        "h_ref_mpa": h_ref_mpa,
        "partition_id": config.partition_id,
        "observation_operator": config.observation.as_dict(),
        "observation_operator_sha256": config.observation.fingerprint(),
        "lengths_um": list(config.ell_values_um()),
        "alphas": list(config.alpha_values()),
        "equations": {
            "helmholtz": "chi - ell^2 Laplacian(chi) = p0",
            "frozen_residual": "r_ell = p0 - chi_ell",
            "nonlocal_hardening": "q_nl = H_chi r_ell",
            "local_energy": "0.5 H_chi integral((p0-chi)^2) dOmega",
            "gradient_energy": "0.5 H_chi ell^2 integral(|grad chi|^2) dOmega",
        },
        "mechanical_solution_recomputed": False,
        "high_fidelity_auto_execution": False,
    }
    key_payload = dict(data)
    key_payload["created_with"] = {
        "software_version": __version__,
        "git_commit": git["commit"],
    }
    data["cache_key_sha256"] = sha256(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return data


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty metric table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    fieldnames = list(rows[0])
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def screen_frozen_field(
    config: JointIdentificationConfig,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run or resume the F0 frozen-PEEQ screen."""

    local_manifest, status, partition, h_ref, _ = _load_local_context(config)
    peeq_hash = str(status["outputs"]["PEEQ"])
    manifest_sha = _manifest_sha256(config.local_campaign / "manifest.json")
    f0_directory = config.output_directory / "f0"
    expected_manifest = _frozen_cache_manifest(
        config,
        local_manifest_sha256=manifest_sha,
        peeq_sha256=peeq_hash,
        h_ref_sha256=fingerprint_file(config.h_ref_path),
        h_ref_mpa=h_ref,
    )
    manifest_path = f0_directory / "manifest.json"
    metrics_path = f0_directory / "frozen_screen.csv"
    diagnostics_path = f0_directory / "length_diagnostics.json"
    validation_path = f0_directory / "proxy_validation.json"
    if dry_run:
        return {
            **inspect_joint_identification(config)["f0"],
            "output": str(f0_directory),
            "cache_key_sha256": expected_manifest["cache_key_sha256"],
            "status": "dry_run",
        }
    if manifest_path.exists():
        existing_manifest = load_json_object(manifest_path)
        if existing_manifest.get("cache_key_sha256") != expected_manifest["cache_key_sha256"]:
            raise RuntimeError("existing F0 cache is incompatible with the requested campaign")
        if metrics_path.is_file() and diagnostics_path.is_file() and validation_path.is_file():
            return {
                "status": "reused",
                "manifest": str(manifest_path),
                "metrics": str(metrics_path),
                "validation": str(validation_path),
                "cache_key_sha256": expected_manifest["cache_key_sha256"],
            }
        raise RuntimeError("existing F0 cache is incomplete; refusing implicit overwrite")
    if f0_directory.exists() and any(f0_directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty F0 directory: {f0_directory}")

    peeq = load_verified_partition_field(
        config.local_campaign,
        partition_id=config.partition_id,
        status=status,
        name="PEEQ",
    )
    if peeq.shape != partition.solve_shape:
        raise ValueError("saved PEEQ shape does not match partition solve bounds")
    core_slice = partition.core_element_slice_local
    peeq_core = np.asarray(peeq[core_slice], dtype=np.float64)
    mesh = local_manifest["config"]["mesh"]
    spacing = float(mesh["base_pixel_size_mm"]) * float(mesh["scale_factor"])
    cell_area = spacing**2
    local_diagnostics = peeq_diagnostic_metrics(
        peeq_core,
        spacing_x_mm=spacing,
        spacing_y_mm=spacing,
        first_positive_plastic_strain=float(
            local_manifest["config"]["material"]["first_positive_plastic_strain"]
        ),
    )
    spectrum = radial_power_spectrum(
        peeq_core,
        spacing_x_mm=spacing,
        spacing_y_mm=spacing,
    )
    frequencies = np.asarray(spectrum["frequency_cycles_per_mm"], dtype=np.float64)
    powers = np.asarray(spectrum["normalized_power"], dtype=np.float64)
    nonzero = frequencies > 0.0
    dominant_frequency = (
        float(frequencies[nonzero][np.argmax(powers[nonzero])])
        if nonzero.any()
        else 0.0
    )
    dominant_wavenumber = 2.0 * np.pi * dominant_frequency

    rows: list[dict[str, Any]] = [
        {
            **NonlocalIdentificationPoint.from_alpha_and_length_um(
                alpha=0.0,
                length_scale_um=None,
                h_ref_mpa=h_ref,
            ).as_dict(),
            "residual_l2_core": 0.0,
            "residual_linf_core": 0.0,
            "residual_mean_core": 0.0,
            "residual_std_core": 0.0,
            "residual_q90_core": 0.0,
            "residual_q95_core": 0.0,
            "residual_q99_core": 0.0,
            "residual_gradient_rms_core": 0.0,
            "residual_total_variation_core": 0.0,
            "nonlocal_hardening_l2_mpa_core": 0.0,
            "nonlocal_hardening_linf_mpa_core": 0.0,
            "local_energy_mpa_mm2": 0.0,
            "gradient_energy_mpa_mm2": 0.0,
            "spectral_multiplier_at_dominant_k_mpa": 0.0,
            "helmholtz_mean_drift": 0.0,
            "helmholtz_residual_relative": 0.0,
        }
    ]
    length_records: list[dict[str, Any]] = []
    for ell_um in config.ell_values_um():
        ell_mm = ell_um / 1_000.0
        filtered = helmholtz_filter_element_field(
            peeq,
            length_scale_mm=ell_mm,
            spacing_x_mm=spacing,
            spacing_y_mm=spacing,
        )
        residual = np.asarray(peeq - filtered.filtered_element_field, dtype=np.float64)
        residual_core = residual[core_slice]
        chi = filtered.filtered_element_field
        residual_diffusivity = field_diffusivity_metrics(
            residual_core,
            raw_field=residual_core,
            spacing_x_mm=spacing,
            spacing_y_mm=spacing,
        )
        residual_squared_integral = float(np.sum(residual**2) * cell_area)
        gradient_squared_integral = _gradient_energy_integral(
            chi,
            spacing_x_mm=spacing,
            spacing_y_mm=spacing,
        )
        length_records.append(
            {
                "length_scale_um": ell_um,
                "length_scale_mm": ell_mm,
                "residual_l2_core": float(np.linalg.norm(residual_core)),
                "residual_linf_core": float(np.max(np.abs(residual_core))),
                "residual_mean_core": float(np.mean(residual_core)),
                "residual_std_core": float(np.std(residual_core)),
                "residual_q90_core": float(np.quantile(residual_core, 0.90)),
                "residual_q95_core": float(np.quantile(residual_core, 0.95)),
                "residual_q99_core": float(np.quantile(residual_core, 0.99)),
                "residual_gradient_rms_core": residual_diffusivity.gradient_rms,
                "residual_total_variation_core": residual_diffusivity.total_variation,
                "residual_squared_integral": residual_squared_integral,
                "chi_gradient_squared_integral": gradient_squared_integral,
                "helmholtz_mean_drift": filtered.mean_drift,
                "helmholtz_residual_relative": filtered.residual_relative,
            }
        )
        for alpha in config.alpha_values():
            point = NonlocalIdentificationPoint.from_alpha_and_length_um(
                alpha=alpha,
                length_scale_um=ell_um,
                h_ref_mpa=h_ref,
            )
            h_chi = point.h_chi_mpa
            spectral_multiplier = (
                h_chi
                * (ell_mm * dominant_wavenumber) ** 2
                / (1.0 + (ell_mm * dominant_wavenumber) ** 2)
            )
            rows.append(
                {
                    **point.as_dict(),
                    "residual_l2_core": float(np.linalg.norm(residual_core)),
                    "residual_linf_core": float(np.max(np.abs(residual_core))),
                    "residual_mean_core": float(np.mean(residual_core)),
                    "residual_std_core": float(np.std(residual_core)),
                    "residual_q90_core": float(np.quantile(residual_core, 0.90)),
                    "residual_q95_core": float(np.quantile(residual_core, 0.95)),
                    "residual_q99_core": float(np.quantile(residual_core, 0.99)),
                    "residual_gradient_rms_core": residual_diffusivity.gradient_rms,
                    "residual_total_variation_core": residual_diffusivity.total_variation,
                    "nonlocal_hardening_l2_mpa_core": float(
                        h_chi * np.linalg.norm(residual_core)
                    ),
                    "nonlocal_hardening_linf_mpa_core": float(
                        h_chi * np.max(np.abs(residual_core))
                    ),
                    "local_energy_mpa_mm2": 0.5
                    * h_chi
                    * residual_squared_integral,
                    "gradient_energy_mpa_mm2": 0.5
                    * h_chi
                    * ell_mm**2
                    * gradient_squared_integral,
                    "spectral_multiplier_at_dominant_k_mpa": spectral_multiplier,
                    "helmholtz_mean_drift": filtered.mean_drift,
                    "helmholtz_residual_relative": filtered.residual_relative,
                }
            )

    validation = _validate_frozen_proxy(
        config,
        local_manifest=local_manifest,
        local_status=status,
        partition=partition,
        h_ref_mpa=h_ref,
        f0_rows=rows,
        spacing_mm=spacing,
    )
    expected_manifest["generated_at"] = datetime.now(UTC).isoformat()
    expected_manifest["source_field"] = {
        "name": "PEEQ",
        "shape": list(peeq.shape),
        "core_shape": list(peeq_core.shape),
        "statistics_core": local_diagnostics,
        "radial_spectrum_core": spectrum,
        "dominant_frequency_cycles_per_mm": dominant_frequency,
        "dominant_wavenumber_rad_per_mm": dominant_wavenumber,
    }
    f0_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(metrics_path, rows)
    _atomic_write(
        diagnostics_path,
        _canonical_json({"lengths": length_records}),
    )
    _atomic_write(validation_path, _canonical_json(validation))
    _atomic_write(manifest_path, _canonical_json(expected_manifest))
    return {
        "status": "completed",
        "manifest": str(manifest_path),
        "metrics": str(metrics_path),
        "validation": str(validation_path),
        "cache_key_sha256": expected_manifest["cache_key_sha256"],
        "helmholtz_solves": len(config.ell_values_um()),
        "parameter_pair_count": len(rows),
    }


def _nearest_f0_row(
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    length_scale_um: float,
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if not row["is_local"]
        and np.isclose(float(row["alpha"]), alpha, rtol=0.0, atol=1e-12)
    ]
    if not candidates:
        raise ValueError(f"F0 grid does not contain alpha={alpha}")
    return min(
        candidates,
        key=lambda row: abs(float(row["length_scale_um"]) - length_scale_um),
    )


def _validate_frozen_proxy(
    config: JointIdentificationConfig,
    *,
    local_manifest: dict[str, Any],
    local_status: dict[str, Any],
    partition: Any,
    h_ref_mpa: float,
    f0_rows: list[dict[str, Any]],
    spacing_mm: float,
) -> dict[str, Any]:
    actual: list[dict[str, Any]] = []
    if not config.existing_high_fidelity:
        return {
            "status": "not_available",
            "reason": "no existing F2 campaigns declared",
        }
    first_report = load_json_object(config.existing_high_fidelity[0].validation_report)
    local_metrics = first_report["metrics"]["local"]
    local_peeq = load_verified_partition_field(
        config.local_campaign,
        partition_id=config.partition_id,
        status=local_status,
        name="PEEQ",
    )[partition.core_element_slice_local]
    actual.append(
        {
            "alpha": 0.0,
            "proxy_nonlocal_hardening_l2_mpa_core": 0.0,
            "relative_l2_error": float(local_metrics["relative_l2_error"]),
            "peeq_maximum": float(np.max(local_peeq)),
            "peeq_standard_deviation": float(np.std(local_peeq)),
            "peeq_total_variation": field_diffusivity_metrics(
                local_peeq,
                raw_field=local_peeq,
                spacing_x_mm=spacing_mm,
                spacing_y_mm=spacing_mm,
            ).total_variation,
        }
    )
    reference_length_um: float | None = None
    for existing in sorted(config.existing_high_fidelity, key=lambda item: item.alpha):
        candidate_manifest_path = existing.campaign / "manifest.json"
        candidate_manifest = load_json_object(candidate_manifest_path)
        validate_mechanical_campaign_pair(local_manifest, candidate_manifest)
        nonlocal_config = candidate_manifest["config"]["nonlocal_plasticity"]
        length_um = float(nonlocal_config["length_scale_mm"]) * 1_000.0
        if reference_length_um is None:
            reference_length_um = length_um
        elif not np.isclose(length_um, reference_length_um, rtol=0.0, atol=1e-12):
            raise ValueError("F0 proxy validation requires one common existing F2 length")
        h_chi = float(nonlocal_config["coupling_modulus_mpa"])
        if not np.isclose(h_chi / h_ref_mpa, existing.alpha, rtol=1e-12, atol=1e-12):
            raise ValueError(f"existing F2 alpha metadata mismatch: {existing.campaign}")
        candidate_status = load_partition_status(
            existing.campaign,
            partition_id=config.partition_id,
            manifest_sha256=_manifest_sha256(candidate_manifest_path),
        )
        peeq = load_verified_partition_field(
            existing.campaign,
            partition_id=config.partition_id,
            status=candidate_status,
            name="PEEQ",
        )[partition.core_element_slice_local]
        report = load_json_object(existing.validation_report)
        proxy = _nearest_f0_row(
            f0_rows,
            alpha=existing.alpha,
            length_scale_um=length_um,
        )
        actual.append(
            {
                "alpha": existing.alpha,
                "proxy_length_scale_um": proxy["length_scale_um"],
                "proxy_nonlocal_hardening_l2_mpa_core": proxy[
                    "nonlocal_hardening_l2_mpa_core"
                ],
                "relative_l2_error": float(
                    report["metrics"]["coupled"]["relative_l2_error"]
                ),
                "peeq_maximum": float(np.max(peeq)),
                "peeq_standard_deviation": float(np.std(peeq)),
                "peeq_total_variation": field_diffusivity_metrics(
                    peeq,
                    raw_field=peeq,
                    spacing_x_mm=spacing_mm,
                    spacing_y_mm=spacing_mm,
                ).total_variation,
            }
        )
    proxy_strength = [row["proxy_nonlocal_hardening_l2_mpa_core"] for row in actual]
    indicators = (
        "relative_l2_error",
        "peeq_maximum",
        "peeq_standard_deviation",
        "peeq_total_variation",
    )
    rank_correlations = {
        name: float(spearmanr(proxy_strength, [row[name] for row in actual]).statistic)
        for name in indicators
    }
    return {
        "status": "evaluated",
        "reference_length_um": reference_length_um,
        "points": actual,
        "rank_correlation_proxy_strength_vs_f2": rank_correlations,
        "interpretation": {
            name: {
                "predictive_direction": correlation <= -0.8,
                "expected_direction": (
                    "a stronger frozen penalty should correspond to a smaller F2 value"
                ),
            }
            for name, correlation in rank_correlations.items()
        },
        "limitation": (
            "F0 does not reintegrate plasticity and is used only for ordering, "
            "spectral equivalence and candidate rejection."
        ),
    }


@dataclass(frozen=True, slots=True)
class ReducedCaseInputs:
    """Coarsened global inputs reused by every F1 candidate."""

    displacement_x_mm: FloatArray
    displacement_y_mm: FloatArray
    yield_stress_mpa: FloatArray
    hardening_coefficient_mpa: FloatArray
    layout: PartitionLayout
    spacing_mm: float
    source_manifest_sha256: str
    reduction_manifest: dict[str, Any]


def _existing_length_um(point: ExistingHighFidelityPoint) -> float:
    manifest = load_json_object(point.campaign / "manifest.json")
    return float(manifest["config"]["nonlocal_plasticity"]["length_scale_mm"]) * 1_000.0


def _load_prepared_array(
    input_directory: Path,
    manifest: dict[str, Any],
    name: str,
) -> FloatArray:
    output = _mapping(manifest.get("outputs"), name="prepared outputs").get(name)
    output_data = _mapping(output, name=f"prepared output {name}")
    path = input_directory / str(output_data["filename"])
    if not path.is_file():
        raise FileNotFoundError(f"missing prepared input field {name}: {path}")
    if fingerprint_file(path) != output_data["sha256"]:
        raise RuntimeError(f"prepared input field fails its manifest hash: {path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if tuple(values.shape) != tuple(output_data["shape"]):
        raise ValueError(f"prepared input field has wrong shape: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"prepared input field contains non-finite values: {path}")
    return np.asarray(values, dtype=np.float64)


def _area_average(field: FloatArray, factor: int) -> FloatArray:
    if field.ndim != 2:
        raise ValueError("element field must be two-dimensional")
    nx, ny = field.shape
    if nx % factor or ny % factor:
        raise ValueError("element-field dimensions must be divisible by reduction")
    return np.asarray(
        field.reshape(nx // factor, factor, ny // factor, factor).mean(axis=(1, 3)),
        dtype=np.float64,
    )


def _coincident_nodes(field: FloatArray, factor: int) -> FloatArray:
    if field.ndim != 2:
        raise ValueError("nodal field must be two-dimensional")
    if (field.shape[0] - 1) % factor or (field.shape[1] - 1) % factor:
        raise ValueError("nodal dimensions minus one must be divisible by reduction")
    return np.asarray(field[::factor, ::factor], dtype=np.float64)


def prepare_low_fidelity_inputs(
    config: JointIdentificationConfig,
) -> ReducedCaseInputs:
    """Coarsen the global prepared case without changing its physical extent."""

    local_manifest, _, _, _, _ = _load_local_context(config)
    prepared_manifest_path = config.input_directory / "manifest.json"
    prepared_manifest = load_json_object(prepared_manifest_path)
    factor = config.low_spatial_reduction
    displacement_x = _coincident_nodes(
        _load_prepared_array(config.input_directory, prepared_manifest, "displacement_x_mm"),
        factor,
    )
    displacement_y = _coincident_nodes(
        _load_prepared_array(config.input_directory, prepared_manifest, "displacement_y_mm"),
        factor,
    )
    yield_stress = _area_average(
        _load_prepared_array(config.input_directory, prepared_manifest, "yield_stress_mpa"),
        factor,
    )
    hardening = _area_average(
        _load_prepared_array(
            config.input_directory,
            prepared_manifest,
            "hardening_coefficient_mpa",
        ),
        factor,
    )
    original_layout = _mapping(local_manifest["layout"], name="local campaign layout")
    original_padding = int(original_layout["padding"])
    if original_padding % factor:
        raise ValueError("partition padding must be divisible by F1 reduction")
    partition_shape_data = tuple(int(value) for value in original_layout["partition_shape"])
    layout = PartitionLayout(
        global_shape=yield_stress.shape,
        partition_shape=(partition_shape_data[0], partition_shape_data[1]),
        padding=original_padding // factor,
    )
    mesh = local_manifest["config"]["mesh"]
    original_spacing = float(mesh["base_pixel_size_mm"]) * float(mesh["scale_factor"])
    reduction_manifest = {
        "schema_version": 1,
        "fidelity": "F1_low",
        "source_prepared_manifest": str(prepared_manifest_path),
        "source_prepared_manifest_sha256": fingerprint_file(prepared_manifest_path),
        "source_campaign_manifest_sha256": _manifest_sha256(
            config.local_campaign / "manifest.json"
        ),
        "spatial_reduction": factor,
        "nodal_reduction": "coincident-node-stride",
        "element_reduction": "two-dimensional-area-mean",
        "physical_extent_preserved": True,
        "source_element_shape": list(original_layout["global_shape"]),
        "reduced_element_shape": list(yield_stress.shape),
        "source_padding_elements": original_padding,
        "reduced_padding_elements": original_padding // factor,
        "source_spacing_mm": original_spacing,
        "reduced_spacing_mm": original_spacing * factor,
        "input_hashes": {
            "displacement_x_mm": fingerprint_array(displacement_x),
            "displacement_y_mm": fingerprint_array(displacement_y),
            "yield_stress_mpa": fingerprint_array(yield_stress),
            "hardening_coefficient_mpa": fingerprint_array(hardening),
        },
    }
    return ReducedCaseInputs(
        displacement_x_mm=displacement_x,
        displacement_y_mm=displacement_y,
        yield_stress_mpa=yield_stress,
        hardening_coefficient_mpa=hardening,
        layout=layout,
        spacing_mm=original_spacing * factor,
        source_manifest_sha256=fingerprint_file(prepared_manifest_path),
        reduction_manifest=reduction_manifest,
    )


def _point_identifier(point: NonlocalIdentificationPoint) -> str:
    if point.is_local:
        return "local"
    assert point.length_scale_um is not None
    length = f"{point.length_scale_um:09.3f}".replace(".", "p")
    alpha = f"{point.alpha:07.3f}".replace(".", "p")
    return f"ell-{length}um-alpha-{alpha}"


def _parse_point_selector(
    selector: str,
    *,
    h_ref_mpa: float,
) -> NonlocalIdentificationPoint:
    normalized = selector.strip().lower()
    if normalized == "local":
        return NonlocalIdentificationPoint.from_alpha_and_length_um(
            alpha=0.0,
            length_scale_um=None,
            h_ref_mpa=h_ref_mpa,
        )
    try:
        alpha_text, length_text = normalized.split(":", maxsplit=1)
        alpha = float(alpha_text)
        length_um = float(length_text)
    except ValueError as error:
        raise ValueError(
            "point selectors must be 'local' or 'ALPHA:LENGTH_UM'"
        ) from error
    return NonlocalIdentificationPoint.from_alpha_and_length_um(
        alpha=alpha,
        length_scale_um=length_um,
        h_ref_mpa=h_ref_mpa,
    )


def _default_f1_validation_points(
    config: JointIdentificationConfig,
    *,
    h_ref_mpa: float,
) -> tuple[NonlocalIdentificationPoint, ...]:
    points = [
        NonlocalIdentificationPoint.from_alpha_and_length_um(
            alpha=0.0,
            length_scale_um=None,
            h_ref_mpa=h_ref_mpa,
        )
    ]
    points.extend(
        NonlocalIdentificationPoint.from_alpha_and_length_um(
            alpha=existing.alpha,
            length_scale_um=_existing_length_um(existing),
            h_ref_mpa=h_ref_mpa,
        )
        for existing in sorted(config.existing_high_fidelity, key=lambda item: item.alpha)
    )
    return tuple(points)


def _material_and_solver_from_local_manifest(
    local_manifest: dict[str, Any],
    config: JointIdentificationConfig,
    *,
    reduced_inputs: ReducedCaseInputs,
    point: NonlocalIdentificationPoint,
) -> CaseStudyConfig:
    config_data = local_manifest["config"]
    material = MaterialConfig(**config_data["material"])
    source_solver = SolverConfig(**config_data["solver"])
    solver = replace(
        source_solver,
        increments=config.low_temporal_increments,
        residual_tolerance=config.low_residual_tolerance,
    )
    source_nonlocal = NonlocalPlasticityConfig(**config_data["nonlocal_plasticity"])
    nonlocal_config = replace(
        source_nonlocal,
        enabled=not point.is_local,
        length_scale_mm=(
            source_nonlocal.length_scale_mm
            if point.length_scale_mm is None
            else point.length_scale_mm
        ),
        coupling_modulus_mpa=point.h_chi_mpa,
    )
    source_mesh = config_data["mesh"]
    return CaseStudyConfig(
        mesh=MeshConfig(
            nx=reduced_inputs.layout.global_shape[0],
            ny=reduced_inputs.layout.global_shape[1],
            base_pixel_size_mm=float(source_mesh["base_pixel_size_mm"]),
            scale_factor=float(source_mesh["scale_factor"])
            * config.low_spatial_reduction,
        ),
        material=material,
        solver=solver,
        nonlocal_plasticity=nonlocal_config,
    )


def _f1_point_manifest(
    config: JointIdentificationConfig,
    point: NonlocalIdentificationPoint,
    reduced_inputs: ReducedCaseInputs,
    case_config: CaseStudyConfig,
) -> dict[str, Any]:
    data = {
        "schema_version": 1,
        "campaign": config.name,
        "fidelity": "F1_low",
        "point_id": _point_identifier(point),
        "parameters": point.as_dict(),
        "configuration_sha256": config.source_sha256,
        "reduction": reduced_inputs.reduction_manifest,
        "case_config": asdict(case_config),
        "partition_id": config.partition_id,
        "observation_operator": {
            **config.observation.as_dict(),
            "grid_mapping": "coincident-node-stride",
            "grid_reduction": config.low_spatial_reduction,
        },
        "high_fidelity_auto_execution": False,
    }
    data["cache_key_sha256"] = sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return data


def _amplitude_config(config: JointIdentificationConfig) -> AmplitudeMetricConfig:
    objectives = _mapping(config.raw.get("objectives"), name="objectives")
    quantiles = tuple(float(value) for value in objectives.get("quantiles", (0.5, 0.9)))
    weights = tuple(
        float(value)
        for value in objectives.get("quantile_weights", (1.0,) * len(quantiles))
    )
    return AmplitudeMetricConfig(
        quantiles=quantiles,
        quantile_weights=weights,
        standard_deviation_weight=float(
            objectives.get("standard_deviation_weight", 1.0)
        ),
    )


def _evaluate_f1_point(
    config: JointIdentificationConfig,
    point: NonlocalIdentificationPoint,
    reduced_inputs: ReducedCaseInputs,
    campaign_directory: Path,
    status: dict[str, Any],
    point_manifest: dict[str, Any],
) -> dict[str, Any]:
    partition = reduced_inputs.layout.get(config.partition_id)
    fem_u = load_verified_partition_field(
        campaign_directory,
        partition_id=config.partition_id,
        status=status,
        name="U",
    )
    peeq = load_verified_partition_field(
        campaign_directory,
        partition_id=config.partition_id,
        status=status,
        name="PEEQ",
    )
    dic_u = np.stack(
        (
            extract_partition_field(
                reduced_inputs.displacement_x_mm,
                layout=reduced_inputs.layout,
                partition=partition,
                location="node",
            ),
            extract_partition_field(
                reduced_inputs.displacement_y_mm,
                layout=reduced_inputs.layout,
                partition=partition,
                location="node",
            ),
        ),
        axis=-1,
    )
    observation = DICObservationOperator(
        DICObservationOperatorConfig(),
        poisson_ratio=float(point_manifest["case_config"]["material"]["poisson_ratio"]),
    )
    dic_observed = observation.observe_displacement(
        dic_u,
        spacing_x_mm=reduced_inputs.spacing_mm,
        spacing_y_mm=reduced_inputs.spacing_mm,
        core_slice=partition.core_element_slice_local,
    )
    fem_observed = observation.observe_displacement(
        fem_u,
        spacing_x_mm=reduced_inputs.spacing_mm,
        spacing_y_mm=reduced_inputs.spacing_mm,
        core_slice=partition.core_element_slice_local,
    )
    nonlocal_hardening = None
    if not point.is_local:
        nonlocal_hardening = load_verified_partition_field(
            campaign_directory,
            partition_id=config.partition_id,
            status=status,
            name="NONLOCAL_HARDENING_MPA",
        )[partition.core_element_slice_local]
    metrics = evaluate_identification_metrics(
        dic_observed.element_field,
        fem_observed.element_field,
        spacing_x_mm=reduced_inputs.spacing_mm,
        spacing_y_mm=reduced_inputs.spacing_mm,
        mask=dic_observed.valid_mask & fem_observed.valid_mask,
        amplitude_config=_amplitude_config(config),
    )
    peeq_core = np.asarray(peeq[partition.core_element_slice_local], dtype=np.float64)
    peeq_metrics = peeq_diagnostic_metrics(
        peeq_core,
        spacing_x_mm=reduced_inputs.spacing_mm,
        spacing_y_mm=reduced_inputs.spacing_mm,
        first_positive_plastic_strain=float(
            point_manifest["case_config"]["material"]["first_positive_plastic_strain"]
        ),
        nonlocal_hardening_mpa=nonlocal_hardening,
    )
    return {
        "schema_version": 1,
        "fidelity": "F1_low",
        "point_id": _point_identifier(point),
        "parameters": point.as_dict(),
        "metrics": metrics,
        "peeq_diagnostics": peeq_metrics,
        "solver_diagnostics": status.get("diagnostics", {}),
        "observation": {
            "operator_sha256": dic_observed.operator_sha256,
            "core_shape": list(dic_observed.element_field.shape),
            "spacing_x_mm": dic_observed.spacing_x_mm,
            "spacing_y_mm": dic_observed.spacing_y_mm,
            "same_operator_for_dic_and_fem": True,
            "padding_excluded_from_metrics": True,
        },
    }


def run_low_fidelity(
    config: JointIdentificationConfig,
    *,
    point_selectors: tuple[str, ...] = (),
    dry_run: bool = False,
    maximum_workers: int = 1,
) -> dict[str, Any]:
    """Run selected F1 points through the existing partition workflow."""

    if maximum_workers < 1:
        raise ValueError("maximum_workers must be positive")
    local_manifest, _, _, h_ref, _ = _load_local_context(config)
    points = (
        tuple(
            _parse_point_selector(selector, h_ref_mpa=h_ref)
            for selector in point_selectors
        )
        if point_selectors
        else _default_f1_validation_points(config, h_ref_mpa=h_ref)
    )
    unique_points = tuple(
        {
            _point_identifier(point): point
            for point in points
        }.values()
    )
    reduced_inputs = prepare_low_fidelity_inputs(config)
    minimum_length_um = min(
        (
            point.length_scale_um
            for point in unique_points
            if point.length_scale_um is not None
        ),
        default=None,
    )
    resolution_ratio = (
        None
        if minimum_length_um is None
        else (minimum_length_um / 1_000.0) / reduced_inputs.spacing_mm
    )
    if (
        resolution_ratio is not None
        and resolution_ratio < config.low_minimum_elements_per_ell
    ):
        raise ValueError(
            "F1 reduction violates the configured minimum number of elements per ell"
        )
    plan = [
        {
            "point_id": _point_identifier(point),
            **point.as_dict(),
            "output": str(
                config.output_directory
                / "f1"
                / "points"
                / _point_identifier(point)
                / "campaign"
            ),
        }
        for point in unique_points
    ]
    if dry_run:
        return {
            "status": "dry_run",
            "point_count": len(plan),
            "points": plan,
            "maximum_workers": maximum_workers,
            "execution_policy": "sequential_reuse_of_reduced_input_buffers",
            "reduction": reduced_inputs.reduction_manifest,
            "minimum_ell_over_h_f1": resolution_ratio,
            "high_fidelity_auto_execution": False,
        }
    if maximum_workers != 1:
        raise ValueError(
            "the current F1 implementation requires --workers 1 to avoid "
            "oversubscribing the MFront thread pool"
        )

    results: list[dict[str, Any]] = []
    for point in unique_points:
        point_id = _point_identifier(point)
        point_directory = config.output_directory / "f1" / "points" / point_id
        campaign_directory = point_directory / "campaign"
        manifest_path = point_directory / "identification_manifest.json"
        report_path = point_directory / "metrics.json"
        case_config = _material_and_solver_from_local_manifest(
            local_manifest,
            config,
            reduced_inputs=reduced_inputs,
            point=point,
        )
        point_manifest = _f1_point_manifest(
            config,
            point,
            reduced_inputs,
            case_config,
        )
        if manifest_path.exists():
            existing = load_json_object(manifest_path)
            if existing.get("cache_key_sha256") != point_manifest["cache_key_sha256"]:
                raise RuntimeError(f"incompatible F1 cache for {point_id}")
        elif point_directory.exists() and any(point_directory.iterdir()):
            raise FileExistsError(f"refusing to overwrite incomplete F1 point: {point_id}")
        else:
            _atomic_write(manifest_path, _canonical_json(point_manifest))
        workflow = PartitionWorkflow(
            config=case_config,
            layout=reduced_inputs.layout,
            displacement_x_mm=reduced_inputs.displacement_x_mm,
            displacement_y_mm=reduced_inputs.displacement_y_mm,
            yield_stress_mpa=reduced_inputs.yield_stress_mpa,
            hardening_coefficient_mpa=reduced_inputs.hardening_coefficient_mpa,
            output_directory=campaign_directory,
        )
        started = time.perf_counter()
        try:
            status = workflow.solve_partition(config.partition_id)
            metrics = _evaluate_f1_point(
                config,
                point,
                reduced_inputs,
                campaign_directory,
                status,
                point_manifest,
            )
            metrics["wall_time_seconds_with_postprocessing"] = time.perf_counter() - started
            _atomic_write(report_path, _canonical_json(metrics))
            results.append(
                {
                    "point_id": point_id,
                    "status": "completed",
                    "metrics": str(report_path),
                    "campaign": str(campaign_directory),
                    "elapsed_seconds": status.get("diagnostics", {}).get("elapsed_seconds"),
                }
            )
        except Exception as error:
            _atomic_write(
                point_directory / "failure.json",
                _canonical_json(
                    {
                        "point_id": point_id,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                ),
            )
            raise
    validation = validate_low_fidelity_ranking(config)
    return {
        "status": "completed",
        "points": results,
        "validation": validation,
        "high_fidelity_auto_execution": False,
    }


def _f1_metrics_path(
    config: JointIdentificationConfig,
    point: NonlocalIdentificationPoint,
) -> Path:
    return (
        config.output_directory
        / "f1"
        / "points"
        / _point_identifier(point)
        / "metrics.json"
    )


def validate_low_fidelity_ranking(
    config: JointIdentificationConfig,
) -> dict[str, Any]:
    """Validate F1 ordering and errors against all declared F2 points."""

    _, _, _, h_ref, _ = _load_local_context(config)
    points = _default_f1_validation_points(config, h_ref_mpa=h_ref)
    missing = [
        str(_f1_metrics_path(config, point))
        for point in points
        if not _f1_metrics_path(config, point).is_file()
    ]
    if missing:
        return {
            "status": "incomplete",
            "missing_metrics": missing,
            "usable_for_candidate_selection": False,
        }
    first_f2 = load_json_object(config.existing_high_fidelity[0].validation_report)
    rows: list[dict[str, Any]] = []
    for point in points:
        f1 = load_json_object(_f1_metrics_path(config, point))
        f1_global = f1["metrics"]["global"]
        f1_relative = f1["metrics"]["localization_relative_top"]
        f1_absolute = f1["metrics"]["localization_absolute_dic_quantile"]
        if point.is_local:
            f2_metrics = first_f2["metrics"]["local"]
        else:
            match = next(
                existing
                for existing in config.existing_high_fidelity
                if np.isclose(existing.alpha, point.alpha, rtol=0.0, atol=1e-12)
            )
            f2_metrics = load_json_object(match.validation_report)["metrics"]["coupled"]
        rows.append(
            {
                "alpha": point.alpha,
                "f1_relative_l2": float(f1_global["relative_l2_error"]),
                "f2_relative_l2": float(f2_metrics["relative_l2_error"]),
                "f1_correlation": float(f1_global["pearson_correlation"]),
                "f2_correlation": float(f2_metrics["pearson_correlation"]),
                "f1_top10_iou": float(f1_relative["intersection_over_union"]),
                "f2_top10_iou": float(f2_metrics["top10_iou"]),
                "f1_absolute_q90_iou": float(f1_absolute["intersection_over_union"]),
                "f2_absolute_q90_iou": float(f2_metrics["dic_q90_iou"]),
                "f1_peeq_q99": float(f1["peeq_diagnostics"]["quantiles"]["q99"]),
            }
        )
    f1_l2 = [row["f1_relative_l2"] for row in rows]
    f2_l2 = [row["f2_relative_l2"] for row in rows]
    f1_correlation = [row["f1_correlation"] for row in rows]
    f2_correlation = [row["f2_correlation"] for row in rows]
    checks = {
        "same_l2_ranking": list(np.argsort(f1_l2)) == list(np.argsort(f2_l2)),
        "same_correlation_ranking": list(np.argsort(f1_correlation))
        == list(np.argsort(f2_correlation)),
        "maximum_absolute_correlation_error_le_0p05": max(
            abs(row["f1_correlation"] - row["f2_correlation"]) for row in rows
        )
        <= 0.05,
        "maximum_relative_l2_error_le_15_percent": max(
            abs(row["f1_relative_l2"] - row["f2_relative_l2"])
            / max(abs(row["f2_relative_l2"]), np.finfo(float).eps)
            for row in rows
        )
        <= 0.15,
        "maximum_top10_iou_error_le_0p05": max(
            abs(row["f1_top10_iou"] - row["f2_top10_iou"]) for row in rows
        )
        <= 0.05,
        "maximum_absolute_q90_iou_error_le_0p05": max(
            abs(row["f1_absolute_q90_iou"] - row["f2_absolute_q90_iou"])
            for row in rows
        )
        <= 0.05,
    }
    report = {
        "schema_version": 1,
        "status": "evaluated",
        "points": rows,
        "checks": checks,
        "usable_for_candidate_selection": all(checks.values()),
        "rule": "F1 cannot select F2 candidates unless every validation check passes.",
    }
    output = config.output_directory / "f1" / "validation.json"
    _atomic_write(output, _canonical_json(report))
    return {**report, "path": str(output)}
