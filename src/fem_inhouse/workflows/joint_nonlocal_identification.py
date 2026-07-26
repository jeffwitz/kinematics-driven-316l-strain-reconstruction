"""Staged, cache-safe identification of ``ell`` and ``H_chi``."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
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
from fem_inhouse.data_preparation import fingerprint_file
from fem_inhouse.identification.metrics import peeq_diagnostic_metrics, radial_power_spectrum
from fem_inhouse.identification.observation import DICObservationOperatorConfig
from fem_inhouse.identification.parameters import NonlocalIdentificationPoint
from fem_inhouse.postprocessing.helmholtz import helmholtz_filter_element_field
from fem_inhouse.postprocessing.metrics import field_diffusivity_metrics
from fem_inhouse.workflows.campaign_access import (
    load_json_object,
    load_partition_status,
    load_verified_partition_field,
    partition_from_manifest,
    validate_mechanical_campaign_pair,
)

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
