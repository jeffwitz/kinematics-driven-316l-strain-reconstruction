from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fem_inhouse.data_preparation import fingerprint_file
from fem_inhouse.workflows.joint_nonlocal_identification import (
    inspect_joint_identification,
    load_joint_identification_config,
    screen_frozen_field,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_configuration(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='synthetic'\n", encoding="utf-8")
    campaign = tmp_path / "local"
    manifest = {
        "config": {
            "mesh": {
                "nx": 8,
                "ny": 6,
                "base_pixel_size_mm": 0.001,
                "scale_factor": 2.0,
            },
            "material": {
                "first_positive_plastic_strain": 1.0e-6,
            },
            "solver": {"mfront_threads": 1},
            "nonlocal_plasticity": {"enabled": False, "coupling_modulus_mpa": 0.0},
        },
        "inputs": {"synthetic": "same"},
        "layout": {
            "global_shape": [8, 6],
            "partition_shape": [1, 1],
            "padding": 0,
            "partitions": [
                {
                    "partition_id": 0,
                    "index": [0, 0],
                    "core_bounds": [0, 8, 0, 6],
                    "solve_bounds": [0, 8, 0, 6],
                    "core_shape": [8, 6],
                    "solve_shape": [8, 6],
                }
            ],
        },
    }
    manifest_path = campaign / "manifest.json"
    _json(manifest_path, manifest)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    x = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)[:, None]
    peeq = 0.01 + 0.005 * np.cos(x) * np.ones((1, 6))
    field_path = campaign / "partitions" / "0000" / "PEEQ.npy"
    field_path.parent.mkdir(parents=True)
    np.save(field_path, peeq)
    _json(
        field_path.with_name("status.json"),
        {
            "complete": True,
            "partition_id": 0,
            "manifest_sha256": manifest_sha,
            "outputs": {"PEEQ": fingerprint_file(field_path)},
            "diagnostics": {},
        },
    )
    _json(
        campaign / "HREF.json",
        {
            "partition_id": 0,
            "source_campaign_manifest_sha256": manifest_sha,
            "source_peeq_sha256": fingerprint_file(field_path),
            "reference_hardening_modulus_mpa": 4_000.0,
        },
    )
    config = tmp_path / "configs" / "identification.yaml"
    config.parent.mkdir()
    config.write_text(
        """
campaign:
  name: synthetic
  input: inputs
  output: output
  local_campaign: local
  partition_id: 0
  h_ref: local/HREF.json
  max_new_high_fidelity_runs: 5
  existing_high_fidelity: []
parameters:
  ell_um: {min: 20.0, max: 40.0, samples: 2}
  alpha: {min: 1.0, max: 2.0, samples: 2}
  h_ref_source: campaign_metadata
observation:
  grid_mapping: identity
  grid_reduction: 1
  spatial_filter: none
  use_core_mask_only: true
""",
        encoding="utf-8",
    )
    return config


def test_frozen_screen_is_cached_and_uses_one_local_point(tmp_path: Path) -> None:
    config = load_joint_identification_config(_synthetic_configuration(tmp_path))
    inspection = inspect_joint_identification(config)
    assert inspection["f0"]["helmholtz_solves"] == 2
    assert inspection["f0"]["parameter_pair_count"] == 5

    report = screen_frozen_field(config)
    assert report["status"] == "completed"
    with (config.output_directory / "f0" / "frozen_screen.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert sum(row["is_local"] == "True" for row in rows) == 1
    assert all(float(row["helmholtz_residual_relative"]) < 1.0e-11 for row in rows)

    reused = screen_frozen_field(config)
    assert reused["status"] == "reused"
    validation = json.loads(
        (config.output_directory / "f0" / "proxy_validation.json").read_text()
    )
    assert validation["status"] == "not_available"


def test_frozen_screen_dry_run_performs_no_output_write(tmp_path: Path) -> None:
    config = load_joint_identification_config(_synthetic_configuration(tmp_path))
    report = screen_frozen_field(config, dry_run=True)
    assert report["status"] == "dry_run"
    assert not config.output_directory.exists()


def test_configuration_rejects_local_alpha_in_positive_domain(tmp_path: Path) -> None:
    path = _synthetic_configuration(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        "alpha: {min: 1.0, max: 2.0, samples: 2}",
        "alpha: {min: 0.0, max: 2.0, samples: 2}",
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="start above zero"):
        load_joint_identification_config(path)
