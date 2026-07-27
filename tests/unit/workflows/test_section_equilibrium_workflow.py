from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fem_inhouse.data_preparation import fingerprint_file
from fem_inhouse.workflows.section_equilibrium import (
    diagnose_section_equilibrium_campaigns,
)


def _write_campaign(path: Path) -> None:
    partition = path / "partitions" / "0000"
    partition.mkdir(parents=True)
    manifest = {
        "config": {"mesh": {"base_pixel_size_mm": 0.1, "scale_factor": 1.0}},
        "layout": {
            "global_shape": [4, 5],
            "partition_shape": [1, 1],
            "padding": 0,
            "partitions": [
                {
                    "partition_id": 0,
                    "index": [0, 0],
                    "core_bounds": [0, 4, 0, 5],
                    "core_shape": [4, 5],
                    "solve_bounds": [0, 4, 0, 5],
                    "solve_shape": [4, 5],
                }
            ],
        },
    }
    manifest_path = path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    stress = np.zeros((4, 5, 3), dtype=np.float64)
    stress[..., 1] = 100.0
    stress_path = partition / "S.npy"
    np.save(stress_path, stress)
    status = {
        "complete": True,
        "partition_id": 0,
        "manifest_sha256": fingerprint_file(manifest_path),
        "outputs": {"S": fingerprint_file(stress_path)},
    }
    (partition / "status.json").write_text(json.dumps(status), encoding="utf-8")


def test_workflow_writes_verified_report_profiles_and_figure(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    output = tmp_path / "diagnostic"

    report = diagnose_section_equilibrium_campaigns(
        (("baseline", campaign),),
        partition_id=0,
        output_directory=output,
        thickness_mm=2.0,
    )

    assert report["status"] == "completed_baseline_no_acceptance_threshold"
    assert report["case_count"] == 1
    assert len(report["metrics"]) == 2
    assert (output / "report.json").is_file()
    assert (output / "metrics.csv").is_file()
    assert (output / "profiles" / "baseline.csv").is_file()
    assert (output / "figures" / "baseline.png").is_file()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnose_section_equilibrium_campaigns(
            (("baseline", campaign),),
            partition_id=0,
            output_directory=output,
            thickness_mm=2.0,
        )


def test_workflow_rejects_a_corrupt_stress_field(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    _write_campaign(campaign)
    stress_path = campaign / "partitions" / "0000" / "S.npy"
    values = np.load(stress_path)
    values[0, 0, 0] = 1.0
    np.save(stress_path, values)

    with pytest.raises(RuntimeError, match="fails its status hash"):
        diagnose_section_equilibrium_campaigns(
            (("baseline", campaign),),
            partition_id=0,
            output_directory=tmp_path / "diagnostic",
            thickness_mm=2.0,
        )
