from __future__ import annotations

import json
from pathlib import Path


def test_sequential_twin_artifact_records_a_negative_geometry_gate() -> None:
    root = Path(__file__).resolve().parents[3]
    report = json.loads(
        (
            root
            / "validation/reference_data/srix_regm_sequential_one_newton_v2/report.json"
        ).read_text()
    )
    spectrum = report["geometry"]["normalized_singular_values"]
    assert len(spectrum) == 4
    assert spectrum[0] == 1.0
    assert spectrum[2] < 0.1
    assert report["claims"]["p43_authorized"] is False


def test_cumulative_endpoint_observable_is_recorded_and_remains_negative() -> None:
    root = Path(__file__).resolve().parents[3]
    report = json.loads(
        (
            root
            / "validation/reference_data/srix_regm_sequential_one_newton_v3/report.json"
        ).read_text()
    )
    cumulative = report["geometries"]["SREGM_cumulative_displacement"]
    assert report["claims"]["cumulative_observable_tested"] is True
    assert cumulative["normalized_singular_values"][0] == 1.0
    assert cumulative["normalized_singular_values"][2] < 0.1
    assert report["claims"]["p43_authorized"] is False
