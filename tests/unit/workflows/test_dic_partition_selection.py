from __future__ import annotations

import json

import numpy as np

from fem_inhouse.workflows.dic_partition_selection import (
    scan_dic_partition_heterogeneity,
    write_dic_partition_heterogeneity_report,
)


def test_scan_ranks_dic_partitions_and_records_indicators(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    x, y = np.meshgrid(np.arange(8.0), np.arange(8.0), indexing="ij")
    displacement_x = 0.001 * x**2
    displacement_y = 0.001 * y**2
    displacement_x[2:4, 5:7] += 0.2
    np.save(inputs / "displacement_x_mm.npy", displacement_x)
    np.save(inputs / "displacement_y_mm.npy", displacement_y)

    report = scan_dic_partition_heterogeneity(
        input_directory=inputs,
        parts_x=2,
        parts_y=2,
        padding=0,
        spacing_x_mm=1.0,
        spacing_y_mm=1.0,
    )

    assert report["observable"].startswith("EVM_HISTORICAL")
    assert len(report["partitions"]) == 4
    assert report["selection_indicator"] == "dic_band_morphology_score_q85"
    assert "q95_minus_q50_over_iqr" in report["partitions"][0]
    assert "band_aspect_ratio" in report["partitions"][0]
    assert report["partitions"][0]["band_score"] >= report["partitions"][-1]["band_score"]


def test_write_dic_partition_heterogeneity_report(tmp_path) -> None:
    output = tmp_path / "selection.json"
    report = {"partitions": [{"partition_id": 3}], "selection_indicator": "test"}

    write_dic_partition_heterogeneity_report(report, output)

    assert json.loads(output.read_text()) == report
