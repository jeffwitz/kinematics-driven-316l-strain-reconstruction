import csv
import json
from dataclasses import asdict

import numpy as np
import pytest

from fem_inhouse.data_preparation import fingerprint_file
from fem_inhouse.partitioning import PartitionLayout
from fem_inhouse.postprocessing import field_error_metrics, helmholtz_filter_element_field
from fem_inhouse.workflows.nonlocality_diagnostic import (
    DecisionThresholds,
    load_decision_thresholds,
    normalize_length_scales,
    run_field_sweep,
    run_nonlocality_diagnostic,
)


def test_length_normalization_adds_zero_converts_deduplicates_and_sorts() -> None:
    lengths = normalize_length_scales(
        [4.0, 0.0, 2.0, 2.0],
        unit="pixels",
        pixel_size_mm=0.00184,
    )
    assert [item.length_pixels for item in lengths] == [0.0, 2.0, 4.0]
    assert lengths[1].length_mm == pytest.approx(0.00368)
    assert lengths[1].length_um == pytest.approx(3.68)

    with pytest.raises(ValueError, match="nonnegative"):
        normalize_length_scales([-1.0], unit="um", pixel_size_mm=0.001)
    with pytest.raises(ValueError, match="finite"):
        normalize_length_scales([np.nan], unit="mm", pixel_size_mm=0.001)


def test_synthetic_sweep_recovers_known_helmholtz_length() -> None:
    source = np.zeros((41, 39))
    source[20, 19] = 1.0
    known_length = 0.2
    dic = helmholtz_filter_element_field(
        source,
        length_scale_mm=known_length,
        spacing_x_mm=0.1,
        spacing_y_mm=0.1,
    ).filtered_element_field
    lengths = normalize_length_scales(
        [0.0, 0.1, known_length, 0.3],
        unit="mm",
        pixel_size_mm=0.1,
    )

    sweep = run_field_sweep(
        dic_evm_reference=dic,
        fem_evm_raw=source,
        lengths=lengths,
        spacing_x_mm=0.1,
        spacing_y_mm=0.1,
        core_slice=(slice(5, 36), slice(5, 34)),
        minimum_artificial_padding_pixels=5,
        minimum_padding_length_ratio=2.0,
    )

    assert sweep.selection["best_by_rmse_mm"] == known_length
    known_record = next(
        record
        for record in sweep.metrics
        if record["field"] == "EVM_HISTORICAL" and record["length_mm"] == known_length
    )
    assert known_record["rmse"] < 1e-14
    np.testing.assert_array_equal(sweep.filtered_evm[0.0], source)


def test_sweep_marks_insufficient_padding_and_keeps_peeq_amplitude_separate() -> None:
    x = np.linspace(0.0, 1.0, 12)[:, None]
    field = np.broadcast_to(x, (12, 10))
    lengths = normalize_length_scales(
        [0.0, 0.1, 0.4],
        unit="mm",
        pixel_size_mm=0.1,
    )
    sweep = run_field_sweep(
        dic_evm_reference=field,
        fem_evm_raw=field,
        peeq_raw=field**2,
        lengths=lengths,
        spacing_x_mm=0.1,
        spacing_y_mm=0.1,
        core_slice=(slice(2, 10), slice(2, 8)),
        minimum_artificial_padding_pixels=5,
        minimum_padding_length_ratio=4.0,
    )

    long_evm = next(
        record
        for record in sweep.metrics
        if record["field"] == "EVM_HISTORICAL" and record["length_mm"] == 0.4
    )
    peeq_baseline = next(
        record
        for record in sweep.metrics
        if record["field"] == "PEEQ" and record["length_mm"] == 0.0
    )
    assert long_evm["boundary_status"] == "boundary_contaminated"
    assert peeq_baseline["rmse"] is None
    assert peeq_baseline["relative_l2"] is None
    assert peeq_baseline["iou_top_10pct"] is not None


def test_confirmatory_threshold_file_and_predeclared_decision(tmp_path) -> None:
    path = tmp_path / "thresholds.yaml"
    path.write_text(
        """
decision_thresholds:
  minimum_correlation_gain: 0.0
  minimum_relative_l2_reduction: 0.0
  minimum_iou_gain: 0.0
  maximum_relative_mean_drift: 1.0e-10
""",
        encoding="utf-8",
    )
    thresholds = load_decision_thresholds(path)
    assert thresholds == DecisionThresholds(0.0, 0.0, 0.0, 1e-10)

    source = np.zeros((9, 9))
    source[4, 4] = 1.0
    lengths = normalize_length_scales([0.0, 0.1], unit="mm", pixel_size_mm=0.1)
    sweep = run_field_sweep(
        dic_evm_reference=helmholtz_filter_element_field(
            source,
            length_scale_mm=0.1,
            spacing_x_mm=0.1,
            spacing_y_mm=0.1,
        ).filtered_element_field,
        fem_evm_raw=source,
        lengths=lengths,
        spacing_x_mm=0.1,
        spacing_y_mm=0.1,
        core_slice=(slice(1, 8), slice(1, 8)),
        minimum_artificial_padding_pixels=5,
        mode="confirmatory",
        decision_thresholds=thresholds,
    )
    assert sweep.selection["criteria_met_on_this_partition"]
    assert sweep.selection["automatic_physical_conclusion"] is None


def _write_synthetic_campaign(tmp_path):
    nx, ny = 8, 6
    spacing = 0.1
    layout = PartitionLayout((nx, ny), (2, 1), padding=2)
    partition = layout.get(0)
    input_directory = tmp_path / "input"
    campaign = tmp_path / "campaign"
    partition_directory = campaign / "partitions" / "0000"
    input_directory.mkdir()
    partition_directory.mkdir(parents=True)

    x = np.arange(nx + 1, dtype=float)[:, None] * spacing
    y = np.arange(ny + 1, dtype=float)[None, :] * spacing
    dic_x = np.broadcast_to(0.01 * x + 0.001 * np.sin(8.0 * x), (nx + 1, ny + 1))
    dic_y = np.broadcast_to(-0.003 * y, (nx + 1, ny + 1))
    np.save(input_directory / "displacement_x_mm.npy", dic_x)
    np.save(input_directory / "displacement_y_mm.npy", dic_y)
    (input_directory / "manifest.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding="utf-8",
    )

    solve_nodes = partition.solve_node_slice_global
    fem_x = np.asarray(dic_x[solve_nodes]).copy()
    fem_y = np.asarray(dic_y[solve_nodes]).copy()
    local_x = np.arange(fem_x.shape[0])[:, None]
    fem_x += 0.0004 * np.exp(-np.square(local_x - 3.0) / 0.5)
    displacement = np.stack((fem_x, fem_y), axis=-1)
    peeq = np.exp(
        -(
            np.square(np.arange(partition.solve_shape[0])[:, None] - 3.0)
            + np.square(np.arange(partition.solve_shape[1])[None, :] - 3.0)
        )
        / 2.0
    )
    u_path = partition_directory / "U.npy"
    peeq_path = partition_directory / "PEEQ.npy"
    np.save(u_path, displacement)
    np.save(peeq_path, peeq)
    status = {
        "complete": True,
        "partition_id": 0,
        "outputs": {
            "U": fingerprint_file(u_path),
            "PEEQ": fingerprint_file(peeq_path),
        },
    }
    (partition_directory / "status.json").write_text(
        json.dumps(status),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "config": {
            "mesh": {
                "nx": nx,
                "ny": ny,
                "base_pixel_size_mm": spacing,
                "scale_factor": 1.0,
            },
            "material": {"poisson_ratio": 0.3},
        },
        "layout": layout.as_dict(),
    }
    (campaign / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return input_directory, campaign, partition


def test_complete_workflow_uses_metadata_core_and_preserves_zero_baseline(tmp_path) -> None:
    input_directory, campaign, partition = _write_synthetic_campaign(tmp_path)
    output = tmp_path / "diagnostic"
    report = run_nonlocality_diagnostic(
        input_directory=input_directory,
        campaign_directory=campaign,
        partition_id=0,
        output_directory=output,
        length_values=[0.0, 1.0, 2.0],
        length_unit="pixels",
        include_peeq=True,
        save_fields="all",
    )

    assert report["status"] == "completed"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["partition"]["core_bounds"] == list(partition.core_bounds)
    assert manifest["partition"]["solve_bounds"] == list(partition.solve_bounds)
    assert manifest["observable"]["peeq_amplitude_compared_to_dic_evm"] is False
    assert np.array_equal(
        np.load(output / "fields" / "evm_fe_raw.npy"),
        np.load(output / "fields" / "evm_fe_ell_000000um.npy"),
    )
    assert (output / "figures" / "metric_curves.svg").is_file()
    assert (output / "figures" / "best_candidates.png").is_file()

    with (output / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    baseline = next(
        row for row in rows if row["field"] == "EVM_HISTORICAL" and row["length_mm"] == "0.0"
    )
    dic = np.load(output / "fields" / "evm_dic_reference.npy")
    fem = np.load(output / "fields" / "evm_fe_raw.npy")
    direct = field_error_metrics(
        dic[partition.core_element_slice_local],
        fem[partition.core_element_slice_local],
    )
    assert float(baseline["rmse"]) == pytest.approx(direct.rmse)

    with pytest.raises(FileExistsError, match="not empty"):
        run_nonlocality_diagnostic(
            input_directory=input_directory,
            campaign_directory=campaign,
            partition_id=0,
            output_directory=output,
            length_values=[0.0],
            length_unit="pixels",
        )


def test_decision_threshold_validation() -> None:
    valid = {
        "minimum_correlation_gain": 0.0,
        "minimum_relative_l2_reduction": 0.0,
        "minimum_iou_gain": 0.0,
        "maximum_relative_mean_drift": 1e-10,
    }
    assert asdict(DecisionThresholds(**valid)) == valid
    with pytest.raises(ValueError, match="minimum_iou_gain"):
        DecisionThresholds(**(valid | {"minimum_iou_gain": -1.0}))
