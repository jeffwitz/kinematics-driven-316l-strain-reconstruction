from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fem_inhouse.workflows.multistep_path_dependence import (
    band_structure_ratio,
    compare_multistep_path_dependence,
    core_slice,
    descriptive_statistics,
    disagreement_fraction,
)

SOLVE = (100, 160, 200, 250)
CORE = (110, 150, 210, 240)


def test_core_slice_drops_the_padding() -> None:
    window = core_slice(SOLVE, CORE)

    assert window == (slice(10, 50), slice(10, 40))


def test_core_slice_rejects_a_core_outside_the_support() -> None:
    with pytest.raises(ValueError, match="inside the solve bounds"):
        core_slice(SOLVE, (90, 150, 210, 240))


def test_descriptive_statistics_reports_the_registered_quantities() -> None:
    stats = descriptive_statistics(np.arange(101, dtype=float))

    assert stats["mean"] == pytest.approx(50.0)
    assert stats["median"] == pytest.approx(50.0)
    assert stats["maximum"] == pytest.approx(100.0)
    assert stats["percentile_99"] == pytest.approx(99.0)


def test_band_structure_ratio_separates_diffuse_from_banded() -> None:
    reference = np.zeros((10, 10))
    reference[:2, :] = 1.0  # the top 20 % carries the localisation
    diffuse = np.full((10, 10), 0.5)
    banded = np.where(reference > 0.0, 1.0, 0.05)

    assert band_structure_ratio(diffuse, reference, top_fraction=0.2) == pytest.approx(1.0)
    assert band_structure_ratio(banded, reference, top_fraction=0.2) > 5.0


def test_disagreement_fraction_counts_activity_flips() -> None:
    first = np.array([[1.0, 0.0], [1.0, 0.0]])
    second = np.array([[1.0, 1.0], [0.0, 0.0]])

    assert disagreement_fraction(first, second, threshold=0.5) == pytest.approx(0.5)


def _write_run(directory: Path, mode: str, field: np.ndarray, increments: int = 40) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "PEEQ.npy", field)
    (directory / "report.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "partition_id": 43,
                "solve_bounds": list(SOLVE),
                "core_bounds": list(CORE),
                "config": {"solver": {"increments": increments}},
            }
        ),
        encoding="utf-8",
    )


def _fields(tmp_path: Path, *, excess: float) -> tuple[Path, Path, Path]:
    generator = np.random.default_rng(2)
    base = generator.random((60, 50)) * 1.0e-3
    measured_root = tmp_path / "measured"
    proportional_root = tmp_path / "proportional"
    _write_run(measured_root, "measured", base * (1.0 + excess))
    _write_run(proportional_root, "proportional", base)
    archived = tmp_path / "archived_PEEQ.npy"
    np.save(archived, base * 1.001)
    return measured_root, proportional_root, archived


def test_comparison_reports_a_material_difference(tmp_path: Path) -> None:
    measured, proportional, archived = _fields(tmp_path, excess=0.30)

    report = compare_multistep_path_dependence(
        measured_directory=measured,
        proportional_directory=proportional,
        archived_field_path=archived,
        output_directory=tmp_path / "out",
        figure_directory=tmp_path / "fig",
    )

    assert report["conclusion"]["relative_l2_band"] == "material"
    assert report["conclusion"]["control_separates"] is True
    assert report["conclusion"]["verdict"] == "material"
    assert report["path_dependence"]["signed_mean_difference"] > 0.0
    assert report["support"]["core_shape"] == [40, 30]
    assert (tmp_path / "out" / "pair_metrics.csv").is_file()
    assert (tmp_path / "fig" / "p0043_path_dependence_peeq.png").is_file()


def test_comparison_withdraws_when_the_control_does_not_separate(tmp_path: Path) -> None:
    generator = np.random.default_rng(5)
    base = generator.random((60, 50)) * 1.0e-3
    measured_root = tmp_path / "measured"
    proportional_root = tmp_path / "proportional"
    _write_run(measured_root, "measured", base * 1.02)
    _write_run(proportional_root, "proportional", base)
    archived = tmp_path / "archived_PEEQ.npy"
    np.save(archived, base * 1.02)  # discretisation as large as the path effect

    report = compare_multistep_path_dependence(
        measured_directory=measured_root,
        proportional_directory=proportional_root,
        archived_field_path=archived,
        output_directory=tmp_path / "out",
        figure_directory=tmp_path / "fig",
    )

    assert report["conclusion"]["control_separates"] is False
    assert report["conclusion"]["verdict"] == "withdrawn_discretisation_not_separated"


def test_comparison_rejects_a_mismatched_increment_count(tmp_path: Path) -> None:
    measured, proportional, archived = _fields(tmp_path, excess=0.1)
    report_path = proportional / "report.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    data["config"]["solver"]["increments"] = 20
    report_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="same increment count"):
        compare_multistep_path_dependence(
            measured_directory=measured,
            proportional_directory=proportional,
            archived_field_path=archived,
            output_directory=tmp_path / "out",
            figure_directory=tmp_path / "fig",
        )


def test_comparison_rejects_a_swapped_mode(tmp_path: Path) -> None:
    measured, proportional, archived = _fields(tmp_path, excess=0.1)

    with pytest.raises(ValueError, match="does not hold a measured run"):
        compare_multistep_path_dependence(
            measured_directory=proportional,
            proportional_directory=measured,
            archived_field_path=archived,
            output_directory=tmp_path / "out",
            figure_directory=tmp_path / "fig",
        )


def test_export_run_as_observation_campaign_builds_the_expected_layout(tmp_path: Path) -> None:
    from fem_inhouse.workflows.multistep_path_dependence import (
        export_run_as_observation_campaign,
    )

    run = tmp_path / "run"
    run.mkdir()
    displacement = np.zeros((5, 4, 2))
    np.save(run / "U.npy", displacement)
    digest = __import__("hashlib").sha256((run / "U.npy").read_bytes()).hexdigest()
    (run / "report.json").write_text(
        json.dumps(
            {
                "partition_id": 43,
                "status": "completed_local_measured_boundary_history",
                "mode": "measured",
                "solve_bounds": list(SOLVE),
                "core_bounds": list(CORE),
                "config": {"solver": {"increments": 40}},
                "outputs": {"U.npy": digest},
            }
        ),
        encoding="utf-8",
    )

    result = export_run_as_observation_campaign(
        run_directory=run, output_directory=tmp_path / "campaign", partition_id=43
    )

    manifest = json.loads((tmp_path / "campaign" / "manifest.json").read_text())
    status = json.loads(
        (tmp_path / "campaign" / "partitions" / "0043" / "status.json").read_text()
    )
    assert manifest["layout"]["partitions"][0]["solve_bounds"] == list(SOLVE)
    assert status["complete"] is True
    assert status["outputs"]["U"] == result["u_sha256"] == digest


def test_export_run_as_observation_campaign_rejects_a_tampered_field(tmp_path: Path) -> None:
    from fem_inhouse.workflows.multistep_path_dependence import (
        export_run_as_observation_campaign,
    )

    run = tmp_path / "run"
    run.mkdir()
    np.save(run / "U.npy", np.zeros((5, 4, 2)))
    (run / "report.json").write_text(
        json.dumps(
            {
                "partition_id": 43,
                "status": "x",
                "mode": "measured",
                "solve_bounds": list(SOLVE),
                "core_bounds": list(CORE),
                "config": {"solver": {"increments": 40}},
                "outputs": {"U.npy": "0" * 64},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match the run report"):
        export_run_as_observation_campaign(
            run_directory=run, output_directory=tmp_path / "campaign", partition_id=43
        )


def test_modal_filter_pins_ends_and_stays_within_noise(tmp_path: Path) -> None:
    import hashlib

    from fem_inhouse.workflows.dic_boundary_modal_filter import (
        filter_dic_boundary_history,
    )

    states, nx, ny = 12, 9, 8
    generator = np.random.default_rng(8)
    load = np.linspace(0.0, 1.0, states)[:, None, None]
    x = np.arange(nx, dtype=float)[None, :, None] * 0.00184
    y = np.arange(ny, dtype=float)[None, None, :] * 0.00184
    history = np.zeros((states, nx, ny, 2))
    history[..., 0] = -2.0e-3 * load * x
    history[..., 1] = 6.0e-3 * load * y
    history += generator.normal(0.0, 2.0e-7, history.shape)
    history[0] = 0.0

    source = tmp_path / "h.npy"
    np.save(source, history)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report = tmp_path / "r.json"
    report.write_text(
        json.dumps(
            {
                "outputs": {"h.npy": digest},
                "partition_id": 43,
                "solve_bounds": [0, nx - 1, 0, ny - 1],
                "core_bounds": [1, nx - 2, 1, ny - 2],
            }
        ),
        encoding="utf-8",
    )

    result = filter_dic_boundary_history(
        history_path=source,
        history_report_path=report,
        output_directory=tmp_path / "out",
        rank=3,
    )

    filtered = np.load(tmp_path / "out" / "modal_filtered_history_mm.npy")
    np.testing.assert_array_equal(filtered[0], history[0])
    np.testing.assert_array_equal(filtered[-1], history[-1])
    np.testing.assert_array_equal(filtered[:, 1:-1, 1:-1, :], history[:, 1:-1, 1:-1, :])
    assert result["filter"]["removed_within_measured_noise"] is True
    assert result["status"] == "completed_modal_boundary_filter"


def test_pin_endpoints_zeroes_both_ends_exactly() -> None:
    from fem_inhouse.workflows.dic_boundary_modal_filter import pin_endpoints

    generator = np.random.default_rng(1)
    pinned = pin_endpoints(generator.random((7, 5)))

    np.testing.assert_array_equal(pinned[0], np.zeros(5))
    np.testing.assert_array_equal(pinned[-1], np.zeros(5))


def test_endpoint_ramp_deviation_vanishes_at_both_ends() -> None:
    from fem_inhouse.workflows.dic_boundary_modal_filter import endpoint_ramp_deviation

    generator = np.random.default_rng(3)
    states = generator.random((9, 4))
    states[0] = 0.0
    ramp, deviation = endpoint_ramp_deviation(states)

    np.testing.assert_allclose(ramp[-1], states[-1])
    np.testing.assert_array_equal(deviation[-1], np.zeros(4))
