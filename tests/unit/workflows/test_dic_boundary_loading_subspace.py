from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fem_inhouse.workflows.dic_boundary_loading_subspace import (
    boundary_mask,
    diagnose_dic_boundary_loading_subspace,
    robust_sigma,
    robust_z_scores,
    second_time_difference,
    temporal_noise_estimate,
    temporal_roughness,
)

PIXEL = 0.00184


def test_second_time_difference_annihilates_a_straight_path() -> None:
    path = np.arange(9, dtype=float)[:, None] * np.array([2.0, -3.0])

    np.testing.assert_allclose(second_time_difference(path), 0.0, atol=1.0e-14)


def test_second_time_difference_rejects_a_short_history() -> None:
    with pytest.raises(ValueError, match="three states"):
        second_time_difference(np.zeros((2, 4)))


def test_robust_sigma_ignores_a_single_outlier() -> None:
    rng = np.random.default_rng(23)
    clean = rng.normal(0.0, 1.0, 401)
    contaminated = clean.copy()
    contaminated[100] = 1.0e4

    assert robust_sigma(clean) == pytest.approx(1.0, rel=0.10)
    assert robust_sigma(contaminated) == pytest.approx(robust_sigma(clean), rel=0.05)
    assert float(np.std(contaminated)) > 100.0 * float(np.std(clean))


def test_temporal_roughness_separates_white_from_smooth() -> None:
    rng = np.random.default_rng(3)
    white = rng.normal(size=400)
    smooth = np.linspace(0.0, 1.0, 400) ** 2

    assert temporal_roughness(white) == pytest.approx(1.0, rel=0.15)
    assert temporal_roughness(smooth) < 0.01


def test_temporal_noise_estimate_recovers_injected_noise() -> None:
    rng = np.random.default_rng(11)
    sigma_mm = 3.0e-5
    path = np.linspace(0.0, 1.0, 60)[:, None] * np.full(200, 2.0e-3)
    observed = path + rng.normal(0.0, sigma_mm, path.shape)

    estimate = temporal_noise_estimate(observed, pixel_size_mm=PIXEL)

    assert estimate.rms_mm == pytest.approx(sigma_mm, rel=0.05)
    assert estimate.robust_mm == pytest.approx(sigma_mm, rel=0.10)
    assert estimate.rms_px == pytest.approx(sigma_mm / PIXEL, rel=0.05)


def test_temporal_noise_estimate_selector_drops_degenerate_states() -> None:
    rng = np.random.default_rng(5)
    observed = rng.normal(0.0, 1.0, (12, 50))
    observed[5] = 0.5 * (observed[4] + observed[6])  # interpolated, d2 is zero
    centred = np.arange(1, 11)

    with_repaired = temporal_noise_estimate(observed, pixel_size_mm=PIXEL)
    without = temporal_noise_estimate(
        observed, pixel_size_mm=PIXEL, selector=~np.isin(centred, [5])
    )

    assert without.robust_mm > with_repaired.robust_mm


def test_robust_z_scores_still_scores_excluded_entries() -> None:
    values = np.array([1.0, -1.0, 1.0, -1.0, 0.0, 20.0])
    mask = np.array([True, True, True, True, True, False])

    scores = robust_z_scores(values, scale_mask=mask)

    assert abs(scores[-1]) > 10.0
    assert scores.shape == values.shape


def test_boundary_mask_selects_the_perimeter() -> None:
    mask = boundary_mask((5, 4))

    assert np.count_nonzero(mask) == 2 * (5 + 4) - 4
    assert not mask[1:-1, 1:-1].any()


def _write_history(directory: Path, history: np.ndarray) -> tuple[Path, Path]:
    history_path = directory / "history.npy"
    np.save(history_path, history)
    digest = hashlib.sha256(history_path.read_bytes()).hexdigest()
    report_path = directory / "history_report.json"
    report_path.write_text(
        json.dumps(
            {
                "outputs": {"history.npy": digest},
                "partition_id": 43,
                "solve_bounds": [0, history.shape[1] - 1, 0, history.shape[2] - 1],
                "core_bounds": [1, history.shape[1] - 2, 1, history.shape[2] - 2],
                "repair": {"corrupted_states": [4]},
            }
        ),
        encoding="utf-8",
    )
    return history_path, report_path


def _synthetic_history(states: int = 14, nx: int = 11, ny: int = 9) -> np.ndarray:
    rng = np.random.default_rng(19)
    load = np.linspace(0.0, 1.0, states)[:, None, None]
    x = np.arange(nx, dtype=float)[None, :, None] * PIXEL
    y = np.arange(ny, dtype=float)[None, None, :] * PIXEL
    history = np.zeros((states, nx, ny, 2))
    history[..., 0] = -2.0e-3 * load * x
    history[..., 1] = 6.0e-3 * load * y
    history += rng.normal(0.0, 1.0e-6, history.shape)
    history[4] = 0.5 * (history[3] + history[5])
    return history


def test_diagnose_boundary_loading_subspace_finds_one_smooth_mode(tmp_path: Path) -> None:
    history_path, report_path = _write_history(tmp_path, _synthetic_history())

    report = diagnose_dic_boundary_loading_subspace(
        history_path=history_path,
        history_report_path=report_path,
        output_directory=tmp_path / "out",
        figure_directory=tmp_path / "fig",
    )

    subspace = report["loading_subspace"]
    assert subspace["low_dimensional_model_supported"] is True
    assert subspace["leading_energy_fraction"][0] > 0.99
    assert subspace["leading_roughness"][0] < 0.1
    assert report["affine_noise"]["excluded_repaired_states"] == [4]
    assert report["mechanics_rerun"] is False
    assert report["history_modified"] is False
    assert (tmp_path / "out" / "state_metrics.csv").is_file()
    assert (tmp_path / "fig" / "p0043_boundary_loading_subspace.png").is_file()


def test_diagnose_boundary_loading_subspace_scores_an_injected_outlier(tmp_path: Path) -> None:
    history = _synthetic_history()
    history[8] += 4.0e-5  # coherent jump of one state only
    history_path, report_path = _write_history(tmp_path, history)

    report = diagnose_dic_boundary_loading_subspace(
        history_path=history_path,
        history_report_path=report_path,
        output_directory=tmp_path / "out",
        figure_directory=tmp_path / "fig",
    )

    assert report["outliers"]["largest_loading_z_state"] == 8


def test_diagnose_boundary_loading_subspace_rejects_a_tampered_history(tmp_path: Path) -> None:
    history_path, report_path = _write_history(tmp_path, _synthetic_history())
    np.save(history_path, _synthetic_history() * 1.5)

    with pytest.raises(ValueError, match="immutable report"):
        diagnose_dic_boundary_loading_subspace(
            history_path=history_path,
            history_report_path=report_path,
            output_directory=tmp_path / "out",
            figure_directory=tmp_path / "fig",
        )
