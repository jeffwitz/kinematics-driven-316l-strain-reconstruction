from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from fem_inhouse.workflows.ebsd_structural_length import measure_ebsd_structural_length


def _write_synthetic_ebsd(path: Path) -> None:
    rng = np.random.default_rng(15)
    white = rng.normal(size=(256, 224))
    fx = np.fft.fftfreq(256)[:, None]
    fy = np.fft.rfftfreq(224)[None, :]
    filtered = np.fft.irfftn(
        np.fft.rfftn(white) * np.exp(-0.5 * ((fx / 0.018) ** 2 + (fy / 0.025) ** 2)),
        s=white.shape,
        axes=(0, 1),
    )
    schmid = 0.25 + 0.08 * filtered / np.std(filtered)
    schmid = np.clip(schmid, 0.01, 0.49)
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("/schmid/max_schmid_factor", data=schmid)
        dataset.attrs["pixel_size_um"] = 1.84
        for name in ("phi1", "Phi", "phi2"):
            handle.create_dataset(f"/orientation/{name}", data=np.ones_like(schmid))


def test_workflow_writes_report_profiles_and_figure(tmp_path: Path) -> None:
    input_path = tmp_path / "synthetic.h5"
    output = tmp_path / "report"
    _write_synthetic_ebsd(input_path)

    report = measure_ebsd_structural_length(
        input_path,
        output,
        bootstrap_samples=200,
    )

    assert report["input"]["dataset"] == "/schmid/max_schmid_factor"
    assert report["lengths"]["radial_decay"]["length_um"] > 0.0
    assert report["bootstrap_block_median"]["valid_block_count"] > 0
    for name in (
        "report.json",
        "radial_profile.csv",
        "direction_x_profile.csv",
        "direction_y_profile.csv",
        "correlation_profiles.png",
    ):
        assert (output / name).is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        measure_ebsd_structural_length(input_path, output)
