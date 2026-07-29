from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fem_inhouse.workflows.dic_photometric_quality import (
    diagnose_dic_photometric_quality,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replay(
    root: Path,
    *,
    label: str,
    dic: np.ndarray,
    observed: np.ndarray,
) -> Path:
    replay = root / label
    replay.mkdir()
    np.save(replay / "dic_evm.npy", dic)
    np.save(replay / "fem_observed_evm.npy", observed)
    report = {
        "status": "completed_symmetric_image_observation",
        "partition_id": 43,
        "core_bounds": [0, dic.shape[0], 0, dic.shape[1]],
        "profile": {"name": "legacy_script_2021"},
        "outputs": {
            "dic_evm.npy": _hash(replay / "dic_evm.npy"),
            "fem_observed_evm.npy": _hash(replay / "fem_observed_evm.npy"),
        },
    }
    (replay / "report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return replay


@pytest.mark.measurement
def test_photometric_quality_workflow_is_traceable_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    pytest.importorskip("cv2")
    shape = (12, 10)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    np.save(prepared / "displacement_x_mm.npy", np.zeros((13, 11)))
    np.save(prepared / "displacement_y_mm.npy", np.zeros((13, 11)))
    (prepared / "manifest.json").write_text("{}\n", encoding="utf-8")

    nodal_shape = (shape[0] + 1, shape[1] + 1)
    reference = np.full(
        (400 + nodal_shape[0], 1211 + nodal_shape[1]),
        100,
        dtype=np.uint8,
    )
    current = reference.copy()
    nodal_ramp = np.arange(np.prod(nodal_shape), dtype=np.uint8).reshape(nodal_shape) // 5
    current[
        400 : 400 + nodal_shape[0],
        1211 : 1211 + nodal_shape[1],
    ] += nodal_ramp
    ramp = 0.25 * (
        nodal_ramp[:-1, :-1]
        + nodal_ramp[1:, :-1]
        + nodal_ramp[:-1, 1:]
        + nodal_ramp[1:, 1:]
    )
    reference_path = tmp_path / "reference.tif"
    current_path = tmp_path / "current.tif"
    Image.fromarray(reference).save(reference_path)
    Image.fromarray(current).save(current_path)

    coordinates = np.indices(shape, dtype=float)
    dic = 0.01 + 1.0e-5 * (coordinates[0] + coordinates[1])
    local = _replay(
        tmp_path,
        label="local",
        dic=dic,
        observed=dic + ramp * 1.0e-4,
    )
    coupled = _replay(
        tmp_path,
        label="a100",
        dic=dic,
        observed=dic + ramp * 0.5e-4,
    )
    output = tmp_path / "output"
    figures = tmp_path / "figures"

    report = diagnose_dic_photometric_quality(
        reference_image=reference_path,
        final_image=current_path,
        prepared_case=prepared,
        replays=(("local", 0.0, local), ("a100", 1.0, coupled)),
        output_directory=output,
        figure_directory=figures,
    )

    assert report["status"] == "completed_baseline_no_acceptance_threshold"
    assert report["mechanics_rerun"] is False
    assert report["micromorphic_identification_run"] is False
    assert [row["alpha"] for row in report["rows"]] == [0.0, 1.0]
    assert report["rows"][0]["association"]["pearson"] > 0.99
    assert 0.9 <= report["sensitivity_mask"]["retained_fraction"] <= 0.92
    assert (output / "decile_metrics.csv").is_file()
    assert (figures / "photometric_quality_and_error.png").is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnose_dic_photometric_quality(
            reference_image=reference_path,
            final_image=current_path,
            prepared_case=prepared,
            replays=(("local", 0.0, local),),
            output_directory=output,
            figure_directory=figures,
        )


@pytest.mark.measurement
def test_photometric_quality_rejects_corrupted_replay(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    shape = (3, 4)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    np.save(prepared / "displacement_x_mm.npy", np.zeros((4, 5)))
    np.save(prepared / "displacement_y_mm.npy", np.zeros((4, 5)))
    (prepared / "manifest.json").write_text("{}\n", encoding="utf-8")
    image = np.zeros((404, 1216), dtype=np.uint8)
    image_path = tmp_path / "image.tif"
    Image.fromarray(image).save(image_path)
    replay = _replay(
        tmp_path,
        label="local",
        dic=np.ones(shape),
        observed=np.ones(shape),
    )
    with (replay / "fem_observed_evm.npy").open("ab") as stream:
        stream.write(b"corruption")

    with pytest.raises(ValueError, match="immutable replay"):
        diagnose_dic_photometric_quality(
            reference_image=image_path,
            final_image=image_path,
            prepared_case=prepared,
            replays=(("local", 0.0, replay),),
            output_directory=tmp_path / "output",
            figure_directory=tmp_path / "figures",
        )
