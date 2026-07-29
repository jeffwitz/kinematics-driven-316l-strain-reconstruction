from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fem_inhouse.identification.observation import DICObservationOperatorConfig
from fem_inhouse.workflows.dic_observation_replay import replay_dic_observation


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path: Path) -> tuple[Path, Path, Path]:
    campaign = tmp_path / "campaign"
    partition = campaign / "partitions/0043"
    partition.mkdir(parents=True)
    size = 96
    coordinates = np.indices((size, size), dtype=float)
    displacement = np.zeros((size, size, 2), dtype=np.float64)
    displacement[..., 0] = 1.0e-5 * coordinates[0]
    displacement[..., 1] = 2.0e-5 * coordinates[1]
    np.save(partition / "U.npy", displacement)
    (campaign / "manifest.json").write_text(
        json.dumps(
            {
                "layout": {
                    "partitions": [
                        {
                            "partition_id": 43,
                            "core_bounds": [0, size - 1, 0, size - 1],
                            "solve_bounds": [0, size - 1, 0, size - 1],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (partition / "status.json").write_text(
        json.dumps(
            {
                "complete": True,
                "outputs": {"U": _hash(partition / "U.npy")},
            }
        ),
        encoding="utf-8",
    )
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    np.save(prepared / "displacement_x_mm.npy", displacement[..., 0])
    np.save(prepared / "displacement_y_mm.npy", displacement[..., 1])
    (prepared / "manifest.json").write_text("{}\n", encoding="utf-8")

    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(400 + size, 1211 + size), dtype=np.uint8)
    image_path = tmp_path / "reference.tif"
    Image.fromarray(image).save(image_path)
    return campaign, prepared, image_path


def test_synthetic_observation_configuration_is_in_cache_fingerprint() -> None:
    direct = DICObservationOperatorConfig()
    synthetic = DICObservationOperatorConfig(
        mode="synthetic_disflow",
        disflow_profile="legacy_script_2021",
        warp_mode="iterative_forward_inverse",
        mask_mode="declared_all_valid",
    )
    assert direct.fingerprint() != synthetic.fingerprint()
    with pytest.raises(ValueError, match="requires profile"):
        DICObservationOperatorConfig(mode="synthetic_disflow")


@pytest.mark.measurement
def test_small_replay_is_traceable_and_refuses_overwrite(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    campaign, prepared, image = _case(tmp_path)
    output = tmp_path / "output"

    report = replay_dic_observation(
        campaign=campaign,
        prepared_case=prepared,
        reference_image=image,
        partition_id=43,
        profile_name="legacy_script_2021",
        output_directory=output,
    )

    assert report["status"] == "completed_symmetric_image_observation"
    assert report["evm_post_filter_applied"] is False
    assert report["mask"]["mode"] == "declared_all_valid"
    assert np.load(output / "fem_observed_evm.npy").shape == (95, 95)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        replay_dic_observation(
            campaign=campaign,
            prepared_case=prepared,
            reference_image=image,
            partition_id=43,
            profile_name="legacy_script_2021",
            output_directory=output,
        )


@pytest.mark.measurement
def test_replay_rejects_corrupted_source_displacement(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    campaign, prepared, image = _case(tmp_path)
    with (campaign / "partitions/0043/U.npy").open("ab") as stream:
        stream.write(b"corruption")

    with pytest.raises(ValueError, match="immutable campaign status"):
        replay_dic_observation(
            campaign=campaign,
            prepared_case=prepared,
            reference_image=image,
            partition_id=43,
            profile_name="legacy_script_2021",
            output_directory=tmp_path / "output",
        )
