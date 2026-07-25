from __future__ import annotations

import json

import numpy as np
import pytest

from fem_inhouse.data_preparation import fingerprint_file
from fem_inhouse.partitioning import PartitionLayout
from fem_inhouse.workflows.coupled_nonlocal_validation import (
    validate_coupled_nonlocal_campaign,
)


def _write_campaign(
    directory,
    *,
    layout: PartitionLayout,
    displacement: np.ndarray,
    enabled: bool,
    coupling_modulus_mpa: float | None = None,
) -> None:
    partition_directory = directory / "partitions" / "0000"
    partition_directory.mkdir(parents=True)
    fields = {
        "U": displacement,
        "S_3D": np.zeros((4, 4, 3, 3)),
        "PLANE_STRESS_RESIDUAL_MPA": np.zeros((4, 4, 3)),
        "PEEQ": np.full((4, 4), 0.01),
        "PEEQ_NONLOCAL": np.full((4, 4), 0.009),
        "PEEQ_MISMATCH": np.full((4, 4), 0.001),
        "NONLOCAL_HARDENING_MPA": np.full((4, 4), 3.0),
        "YIELD_SURFACE_RADIUS_MPA": np.full((4, 4), 300.0),
        "NONLOCAL_RESIDUAL": np.zeros((4, 4)),
    }
    hashes = {}
    for name, values in fields.items():
        path = partition_directory / f"{name}.npy"
        np.save(path, values)
        hashes[name] = fingerprint_file(path)
    manifest = {
        "inputs": {
            "displacement_x_mm": "same-x",
            "displacement_y_mm": "same-y",
            "yield_stress_mpa": "same-yield",
            "hardening_coefficient_mpa": "same-hardening",
        },
        "layout": layout.as_dict(),
        "config": {
            "mesh": {
                "nx": 4,
                "ny": 4,
                "base_pixel_size_mm": 1.0,
                "scale_factor": 1.0,
            },
            "material": {"poisson_ratio": 0.3},
            "solver": {
                "increments": 2,
                "constitutive_backend": "mfront-native-plane-stress",
                "mfront_threads": 1 if not enabled else 2,
            },
            "nonlocal_plasticity": {
                "enabled": enabled,
                "length_scale_mm": 0.05,
                "coupling_modulus_mpa": (
                    3000.0 if enabled else 0.0
                )
                if coupling_modulus_mpa is None
                else coupling_modulus_mpa,
            },
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (partition_directory / "status.json").write_text(
        json.dumps(
            {
                "complete": True,
                "outputs": hashes,
                "diagnostics": {"cutbacks": 0},
            }
        )
    )


def test_raw_coupled_validation_uses_core_and_records_no_post_filter(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    coordinates = np.arange(5, dtype=float)
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    dic = np.stack((0.01 * x**2 + 0.002 * x * y, 0.008 * y**2), axis=-1)
    np.save(inputs / "displacement_x_mm.npy", dic[..., 0])
    np.save(inputs / "displacement_y_mm.npy", dic[..., 1])
    layout = PartitionLayout((4, 4), (1, 1), padding=0)
    local = tmp_path / "local"
    coupled = tmp_path / "coupled"
    _write_campaign(local, layout=layout, displacement=0.6 * dic, enabled=False)
    _write_campaign(coupled, layout=layout, displacement=0.9 * dic, enabled=True)
    output = tmp_path / "validation.json"

    report = validate_coupled_nonlocal_campaign(
        input_directory=inputs,
        local_campaign_directory=local,
        coupled_campaign_directory=coupled,
        partition_id=0,
        output_path=output,
    )

    assert report["comparison_contract"]["post_filter_applied"] is False
    assert report["comparison_contract"]["mechanical_solution_modified_by_candidate"] is True
    assert report["metrics"]["coupled"]["relative_l2_error"] < report["metrics"]["local"][
        "relative_l2_error"
    ]
    assert report["mechanical_checks"]["maximum_plane_stress_residual_mpa"] == 0.0
    assert json.loads(output.read_text())["partition_id"] == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validate_coupled_nonlocal_campaign(
            input_directory=inputs,
            local_campaign_directory=local,
            coupled_campaign_directory=coupled,
            partition_id=0,
            output_path=output,
        )


def test_raw_coupled_validation_rejects_mismatched_layouts(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    displacement = np.zeros((5, 5, 2))
    np.save(inputs / "displacement_x_mm.npy", displacement[..., 0])
    np.save(inputs / "displacement_y_mm.npy", displacement[..., 1])
    local = tmp_path / "local"
    coupled = tmp_path / "coupled"
    _write_campaign(
        local,
        layout=PartitionLayout((4, 4), (1, 1), padding=0),
        displacement=displacement,
        enabled=False,
    )
    _write_campaign(
        coupled,
        layout=PartitionLayout((4, 4), (1, 1), padding=1),
        displacement=displacement,
        enabled=True,
    )

    with pytest.raises(ValueError, match="same partition layout"):
        validate_coupled_nonlocal_campaign(
            input_directory=inputs,
            local_campaign_directory=local,
            coupled_campaign_directory=coupled,
            partition_id=0,
            output_path=tmp_path / "report.json",
        )


def test_hchi_zero_campaign_is_accepted_as_mechanically_local_reference(
    tmp_path,
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    coordinates = np.arange(5, dtype=float)
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    displacement = np.stack((0.01 * x**2, 0.01 * y**2), axis=-1)
    np.save(inputs / "displacement_x_mm.npy", displacement[..., 0])
    np.save(inputs / "displacement_y_mm.npy", displacement[..., 1])
    layout = PartitionLayout((4, 4), (1, 1), padding=0)
    local = tmp_path / "hchi-zero"
    coupled = tmp_path / "coupled"
    _write_campaign(
        local,
        layout=layout,
        displacement=0.8 * displacement,
        enabled=True,
        coupling_modulus_mpa=0.0,
    )
    _write_campaign(
        coupled,
        layout=layout,
        displacement=0.9 * displacement,
        enabled=True,
    )

    report = validate_coupled_nonlocal_campaign(
        input_directory=inputs,
        local_campaign_directory=local,
        coupled_campaign_directory=coupled,
        partition_id=0,
        output_path=tmp_path / "report.json",
    )

    assert report["status"] == "completed"
