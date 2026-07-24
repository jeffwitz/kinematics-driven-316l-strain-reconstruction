import json
from pathlib import Path

import numpy as np
import pytest

from fem_inhouse.cli import main
from fem_inhouse.data_preparation import (
    PreparationConfig,
    fingerprint_file,
    prepare_case_study,
    verify_raw_case_study,
)


def _raw_case(directory: Path, *, with_nan: bool = True) -> dict[str, np.ndarray]:
    directory.mkdir()
    arrays = {
        "U_40.npy": np.arange(6, dtype=np.float32).reshape(3, 2) + 10,
        "V_40.npy": np.arange(6, dtype=np.float32).reshape(3, 2) - 3,
        "el_thresh50.npy": np.arange(6, dtype=np.float64).reshape(3, 2) + 50,
        "Hardening_coeff_el_Thresh50.npy": np.arange(6, dtype=np.float64).reshape(3, 2) + 1,
    }
    if with_nan:
        arrays["Hardening_coeff_el_Thresh50.npy"][0, 0] = np.nan
    files = {}
    for filename, values in arrays.items():
        path = directory / filename
        np.save(path, values)
        files[filename] = {
            "bytes": path.stat().st_size,
            "dtype": str(values.dtype),
            "sha256": fingerprint_file(path),
            "shape": list(values.shape),
        }
    (directory / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": files}),
        encoding="utf-8",
    )
    return arrays


def test_prepare_case_maps_scales_repairs_and_edge_pads(tmp_path) -> None:
    raw_directory = tmp_path / "raw"
    expected = _raw_case(raw_directory)
    output_directory = tmp_path / "prepared"
    config = PreparationConfig(
        pixel_size_um=2.0,
        hardening_scale_mpa=380.0,
        nonfinite_policy="nearest",
    )

    manifest = prepare_case_study(raw_directory, output_directory, config=config)

    displacement_x = np.load(output_directory / "displacement_x_mm.npy")
    displacement_y = np.load(output_directory / "displacement_y_mm.npy")
    yield_stress = np.load(output_directory / "yield_stress_mpa.npy")
    hardening = np.load(output_directory / "hardening_coefficient_mpa.npy")
    assert displacement_x.shape == (4, 3)
    assert displacement_y.shape == (4, 3)
    np.testing.assert_allclose(displacement_x[:-1, :-1], expected["V_40.npy"] * 0.002)
    np.testing.assert_allclose(displacement_y[:-1, :-1], expected["U_40.npy"] * 0.002)
    np.testing.assert_allclose(displacement_x[-1, :-1], displacement_x[-2, :-1])
    np.testing.assert_allclose(displacement_x[:, -1], displacement_x[:, -2])
    np.testing.assert_array_equal(yield_stress, expected["el_thresh50.npy"])
    assert hardening[0, 0] == pytest.approx(expected["Hardening_coeff_el_Thresh50.npy"][0, 1] * 380)
    np.testing.assert_allclose(
        hardening[1:],
        expected["Hardening_coeff_el_Thresh50.npy"][1:] * 380,
    )
    assert manifest["transformations"]["hardening_nonfinite_repaired_indices"] == [[0, 0]]
    assert manifest["config"]["hardening_scale_mpa"] == 380.0
    assert manifest["outputs"]["displacement_x_mm"]["shape"] == [4, 3]
    assert json.loads((output_directory / "manifest.json").read_text()) == manifest

    assert prepare_case_study(raw_directory, output_directory, config=config) == manifest


def test_prepare_case_requires_explicit_nonfinite_policy(tmp_path) -> None:
    raw_directory = tmp_path / "raw"
    _raw_case(raw_directory)

    with pytest.raises(ValueError, match="select nonfinite_policy='nearest' explicitly"):
        prepare_case_study(raw_directory, tmp_path / "prepared")


def test_prepare_case_rejects_changed_configuration_for_existing_output(tmp_path) -> None:
    raw_directory = tmp_path / "raw"
    _raw_case(raw_directory, with_nan=False)
    output_directory = tmp_path / "prepared"
    prepare_case_study(raw_directory, output_directory)

    with pytest.raises(RuntimeError, match="different source or configuration"):
        prepare_case_study(
            raw_directory,
            output_directory,
            config=PreparationConfig(hardening_scale_mpa=396.0),
        )


def test_raw_integrity_verification_rejects_modified_file(tmp_path) -> None:
    raw_directory = tmp_path / "raw"
    _raw_case(raw_directory, with_nan=False)
    assert verify_raw_case_study(raw_directory)["schema_version"] == 1
    with (raw_directory / "U_40.npy").open("ab") as stream:
        stream.write(b"corruption")

    with pytest.raises(ValueError, match="byte size"):
        verify_raw_case_study(raw_directory)


def test_prepare_case_cli_writes_manifest(tmp_path, capsys) -> None:
    raw_directory = tmp_path / "raw"
    _raw_case(raw_directory)
    output_directory = tmp_path / "prepared"

    assert (
        main(
            [
                "prepare-case",
                "--raw",
                str(raw_directory),
                "--output",
                str(output_directory),
                "--nonfinite-policy",
                "nearest",
                "--hardening-scale-mpa",
                "396",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["config"]["hardening_scale_mpa"] == 396.0
    assert (output_directory / "hardening_coefficient_mpa.npy").is_file()


def test_prepare_case_supports_a_reproducible_central_crop(tmp_path) -> None:
    raw_directory = tmp_path / "raw"
    arrays = _raw_case(raw_directory)
    output_directory = tmp_path / "prepared"

    manifest = prepare_case_study(
        raw_directory,
        output_directory,
        config=PreparationConfig(crop_nx=1, crop_ny=2),
    )

    np.testing.assert_array_equal(
        np.load(output_directory / "yield_stress_mpa.npy"),
        arrays["el_thresh50.npy"][1:2, :],
    )
    assert manifest["config"]["crop_nx"] == 1
    assert manifest["transformations"]["source_crop_bounds_axis_0_axis_1"] == [1, 2, 0, 2]


def test_preparation_config_defaults() -> None:
    assert PreparationConfig().pixel_size_um == 1.84


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"pixel_size_um": 0.0}, "pixel_size_um"),
        ({"hardening_scale_mpa": 0.0}, "hardening_scale_mpa"),
        ({"nonfinite_policy": "invalid"}, "nonfinite_policy"),
        ({"nodal_completion": "invalid"}, "nodal_completion"),
        ({"crop_nx": 1}, "specified together"),
        ({"crop_nx": 0, "crop_ny": 1}, "positive"),
    ],
)
def test_preparation_config_rejects_invalid_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        PreparationConfig(**kwargs)
