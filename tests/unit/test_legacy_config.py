from pathlib import Path

import numpy as np
import pytest

from fem_inhouse import legacy_config
from fem_inhouse.legacy_config import LegacyCasePaths


def _paths(tmp_path: Path) -> LegacyCasePaths:
    return LegacyCasePaths(
        input_directory=tmp_path / "inputs",
        dic_directory=tmp_path / "dic",
        macro_stress_strain_file=tmp_path / "stress_strain.npy",
        validation_directory=tmp_path / "results",
    )


def _write_inputs(paths: LegacyCasePaths, *, padding: int = 0) -> None:
    paths.input_directory.mkdir(parents=True)
    nodal_shape = (
        legacy_config.NX + 1 + 2 * padding,
        legacy_config.NY + 1 + 2 * padding,
    )
    element_shape = (
        legacy_config.NX + 2 * padding,
        legacy_config.NY + 2 * padding,
    )
    np.save(paths.input_directory / "displacement_x_mm.npy", np.ones(nodal_shape))
    np.save(paths.input_directory / "displacement_y_mm.npy", np.full(nodal_shape, 2.0))
    np.save(paths.input_directory / "yield_stress_mpa.npy", np.full(element_shape, 250.0))
    np.save(
        paths.input_directory / "hardening_coefficient_mpa.npy",
        np.full(element_shape, 500.0),
    )


def test_legacy_inputs_load_and_crop_portably(tmp_path) -> None:
    paths = _paths(tmp_path)
    _write_inputs(paths, padding=2)

    ux, uy, yield_map, hardening_map = legacy_config.load_case5_inputs(paths)

    assert ux.shape == (legacy_config.NX + 1, legacy_config.NY + 1)
    assert uy.shape == ux.shape
    assert yield_map.shape == (legacy_config.NX, legacy_config.NY)
    assert hardening_map.shape == yield_map.shape
    np.testing.assert_array_equal((ux.mean(), uy.mean()), (1.0, 2.0))
    assert legacy_config.window_tag() == f"{legacy_config.NX}x{legacy_config.NY}"


def test_crop_contract_rejects_invalid_requests() -> None:
    values = np.arange(6 * 8).reshape(6, 8)
    np.testing.assert_array_equal(
        legacy_config.crop_center(values, 2, 4),
        values[2:4, 2:6],
    )
    with pytest.raises(ValueError, match="positive"):
        legacy_config.crop_center(values, 0, 2)
    with pytest.raises(ValueError, match="cannot crop"):
        legacy_config.crop_center(values, 7, 2)


def test_missing_and_invalid_legacy_inputs_fail_clearly(tmp_path) -> None:
    paths = _paths(tmp_path)
    paths.input_directory.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="legacy_data_contract"):
        legacy_config.load_case5_inputs(paths)

    np.save(paths.input_directory / "displacement_x_mm.npy", np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="2D"):
        legacy_config.load_case5_inputs(paths)


@pytest.mark.parametrize(
    ("filename", "value", "message"),
    [
        ("yield_stress_mpa.npy", 0.0, "strictly positive"),
        ("hardening_coefficient_mpa.npy", -1.0, "nonnegative"),
    ],
)
def test_material_domain_is_checked(tmp_path, filename, value, message) -> None:
    paths = _paths(tmp_path)
    _write_inputs(paths)
    target = paths.input_directory / filename
    values = np.load(target)
    values[0, 0] = value
    np.save(target, values)

    with pytest.raises(ValueError, match=message):
        legacy_config.load_case5_inputs(paths)
