from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fem_inhouse.data_preparation import fingerprint_file
from fem_inhouse.workflows.material_map_controls import prepare_material_map_control


def _source(path: Path) -> tuple[np.ndarray, np.ndarray]:
    path.mkdir()
    displacement = np.zeros((5, 4), dtype=np.float64)
    yield_stress = np.arange(12, dtype=np.float64).reshape(4, 3) + 50.0
    hardening = 10.0 * yield_stress
    arrays = {
        "displacement_x_mm": displacement,
        "displacement_y_mm": displacement + 1.0,
        "yield_stress_mpa": yield_stress,
        "hardening_coefficient_mpa": hardening,
    }
    outputs = {}
    for name, values in arrays.items():
        target = path / f"{name}.npy"
        np.save(target, values)
        outputs[name] = {"sha256": fingerprint_file(target)}
    (path / "manifest.json").write_text(json.dumps({"outputs": outputs}), encoding="utf-8")
    return yield_stress, hardening


def test_homogeneous_control_changes_only_material_maps(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source(source)
    output = tmp_path / "homogeneous"

    manifest = prepare_material_map_control(
        source,
        output,
        mode="homogeneous",
        homogeneous_yield_stress_mpa=124.0,
        homogeneous_hardening_coefficient_mpa=380.0,
    )

    np.testing.assert_array_equal(np.load(output / "yield_stress_mpa.npy"), 124.0)
    np.testing.assert_array_equal(np.load(output / "hardening_coefficient_mpa.npy"), 380.0)
    np.testing.assert_array_equal(
        np.load(output / "displacement_x_mm.npy"),
        np.load(source / "displacement_x_mm.npy"),
    )
    assert manifest["transformation"]["mode"] == "homogeneous"


def test_translated_control_preserves_distributions_and_pairing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    yield_stress, _hardening = _source(source)
    output = tmp_path / "translated"

    manifest = prepare_material_map_control(
        source,
        output,
        mode="translated",
        shift_x_pixels=1,
        shift_y_pixels=-1,
    )

    shifted_yield = np.load(output / "yield_stress_mpa.npy")
    shifted_hardening = np.load(output / "hardening_coefficient_mpa.npy")
    np.testing.assert_array_equal(
        shifted_yield,
        np.roll(yield_stress, shift=(1, -1), axis=(0, 1)),
    )
    np.testing.assert_array_equal(shifted_hardening, 10.0 * shifted_yield)
    np.testing.assert_array_equal(
        np.sort(shifted_yield, axis=None),
        np.sort(yield_stress, axis=None),
    )
    assert manifest["transformation"]["joint_pairing_preserved"] is True


def test_control_refuses_overwrite_and_corrupt_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source(source)
    output = tmp_path / "control"
    prepare_material_map_control(source, output, mode="homogeneous")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_material_map_control(source, output, mode="homogeneous")

    values = np.load(source / "yield_stress_mpa.npy")
    values[0, 0] += 1.0
    np.save(source / "yield_stress_mpa.npy", values)
    with pytest.raises(RuntimeError, match="fails its manifest hash"):
        prepare_material_map_control(source, tmp_path / "corrupt", mode="translated")
