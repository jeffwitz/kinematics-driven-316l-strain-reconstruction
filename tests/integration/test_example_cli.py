import json

import numpy as np
import pytest

from fem_inhouse.cli import main
from fem_inhouse.examples import (
    reduced_biaxial_case,
    save_reduced_example,
    validate_reduced_case,
)


def test_reduced_tabular_case_passes_declared_thresholds() -> None:
    _result, report = validate_reduced_case(reduced_biaxial_case(nx=4, ny=4))

    assert report.passed
    assert report.relative_stress_error < 0.005
    assert report.relative_plastic_strain_error < 0.005
    assert report.relative_displacement_error < 1e-8
    assert report.relative_reaction_imbalance < 1e-10


def test_reduced_case_rejects_elastic_target() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        reduced_biaxial_case(target_stress_mpa=200.0)


def test_example_command_saves_self_describing_results(tmp_path, capsys) -> None:
    destination = tmp_path / "example"
    assert main(["example", "--output", str(destination), "--nx", "4", "--ny", "4"]) == 0
    printed = json.loads(capsys.readouterr().out)
    report = json.loads((destination / "report.json").read_text(encoding="utf-8"))

    assert printed["passed"]
    assert report["validation"]["passed"]
    assert report["config"]["solver"]["hardening_mode"] == "tabular"
    assert np.load(destination / "stress_mpa.npy").shape == (4, 4, 3)
    assert np.load(destination / "displacement_mm.npy").shape == (5, 5, 2)
    assert (destination / "equivalent_plastic_strain.npy").exists()


def test_backend_validate_and_layout_commands(tmp_path, capsys) -> None:
    assert main(["backend"]) == 0
    assert capsys.readouterr().out.startswith("pypardiso")

    assert main(["validate", "--nx", "4", "--ny", "4"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"]

    manifest_path = tmp_path / "layout.json"
    assert (
        main(
            [
                "layout",
                "--count",
                "25",
                "--padding",
                "150",
                "--output",
                str(manifest_path),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == str(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["partition_shape"] == [5, 5]
    assert len(manifest["partitions"]) == 25


def test_save_example_function_returns_report(tmp_path) -> None:
    report = save_reduced_example(tmp_path / "direct", nx=4, ny=4)
    assert report.passed
