import json

import numpy as np
import pytest

from fem_inhouse.cli import main
from fem_inhouse.examples import (
    reduced_biaxial_case,
    save_reduced_example,
    validate_reduced_case,
)
from fem_inhouse.results import FEMResult
from fem_inhouse.workflows import partitioned


def test_reduced_python_ludwik_case_passes_declared_thresholds() -> None:
    _result, report = validate_reduced_case(
        reduced_biaxial_case(nx=4, ny=4, constitutive_backend="python")
    )

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
    assert (
        main(
            [
                "example",
                "--output",
                str(destination),
                "--nx",
                "4",
                "--ny",
                "4",
                "--constitutive-backend",
                "python",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    report = json.loads((destination / "report.json").read_text(encoding="utf-8"))

    assert printed["passed"]
    assert report["validation"]["passed"]
    assert report["diagnostics"]["backend"].startswith("pypardiso")
    assert report["diagnostics"]["converged_increments"] > 0
    assert report["diagnostics"]["linear_solve_seconds"] > 0
    assert report["config"]["solver"]["hardening_mode"] == "ludwik"
    assert report["config"]["solver"]["constitutive_backend"] == "python"
    assert np.load(destination / "stress_mpa.npy").shape == (4, 4, 3)
    assert np.load(destination / "displacement_mm.npy").shape == (5, 5, 2)
    assert (destination / "equivalent_plastic_strain.npy").exists()


def test_backend_validate_and_layout_commands(tmp_path, capsys) -> None:
    assert main(["backend"]) == 0
    assert capsys.readouterr().out.startswith("pypardiso")

    assert (
        main(
            [
                "validate",
                "--nx",
                "4",
                "--ny",
                "4",
                "--constitutive-backend",
                "python",
            ]
        )
        == 0
    )
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
    destination = tmp_path / "direct"
    report = save_reduced_example(
        destination,
        nx=4,
        ny=4,
        constitutive_backend="python",
    )
    assert report.passed
    for field_name in (
        "S_3D",
        "E_3D",
        "EE_3D",
        "PE_3D",
        "S33_RESIDUAL_MPA",
    ):
        assert (destination / f"{field_name}.npy").is_file()
    metadata = json.loads((destination / "report.json").read_text(encoding="utf-8"))
    assert metadata["result_field_metadata"]["S_3D"]["unit"] == "MPa"


def test_partition_cli_supports_job_arrays_resume_and_stitch(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    np.save(input_directory / "displacement_x_mm.npy", np.zeros((6, 6)))
    np.save(input_directory / "displacement_y_mm.npy", np.zeros((6, 6)))
    np.save(input_directory / "yield_stress_mpa.npy", np.full((5, 5), 250.0))
    np.save(
        input_directory / "hardening_coefficient_mpa.npy",
        np.full((5, 5), 500.0),
    )

    def fake_solver(config, **_fields):
        nx, ny = config.mesh.nx, config.mesh.ny
        element_tensor = np.zeros((nx, ny, 3, 3))
        return FEMResult(
            displacement_mm=np.zeros((nx + 1, ny + 1, 2)),
            stress_mpa=np.ones((nx, ny, 3)),
            total_strain=np.zeros((nx, ny, 3)),
            plastic_strain=np.zeros((nx, ny, 3)),
            equivalent_plastic_strain=np.zeros((nx, ny)),
            reaction_force=np.zeros((nx + 1, ny + 1, 2)),
            stress_tensor_mpa=element_tensor.copy(),
            total_strain_tensor=element_tensor.copy(),
            elastic_strain_tensor=element_tensor.copy(),
            plastic_strain_tensor=element_tensor.copy(),
            plane_stress_residual_mpa=np.zeros((nx, ny)),
        )

    monkeypatch.setattr(partitioned, "run_case_study", fake_solver)
    common = [
        "partition",
        "--input",
        str(input_directory),
        "--output",
        str(output_directory),
        "--count",
        "25",
        "--padding",
        "0",
    ]

    assert main([*common, "--list-pending"]) == 0
    assert json.loads(capsys.readouterr().out)["pending"] == list(range(25))

    assert main([*common, "--partition-id", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["complete"]
    for field_name in (
        "U",
        "S",
        "S_3D",
        "E",
        "E_3D",
        "EE_3D",
        "PE",
        "PE_3D",
        "PEEQ",
        "S33_RESIDUAL_MPA",
        "RF",
    ):
        assert (output_directory / "partitions" / "0000" / f"{field_name}.npy").is_file()

    assert main([*common, "--solve-pending"]) == 0
    solve_report = json.loads(capsys.readouterr().out)
    assert solve_report["solved"] == list(range(1, 25))
    assert solve_report["remaining"] == []

    stitched_path = tmp_path / "global_stress.npy"
    assert main([*common, "--stitch", "S", "--field-output", str(stitched_path)]) == 0
    assert capsys.readouterr().out.strip() == str(stitched_path)
    np.testing.assert_array_equal(np.load(stitched_path), 1.0)

    tensor_path = tmp_path / "global_stress_tensor.npy"
    assert main([*common, "--stitch", "S_3D", "--field-output", str(tensor_path)]) == 0
    assert capsys.readouterr().out.strip() == str(tensor_path)
    assert np.load(tensor_path).shape == (5, 5, 3, 3)

    reaction_path = tmp_path / "global_reaction.npy"
    assert main([*common, "--stitch", "RF", "--field-output", str(reaction_path)]) == 0
    assert capsys.readouterr().out.strip() == str(reaction_path)
    assert np.load(reaction_path).shape == (6, 6, 2)


def test_compare_fields_cli_writes_report_and_signed_map(tmp_path, capsys) -> None:
    reference_path = tmp_path / "reference.npy"
    prediction_path = tmp_path / "prediction.npy"
    report_path = tmp_path / "comparison" / "report.json"
    difference_path = tmp_path / "comparison" / "difference.npy"
    np.save(reference_path, np.array([[4.0, 3.0], [2.0, 1.0]]))
    np.save(prediction_path, np.array([[4.1, 2.9], [2.0, 1.0]]))
    arguments = [
        "compare-fields",
        "--reference",
        str(reference_path),
        "--prediction",
        str(prediction_path),
        "--report",
        str(report_path),
        "--difference",
        str(difference_path),
        "--top-fraction",
        "0.25",
        "--max-rmse",
        "0.1",
        "--max-mae",
        "0.1",
        "--min-correlation",
        "0.99",
        "--min-localization-iou",
        "1.0",
    ]

    assert main(arguments) == 0
    printed = json.loads(capsys.readouterr().out)
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert printed == persisted
    assert persisted["passed"]
    np.testing.assert_allclose(
        np.load(difference_path),
        [[0.1, -0.1], [0.0, 0.0]],
        atol=1e-15,
    )

    rejected_arguments = arguments.copy()
    rejected_arguments[rejected_arguments.index("--max-rmse") + 1] = "0.01"
    assert main(rejected_arguments) == 1
    assert not json.loads(capsys.readouterr().out)["passed"]
