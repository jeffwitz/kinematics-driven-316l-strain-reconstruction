import json

import numpy as np
import pytest

from fem_inhouse.config import CaseStudyConfig, MeshConfig
from fem_inhouse.core.tensor_reconstruction import reconstruct_python_plane_stress_state
from fem_inhouse.partitioning import PartitionLayout
from fem_inhouse.results import FEMResult
from fem_inhouse.workflows import partitioned
from fem_inhouse.workflows.partitioned import PartitionWorkflow, fingerprint_array


def _workflow(tmp_path, *, yield_offset: float = 0.0) -> PartitionWorkflow:
    nx, ny = 7, 5
    x = np.arange(nx + 1)[:, None]
    y = np.arange(ny + 1)[None, :]
    displacement_x = np.broadcast_to(x, (nx + 1, ny + 1)).astype(float)
    displacement_y = np.broadcast_to(y, (nx + 1, ny + 1)).astype(float)
    yield_map = np.arange(nx * ny, dtype=float).reshape(nx, ny) + 200.0 + yield_offset
    hardening_map = np.arange(nx * ny, dtype=float).reshape(nx, ny) + 400.0
    return PartitionWorkflow(
        config=CaseStudyConfig(MeshConfig(nx=nx, ny=ny)),
        layout=PartitionLayout((nx, ny), (3, 2), padding=1),
        displacement_x_mm=displacement_x,
        displacement_y_mm=displacement_y,
        yield_stress_mpa=yield_map,
        hardening_coefficient_mpa=hardening_map,
        output_directory=tmp_path / "run",
    )


def _fake_solver(calls):
    def solve(
        config,
        *,
        displacement_x_mm,
        displacement_y_mm,
        yield_stress_mpa,
        hardening_coefficient_mpa,
    ):
        calls.append(config.mesh)
        stress = np.stack(
            (
                yield_stress_mpa,
                hardening_coefficient_mpa,
                yield_stress_mpa + hardening_coefficient_mpa,
            ),
            axis=-1,
        )
        displacement = np.stack((displacement_x_mm, displacement_y_mm), axis=-1)
        tensor_state = reconstruct_python_plane_stress_state(
            np.zeros_like(stress),
            np.zeros_like(stress),
            stress,
            config.material.poisson_ratio,
        )
        return FEMResult(
            displacement_mm=displacement,
            stress_mpa=stress,
            total_strain=np.zeros_like(stress),
            plastic_strain=np.zeros_like(stress),
            equivalent_plastic_strain=np.asarray(yield_stress_mpa),
            reaction_force=np.zeros_like(displacement),
            stress_tensor_mpa=tensor_state.stress_tensor_mpa,
            total_strain_tensor=tensor_state.total_strain_tensor,
            elastic_strain_tensor=tensor_state.elastic_strain_tensor,
            plastic_strain_tensor=tensor_state.plastic_strain_tensor,
            plane_stress_residual_mpa=tensor_state.plane_stress_residual_mpa,
        )

    return solve


def test_array_fingerprint_is_stable_chunked_and_sensitive() -> None:
    values = np.arange(24, dtype=np.float64).reshape(4, 6)
    assert fingerprint_array(values, chunk_elements=3) == fingerprint_array(
        values.copy(),
        chunk_elements=100,
    )
    changed = values.copy()
    changed[0, 0] = -1
    assert fingerprint_array(changed) != fingerprint_array(values)
    assert len(fingerprint_array(np.empty((0, 2)))) == 64
    with pytest.raises(ValueError, match="chunk_elements"):
        fingerprint_array(values, chunk_elements=0)


def test_manifest_is_deterministic_and_rejects_changed_inputs(tmp_path) -> None:
    workflow = _workflow(tmp_path)
    first_digest = workflow.prepare()
    second_digest = workflow.prepare()
    manifest = json.loads(workflow.manifest_path.read_text(encoding="utf-8"))

    assert first_digest == second_digest
    assert manifest["layout"]["partition_shape"] == [3, 2]
    assert manifest["software"]["version"]
    assert len(manifest["software"]["source_sha256"]) == 64
    assert set(manifest["inputs"]) == {
        "displacement_x_mm",
        "displacement_y_mm",
        "yield_stress_mpa",
        "hardening_coefficient_mpa",
    }
    assert manifest["result_field_metadata"]["S_3D"]["unit"] == "MPa"

    with pytest.raises(RuntimeError, match="manifest does not match"):
        _workflow(tmp_path, yield_offset=1.0).prepare()


def test_partition_solves_are_resumable_and_stitchable(tmp_path, monkeypatch) -> None:
    workflow = _workflow(tmp_path)
    calls = []
    monkeypatch.setattr(partitioned, "run_case_study", _fake_solver(calls))

    assert workflow.pending_partition_ids() == list(range(6))
    assert workflow.solve_pending() == list(range(6))
    assert len(calls) == 6
    assert workflow.pending_partition_ids() == []

    workflow.solve_partition(0)
    assert len(calls) == 6
    status = workflow.solve_partition(0, force=True)
    assert len(calls) == 7
    assert status["diagnostics"]["write_seconds"] >= 0

    stitched_stress = workflow.stitch("S")
    expected_stress = np.stack(
        (
            workflow.yield_stress_mpa,
            workflow.hardening_coefficient_mpa,
            workflow.yield_stress_mpa + workflow.hardening_coefficient_mpa,
        ),
        axis=-1,
    )
    np.testing.assert_array_equal(stitched_stress, expected_stress)
    stitched_stress_tensor = workflow.stitch("S_3D")
    assert stitched_stress_tensor.shape == (7, 5, 3, 3)
    np.testing.assert_array_equal(stitched_stress_tensor[..., 0, 0], expected_stress[..., 0])

    custom_path = tmp_path / "custom_u.npy"
    stitched_displacement = workflow.stitch("U", output_path=custom_path)
    np.testing.assert_array_equal(
        stitched_displacement[..., 0],
        workflow.displacement_x_mm,
    )
    assert custom_path.exists()


def test_corrupt_or_incomplete_partition_is_detected(tmp_path, monkeypatch) -> None:
    workflow = _workflow(tmp_path)
    monkeypatch.setattr(partitioned, "run_case_study", _fake_solver([]))
    workflow.solve_partition(0)

    np.save(workflow._result_path(0, "S"), np.zeros((1, 1, 3)))
    assert 0 in workflow.pending_partition_ids()
    with pytest.raises(RuntimeError, match="incomplete partitions"):
        workflow.stitch("S")
    with pytest.raises(KeyError, match="unknown result field"):
        workflow.stitch("unknown")


def test_invalid_workflow_layout_and_partition_id_are_rejected(tmp_path) -> None:
    workflow = _workflow(tmp_path)
    with pytest.raises(ValueError, match="configured mesh"):
        PartitionWorkflow(
            config=workflow.config,
            layout=PartitionLayout((6, 5), (2, 1)),
            displacement_x_mm=workflow.displacement_x_mm,
            displacement_y_mm=workflow.displacement_y_mm,
            yield_stress_mpa=workflow.yield_stress_mpa,
            hardening_coefficient_mpa=workflow.hardening_coefficient_mpa,
            output_directory=tmp_path / "invalid",
        )
    with pytest.raises(KeyError, match="unknown partition_id"):
        workflow.solve_partition(99)

    with pytest.raises(ValueError, match="displacement_x_mm has shape"):
        PartitionWorkflow(
            config=workflow.config,
            layout=workflow.layout,
            displacement_x_mm=np.zeros((2, 2)),
            displacement_y_mm=workflow.displacement_y_mm,
            yield_stress_mpa=workflow.yield_stress_mpa,
            hardening_coefficient_mpa=workflow.hardening_coefficient_mpa,
            output_directory=tmp_path / "invalid-fields",
        )
