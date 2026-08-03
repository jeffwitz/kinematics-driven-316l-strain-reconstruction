import json
import logging
from dataclasses import replace

import numpy as np
import pytest

from fem_inhouse.config import CaseStudyConfig, MeshConfig
from fem_inhouse.core.tensor_reconstruction import reconstruct_python_plane_stress_state
from fem_inhouse.partitioning import PartitionLayout, extract_partition_field
from fem_inhouse.results import FEMResult, FrameResult
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


def _fake_solver(calls, *, include_hourglass: bool = False):
    def solve(
        config,
        *,
        displacement_x_mm,
        displacement_y_mm,
        yield_stress_mpa,
        hardening_coefficient_mpa,
        snapshots=(),
        verbose=False,
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
        frames = {
            float(fraction): FrameResult(
                stress_mpa=stress * fraction,
                total_strain=np.zeros_like(stress),
                equivalent_plastic_strain=np.asarray(yield_stress_mpa) * fraction,
                displacement_mm=displacement * fraction,
            )
            for fraction in snapshots
        }
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
            nonlocal_equivalent_plastic_strain=np.asarray(yield_stress_mpa),
            equivalent_plastic_strain_mismatch=np.zeros_like(yield_stress_mpa),
            nonlocal_hardening_mpa=np.zeros_like(yield_stress_mpa),
            yield_surface_radius_mpa=np.asarray(yield_stress_mpa),
            nonlocal_residual=np.zeros_like(yield_stress_mpa),
            hourglass_energy_by_element=(
                np.asarray(yield_stress_mpa) if include_hourglass else None
            ),
            frames=frames,
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
    assert "PEEQ_NONLOCAL" not in manifest["result_fields"]

    with pytest.raises(RuntimeError, match="manifest does not match"):
        _workflow(tmp_path, yield_offset=1.0).prepare()


def test_nonlocal_campaign_persists_and_stitches_coupling_fields(tmp_path, monkeypatch) -> None:
    workflow = _workflow(tmp_path)
    workflow.config = replace(
        workflow.config,
        nonlocal_plasticity=replace(
            workflow.config.nonlocal_plasticity,
            enabled=True,
            coupling_modulus_mpa=1_000.0,
        ),
    )
    workflow.output_directory = tmp_path / "nonlocal-run"
    monkeypatch.setattr(partitioned, "run_case_study", _fake_solver([]))

    manifest = workflow._manifest_data()
    assert "PEEQ_NONLOCAL" in manifest["result_fields"]
    workflow.solve_pending()
    stitched = workflow.stitch("PEEQ_NONLOCAL")

    np.testing.assert_array_equal(stitched, workflow.yield_stress_mpa)


def test_reduced_campaign_persists_and_stitches_hourglass_energy(
    tmp_path, monkeypatch
) -> None:
    workflow = _workflow(tmp_path)
    workflow.config = replace(
        workflow.config,
        solver=replace(
            workflow.config.solver,
            element_formulation="cps4r",
        ),
    )
    workflow.output_directory = tmp_path / "reduced-run"
    monkeypatch.setattr(
        partitioned,
        "run_case_study",
        _fake_solver([], include_hourglass=True),
    )

    manifest = workflow._manifest_data()
    assert "HOURGLASS_ENERGY_BY_ELEMENT" in manifest["result_fields"]
    assert (
        manifest["result_field_metadata"]["HOURGLASS_ENERGY_BY_ELEMENT"]["unit"]
        == "N mm for implicit 1 mm thickness"
    )

    workflow.solve_pending()
    stitched = workflow.stitch("HOURGLASS_ENERGY_BY_ELEMENT")

    np.testing.assert_array_equal(stitched, workflow.yield_stress_mpa)


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


def test_partition_snapshots_are_manifested_hashed_and_resumable(
    tmp_path,
    monkeypatch,
) -> None:
    workflow = _workflow(tmp_path)
    workflow.snapshot_fractions = (1.0, 0.25, 0.5, 0.75)
    workflow.__post_init__()
    calls = []
    monkeypatch.setattr(partitioned, "run_case_study", _fake_solver(calls))

    status = workflow.solve_partition(0)

    assert workflow._manifest_data()["snapshots"]["fractions"] == [
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert set(status["snapshots"]) == {"0.250000", "0.500000", "0.750000", "1.000000"}
    snapshot_u = np.load(workflow._snapshot_path(0, 0.5, "U"))
    expected_x = (
        extract_partition_field(
            workflow.displacement_x_mm,
            layout=workflow.layout,
            partition=workflow.layout.get(0),
            location="node",
        )
        * 0.5
    )
    np.testing.assert_array_equal(snapshot_u[..., 0], expected_x)
    assert workflow.solve_partition(0) == status
    assert len(calls) == 1

    snapshot_path = workflow._snapshot_path(0, 0.5, "PEEQ")
    np.save(snapshot_path, np.zeros((1, 1)))
    assert 0 in workflow.pending_partition_ids()


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

    with pytest.raises(ValueError, match="snapshot_fractions"):
        workflow.snapshot_fractions = (0.0,)
        workflow.__post_init__()


def _verbose_recording_solver(seen: list[bool]):
    solver = _fake_solver([])

    def solve(config, *, verbose=False, **kwargs):
        seen.append(verbose)
        return solver(config, verbose=verbose, **kwargs)

    return solve


@pytest.mark.parametrize(
    ("level", "expected"),
    [(logging.INFO, True), (logging.WARNING, False)],
)
def test_the_solver_follows_the_caller_logging_level(
    tmp_path, monkeypatch, caplog, level, expected
) -> None:
    """`--verbose` must reach the solver, not just raise the logging level.

    It did not: a partition solve ran silently from the first line to the last,
    so a multi-hour campaign could not be followed at all.
    """

    seen: list[bool] = []
    monkeypatch.setattr(partitioned, "run_case_study", _verbose_recording_solver(seen))
    workflow = _workflow(tmp_path)

    with caplog.at_level(level, logger=partitioned.LOGGER.name):
        workflow.solve_partition(0)

    assert seen == [expected]


def test_a_crystal_campaign_manifest_carries_its_parameter_provenance(tmp_path) -> None:
    """Section 5. A reader of the archive must be able to tell an identified
    value from a transposed one without leaving the manifest."""

    workflow = _workflow(tmp_path)
    workflow.config = replace(
        workflow.config,
        solver=replace(
            workflow.config.solver,
            constitutive_backend="mfront-3d-condensed-plane-stress",
            mfront_behaviour_id="fcc_forest_rubin_srix",
            constitutive_options={"parameter_set": "316l_srix_exploratory_r2"},
        ),
    )

    crystal = workflow._manifest_data()["crystal_parameters"]

    assert crystal["identifier"] == "316l_srix_exploratory_r2"
    assert crystal["values"]["R_mpa"] == 2.0
    assert crystal["units"]["R_mpa"] == "MPa"
    assert crystal["claims_material_identification"] is False
    assert crystal["origins"]["overstress_modulus"]["status"] == "exploratory"
    assert len(crystal["interaction_matrix"]["coefficients"]) == 7
    assert set(crystal["run"]) == {"mfront_source", "toolchain", "git_commit"}


def test_a_j2_campaign_manifest_has_no_crystal_provenance(tmp_path) -> None:
    """There is no selectable parameter set, so an empty block would mislead."""

    assert "crystal_parameters" not in _workflow(tmp_path)._manifest_data()
