"""Fast, MGIS-free checks for the NumPy SRIX backend."""

import numpy as np
import pytest

from fem_inhouse.core.crystal_orientation import HomogeneousOrientationProvider
from fem_inhouse.core.plane_stress_material import ConstitutiveTrial, InPlaneConstitutiveTrial
from fem_inhouse.core.srix_numpy import (
    _ENGINEERING_TO_KELVIN_STRAIN_SCALE,
    _KELVIN_TO_ENGINEERING_STRESS_SCALE,
    _PLANE,
    _TRANSVERSE,
    SrixNumpy3DMaterialPointBatch,
    SrixNumpyCondensedPlaneStressBatch,
)


def test_numpy_srix_elastic_batch_and_transaction() -> None:
    material = SrixNumpy3DMaterialPointBatch(point_count=3)
    strain = np.tile([1.0e-5, -2.0e-5, 0.0, 3.0e-5, 0.0, 0.0], (3, 1))
    trial = material.evaluate(strain, time_increment=1.0)
    assert np.allclose(trial.stress_kelvin_mpa[0], trial.stress_kelvin_mpa[1])
    before = material.committed_state
    material.revert()
    assert np.array_equal(material.committed_state["elastic_strain"], before["elastic_strain"])
    material.evaluate(strain, time_increment=1.0)
    material.commit()
    assert np.allclose(
        material.committed_state["elastic_strain"], trial.material_elastic_strain_kelvin
    )


def test_numpy_srix_orientation_and_chunking_are_consistent() -> None:
    rotation = HomogeneousOrientationProvider.from_euler_bunge_deg(
        17.0, 31.0, 43.0
    ).rotations_global_to_material(5)
    strain = np.tile([3.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0], (5, 1))
    whole = SrixNumpy3DMaterialPointBatch(point_count=5, rotation_global_to_material=rotation)
    chunked = SrixNumpy3DMaterialPointBatch(
        point_count=5, rotation_global_to_material=rotation, batch_size=2
    )
    first = whole.evaluate(strain, time_increment=1.0)
    second = chunked.evaluate(strain, time_increment=1.0)
    assert np.allclose(first.stress_kelvin_mpa, second.stress_kelvin_mpa)
    assert np.allclose(first.consistent_tangent_kelvin_mpa, second.consistent_tangent_kelvin_mpa)


def test_numpy_srix_algorithmic_tangent_matches_directional_difference() -> None:
    rotation = HomogeneousOrientationProvider.from_euler_bunge_deg(
        17.0, 31.0, 43.0
    ).rotations_global_to_material(1)
    material = SrixNumpy3DMaterialPointBatch(point_count=1, rotation_global_to_material=rotation)
    strain = np.array([[3.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0]])
    tangent = material.evaluate(strain, time_increment=1.0).consistent_tangent_kelvin_mpa[0]
    material.revert()
    finite_difference = np.empty((6, 6))
    for component in range(6):
        h = 1.0e-7
        plus = strain.copy()
        minus = strain.copy()
        plus[0, component] += h
        minus[0, component] -= h
        stress_plus = material.evaluate(plus, time_increment=1.0).stress_kelvin_mpa[0]
        material.revert()
        stress_minus = material.evaluate(minus, time_increment=1.0).stress_kelvin_mpa[0]
        material.revert()
        finite_difference[:, component] = (stress_plus - stress_minus) / (2.0 * h)
    relative_error = np.linalg.norm(tangent - finite_difference) / np.linalg.norm(finite_difference)
    assert relative_error < 1.0e-6


def test_numpy_srix_reduced_tangent_matches_full_18x18_oracle() -> None:
    rotation = HomogeneousOrientationProvider.from_euler_bunge_deg(
        17.0, 31.0, 43.0
    ).rotations_global_to_material(1)
    material = SrixNumpy3DMaterialPointBatch(point_count=1, rotation_global_to_material=rotation)
    strain_increment = np.array([[3.0e-3, -4.0e-4, 2.0e-4, 1.0e-4, 0.0, -2.0e-4]])
    reduced = material._integrate_chunk(strain_increment, strain_increment)
    material.revert()
    full = material._integrate_chunk_full(strain_increment, strain_increment)
    assert np.allclose(
        reduced.consistent_tangent_kelvin_mpa,
        full.consistent_tangent_kelvin_mpa,
        rtol=2.0e-10,
        atol=1.0e-7,
    )


def test_numpy_srix_plane_stress_closes_all_three_transverse_components() -> None:
    material = SrixNumpyCondensedPlaneStressBatch(SrixNumpy3DMaterialPointBatch(point_count=2))
    trial = material.evaluate(
        np.array([[3.0e-3, -4.0e-4, 2.0e-4], [1.0e-4, 0.0, 0.0]]), time_increment=1.0
    )
    assert np.max(np.abs(trial.plane_stress_residual_mpa)) < 1.0e-7
    assert trial.tangent_in_plane_mpa is not None


def test_numpy_srix_tangent_transverse_predictor_matches_committed_seed() -> None:
    committed = SrixNumpyCondensedPlaneStressBatch(
        SrixNumpy3DMaterialPointBatch(point_count=2),
        local_transverse_predictor="committed",
    )
    tangent = SrixNumpyCondensedPlaneStressBatch(
        SrixNumpy3DMaterialPointBatch(point_count=2),
        local_transverse_predictor="tangent",
    )
    path = (
        np.array([[5.0e-4, -1.0e-4, 2.0e-4], [2.0e-4, 0.0, 1.0e-4]]),
        np.array([[1.2e-3, -2.0e-4, 4.0e-4], [7.0e-4, 0.0, 2.0e-4]]),
        np.array([[2.0e-3, -3.0e-4, 6.0e-4], [1.1e-3, 0.0, 3.0e-4]]),
    )
    for strain in path:
        trial_committed = committed.evaluate(strain, time_increment=1.0)
        trial_tangent = tangent.evaluate(strain, time_increment=1.0)
        assert np.allclose(
            trial_committed.stress_in_plane_mpa, trial_tangent.stress_in_plane_mpa, rtol=1e-10
        )
        assert np.allclose(
            trial_committed.plane_stress_residual_mpa,
            trial_tangent.plane_stress_residual_mpa,
            atol=1e-8,
        )
        committed.commit()
        tangent.commit()


def test_numpy_srix_response_levels_return_only_requested_payload() -> None:
    material = SrixNumpyCondensedPlaneStressBatch(SrixNumpy3DMaterialPointBatch(point_count=1))
    strain = np.array([[1.0e-3, -1.0e-4, 2.0e-4]])
    residual = material.evaluate_in_plane_response(
        strain, time_increment=1.0, response_level="residual"
    )
    assert isinstance(residual, InPlaneConstitutiveTrial)
    assert residual.tangent_in_plane_mpa is None
    tangent = material.evaluate_in_plane_response(
        strain, time_increment=1.0, response_level="tangent"
    )
    assert isinstance(tangent, InPlaneConstitutiveTrial)
    assert tangent.tangent_in_plane_mpa is not None
    complete = material.evaluate_in_plane_response(
        strain, time_increment=1.0, response_level="complete"
    )
    assert isinstance(complete, ConstitutiveTrial)


def test_numpy_srix_dask_threads_matches_serial() -> None:
    pytest.importorskip("dask")
    rng = np.random.default_rng(7)
    strain = rng.normal(0.0, 1.0e-3, (12, 6))
    serial = SrixNumpy3DMaterialPointBatch(point_count=12, batch_size=3)
    parallel = SrixNumpy3DMaterialPointBatch(
        point_count=12,
        batch_size=3,
        parallel_backend="dask-threads",
        dask_workers=2,
    )
    expected = serial.evaluate(strain, time_increment=1.0, tangent_mode="none")
    actual = parallel.evaluate(strain, time_increment=1.0, tangent_mode="none")
    assert np.array_equal(expected.stress_kelvin_mpa, actual.stress_kelvin_mpa)
    assert np.array_equal(expected.plastic_slip, actual.plastic_slip)


def test_numpy_srix_numba_lu12_matches_numpy_solver() -> None:
    pytest.importorskip("numba")
    strain = np.array([[3.0e-3, -4.0e-4, 2.0e-4, 1.0e-4, 0.0, -2.0e-4]])
    numpy_material = SrixNumpy3DMaterialPointBatch(point_count=1, local_linear_solver="numpy")
    numba_material = SrixNumpy3DMaterialPointBatch(
        point_count=1, local_linear_solver="numba-lu12"
    )
    expected = numpy_material.evaluate(strain, time_increment=1.0, tangent_mode="none")
    actual = numba_material.evaluate(strain, time_increment=1.0, tangent_mode="none")
    assert np.allclose(expected.stress_kelvin_mpa, actual.stress_kelvin_mpa, rtol=1.0e-12)
    assert np.allclose(expected.plastic_slip, actual.plastic_slip, rtol=1.0e-12, atol=1.0e-14)


def test_numpy_srix_coupled_plane_stress_matches_nested() -> None:
    strain = np.array(
        [[1.0e-3, -2.0e-4, 3.0e-4], [2.0e-3, 4.0e-4, -1.0e-4]],
    )
    nested = SrixNumpyCondensedPlaneStressBatch(
        SrixNumpy3DMaterialPointBatch(point_count=2),
        plane_stress_solver="nested",
    )
    coupled = SrixNumpyCondensedPlaneStressBatch(
        SrixNumpy3DMaterialPointBatch(point_count=2),
        plane_stress_solver="coupled",
    )
    expected = nested.evaluate(strain, time_increment=1.0)
    actual = coupled.evaluate(strain, time_increment=1.0)
    assert np.allclose(
        actual.stress_in_plane_mpa,
        expected.stress_in_plane_mpa,
        rtol=1.0e-8,
        atol=1.0e-7,
    )
    assert np.allclose(
        actual.plane_stress_residual_mpa,
        expected.plane_stress_residual_mpa,
        atol=1.0e-8,
    )
    assert actual.tangent_in_plane_mpa is not None


def test_numpy_srix_coupled_direct_tangent_matches_3d_oracle() -> None:
    coupled_bridge = SrixNumpy3DMaterialPointBatch(point_count=2)
    coupled = SrixNumpyCondensedPlaneStressBatch(
        coupled_bridge, plane_stress_solver="coupled"
    )
    for strain in (
        np.array([[1.0e-3, -2.0e-4, 3.0e-4], [2.0e-3, 4.0e-4, -1.0e-4]]),
        np.array([[1.4e-3, -2.5e-4, 3.5e-4], [2.4e-3, 5.0e-4, -1.5e-4]]),
    ):
        actual = coupled.evaluate(strain, time_increment=1.0)
        trial = coupled_bridge._trial
        assert trial is not None
        tangent = coupled_bridge.tangent_from_trial(
            trial.total_strain_kelvin, trial, tangent_mode="full"
        )
        caa = tangent[:, _PLANE][:, :, _PLANE]
        cab = tangent[:, _PLANE][:, :, _TRANSVERSE]
        cba = tangent[:, _TRANSVERSE][:, :, _PLANE]
        cbb = tangent[:, _TRANSVERSE][:, :, _TRANSVERSE]
        oracle = caa - np.einsum("nij,njk->nik", cab, np.linalg.solve(cbb, cba))
        scale = (
            _KELVIN_TO_ENGINEERING_STRESS_SCALE[None, :, None]
            * _ENGINEERING_TO_KELVIN_STRAIN_SCALE[None, None, :]
        )
        assert actual.tangent_in_plane_mpa is not None
        assert np.allclose(actual.tangent_in_plane_mpa, oracle * scale, rtol=1.0e-9)
        coupled.commit()


def test_numpy_srix_fused_coupled_block_matches_numpy_block() -> None:
    strain = np.array(
        [[1.0e-3, -2.0e-4, 3.0e-4], [2.0e-3, 4.0e-4, -1.0e-4]],
    )
    reference = SrixNumpyCondensedPlaneStressBatch(
        SrixNumpy3DMaterialPointBatch(point_count=2),
        plane_stress_solver="coupled",
        coupled_block_solver="numpy",
    )
    fused = SrixNumpyCondensedPlaneStressBatch(
        SrixNumpy3DMaterialPointBatch(point_count=2),
        plane_stress_solver="coupled",
        coupled_block_solver="numba-fused",
    )
    expected = reference.evaluate(strain, time_increment=1.0)
    actual = fused.evaluate(strain, time_increment=1.0)
    assert np.allclose(actual.stress_in_plane_mpa, expected.stress_in_plane_mpa, rtol=1.0e-10)
    assert np.allclose(actual.tangent_in_plane_mpa, expected.tangent_in_plane_mpa, rtol=1.0e-10)
    assert np.allclose(actual.plane_stress_residual_mpa, expected.plane_stress_residual_mpa)


def test_numpy_srix_fused_state_block_matches_numpy_block() -> None:
    strain = np.array(
        [[1.0e-3, -2.0e-4, 3.0e-4], [2.0e-3, 4.0e-4, -1.0e-4]],
    )
    reference = SrixNumpyCondensedPlaneStressBatch(
        SrixNumpy3DMaterialPointBatch(point_count=2),
        plane_stress_solver="coupled",
        coupled_block_solver="numpy",
    )
    fused = SrixNumpyCondensedPlaneStressBatch(
        SrixNumpy3DMaterialPointBatch(point_count=2),
        plane_stress_solver="coupled",
        coupled_block_solver="numba-fused-state",
    )
    expected = reference.evaluate(strain, time_increment=1.0)
    actual = fused.evaluate(strain, time_increment=1.0)
    assert np.allclose(actual.stress_in_plane_mpa, expected.stress_in_plane_mpa, rtol=1.0e-10)
    assert np.allclose(actual.tangent_in_plane_mpa, expected.tangent_in_plane_mpa, rtol=1.0e-10)
    assert np.allclose(actual.plane_stress_residual_mpa, expected.plane_stress_residual_mpa)
