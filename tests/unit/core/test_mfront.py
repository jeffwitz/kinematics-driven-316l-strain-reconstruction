from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fem_inhouse.core.mfront import (
    MFront3DCondensedPlaneStressBatch,
    MFront3DCondensedPlaneStressBlockBatch,
    MFrontMaterialPointBatch,
    MFrontNativePlaneStressBatch,
    condense_kelvin_tangent_to_engineering,
    engineering_strain_to_kelvin,
    kelvin_strain_to_engineering,
    kelvin_stress_to_engineering,
    kelvin_tangent_to_engineering,
)
from fem_inhouse.core.plane_stress_material import LocalPlaneStressConvergenceError


def test_kelvin_strain_round_trip_preserves_engineering_shear() -> None:
    engineering = np.array([[1e-3, -2e-3, 4e-3], [0.0, 3e-4, -8e-4]])
    buffer = np.empty((2, 4))
    kelvin = engineering_strain_to_kelvin(engineering, out=buffer)

    assert kelvin is buffer
    assert kelvin.shape == (2, 4)
    np.testing.assert_allclose(kelvin_strain_to_engineering(kelvin), engineering)


def test_kelvin_stress_and_tangent_recover_plane_stress_elasticity() -> None:
    young = 205_000.0
    poisson = 0.3
    factor = young / (1 - poisson**2)
    kelvin_stress = np.array([12.0, -3.0, 0.0, np.sqrt(2.0) * 7.0])
    kelvin_tangent = np.array(
        [
            [factor, poisson * factor, 0.0, 0.0],
            [poisson * factor, factor, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, young / (1 + poisson)],
        ]
    )
    np.testing.assert_allclose(
        kelvin_stress_to_engineering(kelvin_stress),
        [12.0, -3.0, 7.0],
    )
    np.testing.assert_allclose(
        kelvin_tangent_to_engineering(kelvin_tangent),
        factor
        * np.array(
            [
                [1.0, poisson, 0.0],
                [poisson, 1.0, 0.0],
                [0.0, 0.0, (1 - poisson) / 2],
            ]
        ),
    )


def test_schur_complement_recovers_isotropic_plane_stress_elasticity() -> None:
    young = 205_000.0
    poisson = 0.3
    shear = young / (2.0 * (1.0 + poisson))
    lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    tangent = np.zeros((2, 6, 6))
    tangent[:, :3, :3] = lame
    tangent[:, np.arange(3), np.arange(3)] += 2.0 * shear
    tangent[:, 3, 3] = 2.0 * shear
    tangent[:, 4, 4] = 2.0 * shear
    tangent[:, 5, 5] = 2.0 * shear

    condensed, condition = condense_kelvin_tangent_to_engineering(tangent)
    factor = young / (1.0 - poisson**2)
    expected = factor * np.array(
        [[1.0, poisson, 0.0], [poisson, 1.0, 0.0], [0.0, 0.0, (1.0 - poisson) / 2.0]]
    )
    np.testing.assert_allclose(condensed, np.broadcast_to(expected, condensed.shape))
    np.testing.assert_allclose(condition, 1.75)


@pytest.mark.parametrize(
    ("function", "values"),
    [
        (engineering_strain_to_kelvin, np.zeros((2, 4))),
        (kelvin_strain_to_engineering, np.zeros((2, 3))),
        (kelvin_stress_to_engineering, np.zeros((2, 3))),
        (kelvin_tangent_to_engineering, np.zeros((3, 3))),
    ],
)
def test_kelvin_conversions_reject_wrong_shapes(function: object, values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        function(values)  # type: ignore[operator]


def test_mfront_batch_rejects_missing_library(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MFrontMaterialPointBatch(
            str(tmp_path) + "/missing.so",
            250.0,
            380.0,
            0.245,
        )


def test_mfront_batch_rejects_invalid_thread_count(tmp_path: Path) -> None:
    library = tmp_path / "placeholder.so"
    library.touch()
    with pytest.raises(ValueError, match="at least 1"):
        MFrontMaterialPointBatch(
            library,
            250.0,
            380.0,
            0.245,
            thread_count=0,
        )
    with pytest.raises(TypeError, match="integer"):
        MFrontMaterialPointBatch(
            library,
            250.0,
            380.0,
            0.245,
            thread_count=1.5,  # type: ignore[arg-type]
        )


@pytest.mark.mfront
def test_block_condensation_matches_single_batch_and_rolls_back() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    common: dict[str, Any] = dict(
        library_path=library,
        initial_yield_stress_mpa=np.array([250.0, 260.0, 270.0, 280.0]),
        hardening_coefficient_mpa=np.array([380.0, 390.0, 400.0, 410.0]),
        hardening_exponent=np.full(4, 0.245),
        maximum_local_iterations=15,
        local_condition_check_mode="on_failure",
    )
    single = MFront3DCondensedPlaneStressBatch(**common)
    blocked = MFront3DCondensedPlaneStressBlockBatch(**common, condensation_block_size=2)
    strain = np.array(
        [[0.001, -0.0002, 0.0004], [0.0012, -0.0001, 0.0003],
         [0.0008, 0.0001, -0.0002], [0.0011, 0.0002, 0.0001]]
    )

    single_trial = single.evaluate_in_plane(strain, time_increment=0.1)
    block_trial = blocked.evaluate_in_plane(strain, time_increment=0.1)
    np.testing.assert_allclose(
        block_trial.stress_in_plane_mpa,
        single_trial.stress_in_plane_mpa,
        rtol=0.0,
        atol=1e-12,
    )
    assert block_trial.tangent_in_plane_mpa is not None
    assert single_trial.tangent_in_plane_mpa is not None
    np.testing.assert_allclose(
        block_trial.tangent_in_plane_mpa,
        single_trial.tangent_in_plane_mpa,
        rtol=0.0,
        atol=1e-9,
    )
    single.commit()
    blocked.commit()
    snapshot = blocked.snapshot_state()
    blocked.evaluate_in_plane(strain * 1.1, time_increment=0.1)
    blocked.restore_state(snapshot)
    restored = blocked.evaluate_in_plane(strain, time_increment=0.1)
    np.testing.assert_allclose(
        restored.stress_in_plane_mpa,
        single.evaluate_in_plane(strain, time_increment=0.1).stress_in_plane_mpa,
        rtol=0.0,
        atol=1e-12,
    )
    timing = blocked.timing_statistics
    assert timing.material_block_count == 2
    assert timing.material_block_integration_calls > 0
    assert timing.material_point_integrations == (
        timing.material_point_integrations_with_tangent
        + timing.material_point_integrations_without_tangent
    )
    assert timing.material_point_integrations == (
        2 * timing.material_block_integration_calls
    )
@pytest.mark.mfront
def test_compiled_mfront_behaviour_matches_elastic_plane_stress() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    batch = MFrontMaterialPointBatch(
        library,
        1e9,
        380.0,
        0.245,
    )
    result = batch.evaluate(
        np.array([[1e-3, 0.0, 2e-3]]),
        consistent_tangent=True,
    )
    young = 205_000.0
    poisson = 0.3
    factor = young / (1 - poisson**2)
    elasticity = factor * np.array(
        [
            [1.0, poisson, 0.0],
            [poisson, 1.0, 0.0],
            [0.0, 0.0, (1 - poisson) / 2],
        ]
    )

    np.testing.assert_allclose(result.stress_mpa[0], elasticity @ [1e-3, 0.0, 2e-3])
    assert result.consistent_tangent_mpa is not None
    np.testing.assert_allclose(result.consistent_tangent_mpa[0], elasticity)
    np.testing.assert_array_equal(result.plastic_strain, np.zeros((1, 3)))
    np.testing.assert_array_equal(result.equivalent_plastic_strain, np.zeros(1))

    full_state = batch.current_full_tensor_state()
    assert batch.has_native_plane_stress_state
    assert full_state is not None
    expected_e33 = -poisson / (1.0 - poisson) * 1e-3
    assert full_state.total_strain_tensor[0, 2, 2] == pytest.approx(
        expected_e33,
        abs=1e-14,
    )
    np.testing.assert_allclose(
        full_state.total_strain_tensor,
        full_state.elastic_strain_tensor,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(full_state.plastic_strain_tensor, 0.0, atol=1e-14)
    assert abs(full_state.plane_stress_residual_mpa[0]) <= 1e-9


@pytest.mark.mfront
def test_mfront_native_plastic_state_is_additive_isochoric_and_plane_stress() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    batch = MFrontMaterialPointBatch(library, 250.0, 380.0, 0.245)
    batch.evaluate(np.array([[0.01, -0.001, 0.002]]))
    state = batch.current_full_tensor_state()

    assert state is not None
    np.testing.assert_allclose(
        state.total_strain_tensor,
        state.elastic_strain_tensor + state.plastic_strain_tensor,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.trace(state.plastic_strain_tensor, axis1=-2, axis2=-1),
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
    elastic = state.elastic_strain_tensor[0]
    expected_elastic_33 = -0.3 / 0.7 * (elastic[0, 0] + elastic[1, 1])
    assert elastic[2, 2] == pytest.approx(expected_elastic_33, abs=1e-12)
    assert abs(state.plane_stress_residual_mpa[0]) <= 1e-9

    batch.commit()
    with pytest.raises(RuntimeError, match="no successful MFront trial"):
        batch.current_full_tensor_state()


@pytest.mark.mfront
def test_native_light_paths_defer_tangent_and_tensor_reconstruction() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    batch = MFrontNativePlaneStressBatch(
        library,
        [250.0, 270.0],
        [380.0, 420.0],
        [0.245, 0.245],
        behaviour_name="PixelMicromorphicLudwikJ2Plasticity",
        micromorphic_coupling_modulus_mpa=4_000.0,
    )
    batch.set_nonlocal_equivalent_plastic_strain([0.001, 0.002])
    strain = np.tile([0.008, -0.0008, 0.0015], (2, 1))

    peeq = batch.evaluate_equivalent_plastic_strain(strain, time_increment=0.1)
    first_timing = batch.timing_statistics
    assert np.all(peeq > 0)
    assert first_timing.integration_without_tangent_seconds > 0
    assert first_timing.integration_without_tangent_calls == 1
    assert first_timing.integration_with_tangent_seconds == 0
    assert first_timing.integration_with_tangent_calls == 0
    assert first_timing.tensor_reconstruction_seconds == 0
    assert first_timing.tensor_reconstruction_calls == 0

    in_plane = batch.evaluate_in_plane(strain, time_increment=0.1)
    second_timing = batch.timing_statistics
    assert in_plane.tangent_in_plane_mpa is not None
    assert second_timing.integration_with_tangent_seconds > 0
    assert second_timing.integration_with_tangent_calls == 1
    assert second_timing.tensor_reconstruction_seconds == 0

    full = batch.complete_trial(in_plane)
    assert full.full_stress_tensor_mpa.shape == (2, 3, 3)
    assert batch.timing_statistics.tensor_reconstruction_seconds > 0
    assert batch.timing_statistics.tensor_reconstruction_calls == 1


@pytest.mark.mfront
def test_mfront_trial_can_be_reverted_before_commit() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    batch = MFrontMaterialPointBatch(
        library,
        250.0,
        380.0,
        0.245,
    )
    plastic_trial = batch.evaluate(np.array([[0.005, 0.0, 0.0]]))
    assert plastic_trial.equivalent_plastic_strain[0] > 0
    batch.revert()

    zero_trial = batch.evaluate(np.zeros((1, 3)))
    np.testing.assert_allclose(zero_trial.stress_mpa, 0.0, atol=1e-12)
    np.testing.assert_allclose(zero_trial.equivalent_plastic_strain, 0.0, atol=1e-15)


@pytest.mark.mfront
def test_mfront_new_trial_automatically_discards_previous_trial() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    batch = MFrontMaterialPointBatch(
        library,
        250.0,
        380.0,
        0.245,
    )
    plastic_trial = batch.evaluate(np.array([[0.005, 0.0, 0.0]]))
    assert plastic_trial.equivalent_plastic_strain[0] > 0

    zero_trial = batch.evaluate(np.zeros((1, 3)))
    np.testing.assert_allclose(zero_trial.stress_mpa, 0.0, atol=1e-12)
    np.testing.assert_allclose(zero_trial.equivalent_plastic_strain, 0.0, atol=1e-15)


@pytest.mark.mfront
def test_mfront_thread_pool_matches_serial_integration() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    material_points = 32
    material = (
        np.linspace(230.0, 270.0, material_points),
        np.linspace(330.0, 430.0, material_points),
        np.full(material_points, 0.245),
    )
    serial = MFrontMaterialPointBatch(library, *material)
    parallel = MFrontMaterialPointBatch(library, *material, thread_count=2)
    strain = np.tile([0.006, -0.0006, 0.002], (material_points, 1))

    serial_result = serial.evaluate(strain)
    parallel_result = parallel.evaluate(strain)

    np.testing.assert_allclose(parallel_result.stress_mpa, serial_result.stress_mpa)
    np.testing.assert_allclose(
        parallel_result.equivalent_plastic_strain,
        serial_result.equivalent_plastic_strain,
    )
    assert serial_result.consistent_tangent_mpa is not None
    assert parallel_result.consistent_tangent_mpa is not None
    np.testing.assert_allclose(
        parallel_result.consistent_tangent_mpa,
        serial_result.consistent_tangent_mpa,
    )
    assert serial.linear_system_matrix_type == "symmetric_positive_definite"
    assert parallel.linear_system_matrix_type == "symmetric_positive_definite"


@pytest.mark.mfront
def test_micromorphic_thread_pool_is_reproducible() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    point_count = 64
    material = (
        np.linspace(230.0, 270.0, point_count),
        np.linspace(330.0, 430.0, point_count),
        np.full(point_count, 0.245),
    )
    options = {
        "behaviour_name": "PixelMicromorphicLudwikJ2Plasticity",
        "micromorphic_coupling_modulus_mpa": 2_000.0,
    }
    serial = MFrontMaterialPointBatch(library, *material, **options)
    parallel = MFrontMaterialPointBatch(
        library,
        *material,
        thread_count=4,
        **options,
    )
    chi = np.linspace(0.0, 0.004, point_count)
    strain = np.tile([0.006, -0.0006, 0.002], (point_count, 1))
    serial.set_nonlocal_equivalent_plastic_strain(chi)
    parallel.set_nonlocal_equivalent_plastic_strain(chi)

    serial_result = serial.evaluate(strain)
    parallel_result = parallel.evaluate(strain)

    np.testing.assert_array_equal(parallel_result.stress_mpa, serial_result.stress_mpa)
    np.testing.assert_array_equal(
        parallel_result.equivalent_plastic_strain,
        serial_result.equivalent_plastic_strain,
    )
    np.testing.assert_array_equal(
        parallel_result.yield_surface_radius_mpa,
        serial_result.yield_surface_radius_mpa,
    )
    np.testing.assert_array_equal(
        parallel_result.consistent_tangent_mpa,
        serial_result.consistent_tangent_mpa,
    )


@pytest.mark.mfront
def test_micromorphic_hchi_zero_exactly_matches_reference_behaviour() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    material = (np.array([240.0, 270.0]), np.array([360.0, 420.0]), np.full(2, 0.245))
    reference = MFrontMaterialPointBatch(library, *material)
    micromorphic = MFrontMaterialPointBatch(
        library,
        *material,
        behaviour_name="PixelMicromorphicLudwikJ2Plasticity",
        micromorphic_coupling_modulus_mpa=0.0,
    )
    micromorphic.set_nonlocal_equivalent_plastic_strain([0.0, 0.02])
    strain = np.tile([0.008, -0.0008, 0.0015], (2, 1))

    reference_trial = reference.evaluate(strain)
    coupled_trial = micromorphic.evaluate(strain)

    np.testing.assert_array_equal(coupled_trial.stress_mpa, reference_trial.stress_mpa)
    np.testing.assert_array_equal(
        coupled_trial.equivalent_plastic_strain,
        reference_trial.equivalent_plastic_strain,
    )
    np.testing.assert_array_equal(
        coupled_trial.yield_surface_radius_mpa,
        reference_trial.yield_surface_radius_mpa,
    )
    np.testing.assert_array_equal(
        coupled_trial.consistent_tangent_mpa,
        reference_trial.consistent_tangent_mpa,
    )


@pytest.mark.mfront
def test_micromorphic_correction_has_registered_sign_and_is_transactional() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    batch = MFrontMaterialPointBatch(
        library,
        [250.0, 250.0],
        [500.0, 500.0],
        [0.245, 0.245],
        behaviour_name="PixelMicromorphicLudwikJ2Plasticity",
        micromorphic_coupling_modulus_mpa=5_000.0,
    )
    batch.set_nonlocal_equivalent_plastic_strain([0.0, 0.01])
    trial = batch.evaluate(np.tile([0.005, 0.0, 0.0], (2, 1)))
    mismatch = trial.equivalent_plastic_strain - np.array([0.0, 0.01])

    assert mismatch[0] > 0
    assert mismatch[1] < 0
    assert trial.yield_surface_radius_mpa[0] > trial.yield_surface_radius_mpa[1]
    batch.commit()
    np.testing.assert_array_equal(
        batch.committed_nonlocal_equivalent_plastic_strain,
        [0.0, 0.01],
    )

    batch.set_nonlocal_equivalent_plastic_strain([0.02, 0.02])
    batch.evaluate(np.tile([0.006, 0.0, 0.0], (2, 1)))
    batch.revert()
    np.testing.assert_array_equal(
        batch.committed_nonlocal_equivalent_plastic_strain,
        [0.0, 0.01],
    )


@pytest.mark.mfront
def test_micromorphic_native_tangent_matches_finite_differences_at_fixed_chi() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    batch = MFrontNativePlaneStressBatch(
        library,
        [250.0],
        [380.0],
        [0.245],
        behaviour_name="PixelMicromorphicLudwikJ2Plasticity",
        micromorphic_coupling_modulus_mpa=2_000.0,
    )
    batch.set_nonlocal_equivalent_plastic_strain([0.002])
    target = np.array([[0.008, 0.0003, 0.002]])
    base = batch.evaluate(target, time_increment=1.0)
    assert base.tangent_in_plane_mpa is not None
    numerical = np.empty((1, 3, 3))
    step = 1e-7
    for component in range(3):
        plus = target.copy()
        minus = target.copy()
        plus[:, component] += step
        minus[:, component] -= step
        stress_plus = batch.evaluate(plus, time_increment=1.0).stress_in_plane_mpa
        stress_minus = batch.evaluate(minus, time_increment=1.0).stress_in_plane_mpa
        numerical[:, :, component] = (stress_plus - stress_minus) / (2.0 * step)
    np.testing.assert_allclose(
        base.tangent_in_plane_mpa,
        numerical,
        rtol=2e-5,
        atol=5e-2,
    )


@pytest.mark.mfront
def test_mfront_3d_condensation_matches_native_plane_stress_histories() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    material = (
        np.array([240.0, 270.0]),
        np.array([360.0, 420.0]),
        np.full(2, 0.245),
    )
    native = MFrontNativePlaneStressBatch(library, *material)
    condensed = MFront3DCondensedPlaneStressBatch(library, *material)
    history = (
        [0.001, -0.0002, 0.0005],
        [0.008, -0.0008, 0.0015],
        [0.008, 0.002, 0.003],
        [0.004, 0.001, 0.001],
    )
    for values in history:
        strain = np.tile(values, (2, 1))
        native_trial = native.evaluate(strain, time_increment=0.25)
        condensed_trial = condensed.evaluate(strain, time_increment=0.25)
        np.testing.assert_allclose(
            condensed_trial.stress_in_plane_mpa,
            native_trial.stress_in_plane_mpa,
            rtol=1e-7,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            condensed_trial.full_strain_tensor,
            native_trial.full_strain_tensor,
            rtol=1e-7,
            atol=5e-12,
        )
        np.testing.assert_allclose(
            condensed_trial.observables["equivalent_plastic_strain"],
            native_trial.observables["equivalent_plastic_strain"],
            rtol=1e-7,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            condensed_trial.full_strain_tensor[:, (0, 1), 2],
            0.0,
            atol=1e-14,
        )
        assert np.max(np.abs(condensed_trial.plane_stress_residual_mpa)) < 1e-6
        native.commit()
        condensed.commit()


@pytest.mark.mfront
def test_micromorphic_3d_condensation_matches_native_plane_stress() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    material = (np.full(2, 250.0), np.full(2, 380.0), np.full(2, 0.245))
    native = MFrontNativePlaneStressBatch(
        library,
        *material,
        behaviour_name="PixelMicromorphicLudwikJ2Plasticity",
        micromorphic_coupling_modulus_mpa=1_000.0,
    )
    condensed = MFront3DCondensedPlaneStressBatch(
        library,
        *material,
        behaviour_name="PixelMicromorphicLudwikJ2Plasticity3D",
        micromorphic_coupling_modulus_mpa=1_000.0,
    )
    nonlocal_peeq = np.array([0.001, 0.003])
    native.set_nonlocal_equivalent_plastic_strain(nonlocal_peeq)
    condensed.set_nonlocal_equivalent_plastic_strain(nonlocal_peeq)
    strain = np.tile([0.008, -0.0008, 0.0015], (2, 1))

    native_trial = native.evaluate(strain, time_increment=1.0)
    condensed_trial = condensed.evaluate(strain, time_increment=1.0)

    assert native.linear_system_matrix_type == "symmetric_positive_definite"
    assert condensed.linear_system_matrix_type == "symmetric_positive_definite"
    np.testing.assert_allclose(
        condensed_trial.stress_in_plane_mpa,
        native_trial.stress_in_plane_mpa,
        rtol=1e-7,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        condensed_trial.observables["equivalent_plastic_strain"],
        native_trial.observables["equivalent_plastic_strain"],
        rtol=1e-7,
        atol=1e-11,
    )
    assert np.max(np.abs(condensed_trial.plane_stress_residual_mpa)) < 1e-6


@pytest.mark.mfront
def test_mfront_3d_condensed_tangent_matches_finite_differences() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    batch = MFront3DCondensedPlaneStressBatch(
        library,
        np.full(2, 250.0),
        np.full(2, 380.0),
        np.full(2, 0.245),
    )
    batch.evaluate(np.tile([0.006, -0.0005, 0.001], (2, 1)), time_increment=0.5)
    batch.commit()
    target = np.tile([0.008, 0.0003, 0.002], (2, 1))
    base = batch.evaluate(target, time_increment=0.5)
    assert base.tangent_in_plane_mpa is not None
    numerical = np.empty((2, 3, 3))
    step = 1e-7
    for component in range(3):
        plus = target.copy()
        minus = target.copy()
        plus[:, component] += step
        minus[:, component] -= step
        stress_plus = batch.evaluate(plus, time_increment=0.5).stress_in_plane_mpa
        stress_minus = batch.evaluate(minus, time_increment=0.5).stress_in_plane_mpa
        numerical[:, :, component] = (stress_plus - stress_minus) / (2.0 * step)
    np.testing.assert_allclose(
        base.tangent_in_plane_mpa,
        numerical,
        rtol=2e-5,
        atol=5e-2,
    )


@pytest.mark.mfront
def test_failed_local_condensation_does_not_pollute_committed_state() -> None:
    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    batch = MFront3DCondensedPlaneStressBatch(
        library,
        np.full(2, 250.0),
        np.full(2, 380.0),
        np.full(2, 0.245),
        maximum_local_iterations=1,
    )
    with pytest.raises(LocalPlaneStressConvergenceError, match="did not converge"):
        batch.evaluate(np.tile([0.01, -0.001, 0.002], (2, 1)), time_increment=1.0)
    zero = batch.evaluate(np.zeros((2, 3)), time_increment=1.0)
    np.testing.assert_allclose(zero.stress_in_plane_mpa, 0.0, atol=1e-14)
    np.testing.assert_allclose(
        zero.observables["equivalent_plastic_strain"],
        0.0,
        atol=1e-15,
    )
    assert batch.statistics.local_plane_stress_failures == 1


# ---------------------------------------------------------------------------
# UMAT GPS regression tests (consolidation 2026-08-08).
#
# Each of the three production defects found between 2026-08-06 and
# 2026-08-08 is pinned here as a permanent regression test: the total strain
# applied as an increment (6bfaf86), the per-point rotations handed over as
# strided views with dangling buffers (4deeffb), and the single-threaded
# integration (b201ae0). Plus the anti-vacuous finite-difference tangent
# check at a genuine non-zero increment.
# ---------------------------------------------------------------------------


def _gps_batch(
    library: str,
    *,
    point_count: int = 1,
    orientation: tuple[float, float, float] | None = None,
    thread_count: int = 1,
) -> Any:
    from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch

    options: dict[str, object] = {
        "parameter_set": "316l_srix_transposed_from_nasri2018_rate_1e-3"
    }
    if orientation is not None:
        from fem_inhouse.core.crystal_orientation import rotation_from_euler_bunge_deg

        options["crystal_orientation"] = {
            "mode": "homogeneous",
            "matrix": np.asarray(
                rotation_from_euler_bunge_deg(*orientation), dtype=float
            ).tolist(),
        }
    return create_plane_stress_material_batch(
        "mfront-native-generalised-plane-stress",
        np.full((point_count, 1), 250.0),
        np.full((point_count, 1), 500.0),
        0.245,
        young_modulus_mpa=205000.0,
        poisson_ratio=0.3,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1000,
        first_positive_plastic_strain=1e-6,
        mfront_library=library,
        mfront_threads=thread_count,
        mfront_behaviour_id="fcc_forest_rubin_srix",
        constitutive_options=options,
    )


@pytest.mark.mfront
def test_gps_bridge_writes_the_total_strain_absolutely() -> None:
    """The evaluate contract is TOTAL strain, applied absolutely.

    Regression for 6bfaf86: the bridge added the total to the committed
    gradient, so the imposed strain accumulated as 1+2+3+... instead of
    1,2,3. Increment 1 was unaffected (total = increment), which hid the bug
    from every single-increment comparison.
    """

    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    pytest.importorskip("mgis.behaviour")
    from fem_inhouse.core.mfront import (
        _ENGINEERING_TO_KELVIN_STRAIN_SCALE,
        _PLANE_STRESS_COMPONENTS,
    )

    first = np.array([[0.003, -0.0012, 0.0]])
    second = np.array([[0.004, -0.0016, 0.0]])
    stepped = _gps_batch(library)
    stepped.evaluate(first, time_increment=0.5)
    stepped.commit()
    stepped.evaluate(second, time_increment=0.5)
    # The gradient written into the manager is the ABSOLUTE total, not the
    # committed state plus the total.
    written = np.asarray(stepped._manager.s1.gradients)[0].copy()
    np.testing.assert_allclose(
        written[_PLANE_STRESS_COMPONENTS],
        second[0] * _ENGINEERING_TO_KELVIN_STRAIN_SCALE,
        rtol=0.0,
        atol=1e-14,
    )
    # And the response equals a fresh batch driven through the same history
    # (the law is path-dependent, so the fresh batch must share the first
    # increment to be comparable).
    fresh = _gps_batch(library)
    fresh.evaluate(first, time_increment=0.5)
    fresh.commit()
    stepped_trial = stepped.evaluate(second, time_increment=0.5)
    fresh_trial = fresh.evaluate(second, time_increment=0.5)
    np.testing.assert_allclose(
        stepped_trial.stress_in_plane_mpa,
        fresh_trial.stress_in_plane_mpa,
        rtol=0.0,
        atol=1e-9,
    )


@pytest.mark.mfront
def test_gps_bridge_applies_per_point_rotations_independently() -> None:
    """Each material point must read its OWN rotation components.

    Regression for 4deeffb: the nine Q components were handed to MGIS as a
    strided view under ExternalStorage, so every point read another point's
    rotation, and the buffers were temporaries freed at the end of the loop.
    A single point hides both; two points with different orientations expose
    them.
    """

    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    pytest.importorskip("mgis.behaviour")

    strain = np.array([[0.003, -0.0012, 0.0005]])
    identity = _gps_batch(library, orientation=None)
    tilted = _gps_batch(library, orientation=(35.0, 20.0, 15.0))
    identity_trial = identity.evaluate(strain, time_increment=0.5)
    tilted_trial = tilted.evaluate(strain, time_increment=0.5)
    # The two orientations must give different responses to the same strain.
    difference = identity_trial.stress_in_plane_mpa - tilted_trial.stress_in_plane_mpa
    assert np.max(np.abs(difference)) > 1.0

    from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch

    both = create_plane_stress_material_batch(
        "mfront-native-generalised-plane-stress",
        np.full((2, 1), 250.0),
        np.full((2, 1), 500.0),
        0.245,
        young_modulus_mpa=205000.0,
        poisson_ratio=0.3,
        hardening_mode="ludwik",
        plastic_strain_max=0.2,
        plastic_table_points=1000,
        first_positive_plastic_strain=1e-6,
        mfront_library=library,
        mfront_threads=1,
        mfront_behaviour_id="fcc_forest_rubin_srix",
        constitutive_options={
            "parameter_set": "316l_srix_transposed_from_nasri2018_rate_1e-3",
            "crystal_orientation": {
                "mode": "ebsd",
                "euler_bunge_deg": np.array(
                    [[[0.0, 0.0, 0.0]], [[35.0, 20.0, 15.0]]], dtype=float
                ),
            },
        },
    )
    both_trial = both.evaluate(np.vstack([strain, strain]), time_increment=0.5)
    # Point 0 behaves as the identity single point, point 1 as the tilted one.
    np.testing.assert_allclose(
        both_trial.stress_in_plane_mpa[0],
        identity_trial.stress_in_plane_mpa[0],
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        both_trial.stress_in_plane_mpa[1],
        tilted_trial.stress_in_plane_mpa[0],
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.mfront
def test_gps_bridge_thread_count_matches_serial_results() -> None:
    """Four threads must give the same material response as one.

    Regression for b201ae0: the GPS bridge integrated single-threaded while
    the reference used four, silently paying the thread-count cost.
    """

    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    pytest.importorskip("mgis.behaviour")

    serial = _gps_batch(library, thread_count=1, orientation=(35.0, 20.0, 15.0))
    parallel = _gps_batch(library, thread_count=4, orientation=(35.0, 20.0, 15.0))
    history = np.array(
        [
            [0.001, -0.0004, 0.0],
            [0.002, -0.0008, 0.0002],
            [0.003, -0.0012, 0.0],
        ]
    )
    for step in history:
        serial_trial = serial.evaluate(np.atleast_2d(step), time_increment=1.0 / 3)
        parallel_trial = parallel.evaluate(np.atleast_2d(step), time_increment=1.0 / 3)
        np.testing.assert_allclose(
            serial_trial.stress_in_plane_mpa,
            parallel_trial.stress_in_plane_mpa,
            rtol=0.0,
            atol=1e-9,
        )
        serial.commit()
        parallel.commit()
    assert parallel.timing_statistics.native_thread_count == 4
    assert serial.timing_statistics.native_thread_count == 1


@pytest.mark.mfront
def test_gps_plastic_tangent_matches_finite_differences_at_real_increment() -> None:
    """The FD tangent is checked at a genuine non-zero increment.

    The first version of the A6 check probed the law at a strain increment of
    exactly zero -- the guarded elastic branch -- and reported a pass on a
    tangent that was wrong by a factor of ten at plastic states. This test
    commits through a plastic history and probes the NEXT increment, so every
    finite-difference probe carries a real plastic increment.
    """

    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    pytest.importorskip("mgis.behaviour")

    batch = _gps_batch(library, orientation=(35.0, 20.0, 15.0))
    history = np.array(
        [[i / 12 * 0.02, -0.4 * i / 12 * 0.02, 0.0] for i in range(1, 4)]
    )
    for step in history:
        batch.evaluate(np.atleast_2d(step), time_increment=1.0 / 12)
        batch.commit()
    # The next increment is a genuine plastic step, not zero.
    target = np.array([[4 / 12 * 0.02, -0.4 * 4 / 12 * 0.02, 0.0]])
    trial = batch.evaluate(target, time_increment=1.0 / 12)
    tangent_returned = np.asarray(trial.tangent_in_plane_mpa)[0]
    fd = np.zeros((3, 3))
    perturbation = 1.0e-6
    for column in range(3):
        plus = target.copy()
        minus = target.copy()
        plus[:, column] += perturbation
        minus[:, column] -= perturbation
        stress_plus = np.asarray(
            batch.evaluate(plus, time_increment=1.0 / 12).stress_in_plane_mpa
        )[0]
        stress_minus = np.asarray(
            batch.evaluate(minus, time_increment=1.0 / 12).stress_in_plane_mpa
        )[0]
        fd[:, column] = (stress_plus - stress_minus) / (2 * perturbation)
    batch.revert()
    relative_error = np.max(np.abs(tangent_returned - fd)) / max(
        np.max(np.abs(fd)), 1.0e-30
    )
    assert relative_error <= 1.0e-6


@pytest.mark.mfront
def test_gps_deep_history_is_deterministic_and_transactional() -> None:
    """Deep plastic increments: deterministic results, clean commit/revert.

    The sub-stepping advances and restores the committed state internally;
    commit() and revert() must keep their meaning, and a re-run from the same
    committed state must give bit-identical results.
    """

    library = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY")
    if library is None:
        pytest.skip("MFRONT_BEHAVIOUR_LIBRARY is not set")
    pytest.importorskip("mgis.behaviour")

    history = np.array(
        [[i / 12 * 0.02, -0.4 * i / 12 * 0.02, 0.0] for i in range(1, 13)]
    )
    def _drive() -> list[np.ndarray]:
        batch = _gps_batch(library)
        pass_results: list[np.ndarray] = []
        for step in history:
            trial = batch.evaluate(np.atleast_2d(step), time_increment=1.0 / 12)
            pass_results.append(np.asarray(trial.stress_in_plane_mpa).copy())
            batch.commit()
        return pass_results

    # Determinism: two identical passes from the virgin state reproduce each
    # other to machine precision.
    first_pass = _drive()
    second_pass = _drive()
    batch = _gps_batch(library)
    for step in history:
        batch.evaluate(np.atleast_2d(step), time_increment=1.0 / 12)
        batch.commit()
    for first, second in zip(first_pass, second_pass, strict=True):
        np.testing.assert_array_equal(first, second)
    # Revert from the final committed state restores it exactly.
    snapshot = batch.snapshot_state()
    batch.evaluate(np.atleast_2d(history[-1]), time_increment=1.0 / 12)
    batch.revert()
    restored = batch.snapshot_state()
    np.testing.assert_array_equal(restored[0], snapshot[0])
