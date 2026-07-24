from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from fem_inhouse.core.mfront import (
    MFront3DCondensedPlaneStressBatch,
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
    kelvin = engineering_strain_to_kelvin(engineering)

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
