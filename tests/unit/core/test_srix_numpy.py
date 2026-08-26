"""Fast, MGIS-free checks for the NumPy SRIX backend."""

import numpy as np

from fem_inhouse.core.crystal_orientation import HomogeneousOrientationProvider
from fem_inhouse.core.srix_numpy import (
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


def test_numpy_srix_plane_stress_closes_all_three_transverse_components() -> None:
    material = SrixNumpyCondensedPlaneStressBatch(SrixNumpy3DMaterialPointBatch(point_count=2))
    trial = material.evaluate(
        np.array([[3.0e-3, -4.0e-4, 2.0e-4], [1.0e-4, 0.0, 0.0]]), time_increment=1.0
    )
    assert np.max(np.abs(trial.plane_stress_residual_mpa)) < 1.0e-7
    assert trial.tangent_in_plane_mpa is not None
