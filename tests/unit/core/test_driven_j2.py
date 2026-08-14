from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.core.constitutive import (
    PLANE_STRESS_VON_MISES_METRIC,
    make_hardening,
    return_mapping,
)
from fem_inhouse.core.driven_j2 import DrivenJ2PlaneStressBatch
from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.core.plane_stress_material import ConstitutiveIntegrationError

YOUNG = 205_000.0
POISSON = 0.30


def _material(point_count: int = 1) -> DrivenJ2PlaneStressBatch:
    return DrivenJ2PlaneStressBatch(
        point_count,
        young_modulus_mpa=YOUNG,
        poisson_ratio=POISSON,
    )


def _response(strain: np.ndarray, increment: np.ndarray):
    return _material(len(strain)).evaluate(
        strain,
        increment,
        time_increment=1.0,
    )


def test_driven_increment_reproduces_the_standard_j2_returned_stress() -> None:
    elasticity = plane_stress_elasticity(YOUNG, POISSON)
    metric_product = elasticity @ PLANE_STRESS_VON_MISES_METRIC
    hardening, _ = make_hardening(0.245)
    strain = np.array([[0.005, -0.001, 0.002]])
    expected_stress, _, returned_increment = return_mapping(
        strain @ elasticity.T,
        np.zeros(1),
        np.array([250.0]),
        np.array([500.0]),
        hardening,
        metric_product[0, 0],
        metric_product[0, 1],
        metric_product[2, 2],
    )

    trial = _response(strain, returned_increment)

    assert returned_increment[0] > 0.0
    np.testing.assert_allclose(
        trial.stress_in_plane_mpa,
        expected_stress,
        rtol=2.0e-12,
        atol=1.0e-8,
    )
    assert trial.local_residual_norm_mpa[0] < 1.0e-9


def test_both_driven_j2_tangents_match_central_differences() -> None:
    strain = np.array([[0.005, -0.001, 0.002]])
    increment = np.array([0.002])
    trial = _response(strain, increment)
    step = 1.0e-7

    strain_fd = np.empty((3, 3))
    for component in range(3):
        plus = strain.copy()
        minus = strain.copy()
        plus[0, component] += step
        minus[0, component] -= step
        strain_fd[:, component] = (
            _response(plus, increment).stress_in_plane_mpa[0]
            - _response(minus, increment).stress_in_plane_mpa[0]
        ) / (2.0 * step)
    increment_fd = (
        _response(strain, increment + step).stress_in_plane_mpa[0]
        - _response(strain, increment - step).stress_in_plane_mpa[0]
    ) / (2.0 * step)

    assert trial.tangent_in_plane_mpa is not None
    strain_error = np.linalg.norm(trial.tangent_in_plane_mpa[0] - strain_fd) / np.linalg.norm(
        strain_fd
    )
    increment_error = np.linalg.norm(
        trial.stress_equivalent_plastic_increment_tangent_mpa[0] - increment_fd
    ) / np.linalg.norm(increment_fd)
    assert strain_error < 1.0e-8
    assert increment_error < 1.0e-8


def test_trials_are_repeatable_and_only_commit_mutates_the_state() -> None:
    material = _material()
    strain = np.array([[0.004, -0.0005, 0.001]])
    increment = np.array([0.001])

    first = material.evaluate(strain, increment, time_increment=1.0)
    second = material.evaluate(strain, increment, time_increment=1.0)
    np.testing.assert_array_equal(first.stress_in_plane_mpa, second.stress_in_plane_mpa)
    np.testing.assert_array_equal(material.committed_equivalent_plastic_strain, 0.0)
    np.testing.assert_array_equal(material.committed_plastic_strain, 0.0)

    material.revert()
    np.testing.assert_array_equal(material.committed_equivalent_plastic_strain, 0.0)
    material.evaluate(strain, increment, time_increment=1.0)
    material.commit()
    np.testing.assert_allclose(material.committed_equivalent_plastic_strain, increment)
    committed_plastic = material.committed_plastic_strain

    material.evaluate(strain * 1.2, increment * 0.5, time_increment=0.5)
    material.revert()
    np.testing.assert_allclose(material.committed_equivalent_plastic_strain, increment)
    np.testing.assert_array_equal(material.committed_plastic_strain, committed_plastic)


def test_zero_increment_is_elastic_and_positive_increment_needs_a_direction() -> None:
    material = _material(2)
    trial = material.evaluate(
        np.array([[1.0e-4, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.zeros(2),
        time_increment=1.0,
    )
    expected = np.einsum(
        "ij,pj->pi",
        plane_stress_elasticity(YOUNG, POISSON),
        np.array([[1.0e-4, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    np.testing.assert_allclose(trial.stress_in_plane_mpa, expected)
    np.testing.assert_array_equal(trial.stress_equivalent_plastic_increment_tangent_mpa[1], 0.0)

    with pytest.raises(ConstitutiveIntegrationError, match="non-zero trial J2 direction"):
        _material().evaluate(
            np.zeros((1, 3)),
            np.array([1.0e-4]),
            time_increment=1.0,
        )


def test_invalid_driven_increment_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _material().evaluate(
            np.array([[1.0e-3, 0.0, 0.0]]),
            np.array([-1.0e-4]),
            time_increment=1.0,
        )
