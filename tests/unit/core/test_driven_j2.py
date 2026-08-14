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


def test_the_modal_basis_really_diagonalises_both_operators() -> None:
    """The premise the scalar return is built on, asserted rather than assumed.

    Collapsing the 3x3 local system to one scalar equation is legitimate only
    because the plane-stress elasticity and the von Mises metric commute. If a
    future change to either matrix broke that, the return would keep converging
    and would quietly converge to the wrong stress, so the property is checked
    here rather than trusted.
    """

    from fem_inhouse.core.driven_j2 import (
        _MODAL_BASIS,
        _MODAL_METRIC_EIGENVALUES,
        _modal_elasticity_eigenvalues,
    )

    elasticity = plane_stress_elasticity(YOUNG, POISSON)
    np.testing.assert_allclose(_MODAL_BASIS @ _MODAL_BASIS.T, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(
        _MODAL_BASIS @ elasticity @ _MODAL_BASIS.T,
        np.diag(_modal_elasticity_eigenvalues(YOUNG, POISSON)),
        rtol=1e-13,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        _MODAL_BASIS @ PLANE_STRESS_VON_MISES_METRIC @ _MODAL_BASIS.T,
        np.diag(_MODAL_METRIC_EIGENVALUES),
        atol=1e-15,
    )


def test_the_scalar_return_zeroes_the_local_residual_across_the_admissible_range() -> None:
    """The equation the solver claims to solve, checked at its own solution.

    Sweeping to 99.9 percent of the admissible bound is the point: the old
    3x3 Newton with a backtracking line search failed exactly where the trial
    state runs out of stress to relax, and a test that stays in the comfortable
    middle of the range would not have seen it.
    """

    rng = np.random.default_rng(20260814)
    points = 4096
    material = _material(points)
    strain = rng.normal(scale=4.0e-3, size=(points, 3))
    bound = material.maximum_admissible_equivalent_plastic_increment(strain)
    increment = rng.uniform(0.0, 0.999, size=points) * bound

    trial = material.evaluate(strain, increment, time_increment=1.0)
    elasticity = plane_stress_elasticity(YOUNG, POISSON)
    trial_stress = strain @ elasticity.T
    direction = np.asarray(trial.observables["flow_direction"])
    residual = (
        trial.stress_in_plane_mpa - trial_stress + increment[:, None] * (direction @ elasticity.T)
    )
    scale = np.maximum(np.linalg.norm(trial_stress, axis=1), 1.0)
    assert float(np.max(np.linalg.norm(residual, axis=1) / scale)) < 1.0e-13


def test_an_increment_beyond_the_admissible_bound_is_named_not_merely_refused() -> None:
    """Non-existence must arrive as a quantity, not as a solver giving up.

    Associated J2 drives the deviatoric stress to the origin at a finite
    ``Delta p``; past it no state with ``q > 0`` exists. The old code met that
    wall as `local line search failed at point 117`, which reads like a
    conditioning accident and sent the investigation towards branch following.
    The bound is closed form, so the message can carry both numbers and a
    caller can project onto the admissible set instead of guessing.
    """

    material = _material(1)
    strain = np.array([[3.0e-3, -1.0e-3, 5.0e-4]])
    bound = float(material.maximum_admissible_equivalent_plastic_increment(strain)[0])
    assert bound > 0.0

    # Just inside the wall the solve is expected to work, and the returned
    # equivalent stress is nearly fully relaxed.
    inside = material.evaluate(strain, np.array([0.999 * bound]), time_increment=1.0)
    from fem_inhouse.core.constitutive import von_mises

    trial_equivalent = float(von_mises(strain @ plane_stress_elasticity(YOUNG, POISSON).T)[0])
    assert float(von_mises(inside.stress_in_plane_mpa)[0]) < 0.01 * trial_equivalent

    material.revert()
    with pytest.raises(ConstitutiveIntegrationError, match="exceeds what the trial state") as info:
        material.evaluate(strain, np.array([1.001 * bound]), time_increment=1.0)
    diagnostics = info.value.diagnostics
    assert diagnostics["failure_stage"] == "delta_p_above_admissible_bound"
    assert diagnostics["delta_p"] > diagnostics["delta_p_max"] > 0.0


def test_the_scalar_return_reproduces_the_classical_radial_return_when_it_is_exact() -> None:
    """A pure shear state has one relaxation eigenvalue, so the answer is closed form.

    With only the shear mode populated the equation degenerates to
    ``q = q_trial - a``, which is the textbook radial return. It is the one
    case where the scalar solve can be checked against arithmetic instead of
    against another solver.
    """

    material = _material(1)
    strain = np.array([[0.0, 0.0, 2.0e-3]])
    elasticity = plane_stress_elasticity(YOUNG, POISSON)
    trial_stress = strain @ elasticity.T
    from fem_inhouse.core.constitutive import von_mises

    trial_equivalent = float(von_mises(trial_stress)[0])
    shear_relaxation = 3.0 * YOUNG / (2.0 * (1.0 + POISSON))
    increment = 0.3 * trial_equivalent / shear_relaxation

    trial = material.evaluate(strain, np.array([increment]), time_increment=1.0)
    expected = trial_equivalent - increment * shear_relaxation
    assert float(von_mises(trial.stress_in_plane_mpa)[0]) == pytest.approx(expected, rel=1e-12)
