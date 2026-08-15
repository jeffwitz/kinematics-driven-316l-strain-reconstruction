from __future__ import annotations

import numpy as np

from fem_inhouse.core.constitutive import PLANE_STRESS_VON_MISES_METRIC
from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.core.kelvin import (
    equivalent_plastic_strain,
    stiffness_from_engineering,
    strain_from_engineering,
    strain_operator_from_engineering,
    stress_from_voigt,
    three_dimensional_from_plane_stress_plastic,
)

YOUNG = 205_000.0
POISSON = 0.30


def _tensor(components: np.ndarray) -> np.ndarray:
    """The full 2x2 symmetric tensor behind a Kelvin triple."""

    root = np.sqrt(2.0)
    return np.array(
        [
            [components[0], components[2] / root],
            [components[2] / root, components[1]],
        ]
    )


def test_the_kelvin_dot_product_is_the_tensor_contraction() -> None:
    """The whole point: `A : B` with no metric and no factor of two."""

    generator = np.random.default_rng(1)
    for _ in range(20):
        first = generator.normal(size=3)
        second = generator.normal(size=3)
        contraction = float((_tensor(first) * _tensor(second)).sum())
        assert float(first @ second) == np.float64(contraction) or np.isclose(
            first @ second, contraction, rtol=1e-13
        )


def test_stress_and_strain_pair_to_the_correct_work() -> None:
    """Mixed conventions are where the factor of two hides.

    Stress arrives in Voigt and strain in engineering, and they scale in
    opposite directions. The product of the converted pair must equal the work
    computed the old way, `sigma_voigt . eps_engineering`.
    """

    generator = np.random.default_rng(2)
    stress_voigt = generator.normal(size=3) * 100.0
    strain_engineering = generator.normal(size=3) * 1.0e-3
    old = float(stress_voigt @ strain_engineering)
    new = float(stress_from_voigt(stress_voigt) @ strain_from_engineering(strain_engineering))
    assert np.isclose(new, old, rtol=1e-13)


def test_the_isotropic_shear_stiffness_doubles_in_kelvin() -> None:
    """`C^K = 2G` where the engineering form has `G`, which looks wrong and is not.

    Verified against the repository's own matrix rather than asserted from
    theory, so a change to `plane_stress_elasticity` is caught here.
    """

    engineering = plane_stress_elasticity(YOUNG, POISSON)
    shear = YOUNG / (2.0 * (1.0 + POISSON))
    assert np.isclose(engineering[2, 2], shear, rtol=1e-12)
    kelvin = stiffness_from_engineering(engineering)
    assert np.isclose(kelvin[2, 2], 2.0 * shear, rtol=1e-12)

    # And the converted stiffness must reproduce the same stress.
    generator = np.random.default_rng(3)
    strain_engineering = generator.normal(size=3) * 1.0e-3
    direct = stress_from_voigt(engineering @ strain_engineering)
    through = kelvin @ strain_from_engineering(strain_engineering)
    np.testing.assert_allclose(through, direct, rtol=1e-12)


def test_the_strain_operator_shear_row_is_divided_not_multiplied() -> None:
    """`B_shear / sqrt(2)`, because the engineering slot already holds `2 eps_xy`.

    Multiplying instead would leave every shear four times too large in a
    quadratic form while leaving the normal components right -- a failure that
    passes any test using only uniaxial states.
    """

    engineering = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    kelvin = strain_operator_from_engineering(engineering)
    np.testing.assert_allclose(kelvin[:2], engineering[:2], rtol=0.0)
    np.testing.assert_allclose(kelvin[2], engineering[2] / np.sqrt(2.0), rtol=1e-15)

    displacement = np.array([0.7, -0.3])
    np.testing.assert_allclose(
        kelvin @ displacement, strain_from_engineering(engineering @ displacement), rtol=1e-14
    )


def test_the_equivalent_plastic_strain_needs_the_out_of_plane_component() -> None:
    """A two-dimensional norm is not `p_eq`, however tidy the metric.

    Plastic incompressibility puts a real component out of plane, and dropping
    it understates the equivalent strain. The completion is checked against the
    plane-stress metric route, which is the form the earlier work used.
    """

    generator = np.random.default_rng(4)
    stress = generator.normal(size=3) * 100.0
    metric = PLANE_STRESS_VON_MISES_METRIC
    equivalent = float(np.sqrt(stress @ metric @ stress))
    direction = metric @ stress / equivalent
    increment = 3.5e-4  # the equivalent plastic increment this direction carries

    engineering_plastic = increment * direction
    kelvin_plastic = strain_from_engineering(engineering_plastic)

    # The completed tensor is traceless, which is what makes the norm meaningful.
    full = three_dimensional_from_plane_stress_plastic(kelvin_plastic)
    assert abs(float(full[0] + full[1] + full[2])) < 1e-18

    assert np.isclose(float(equivalent_plastic_strain(kelvin_plastic)), increment, rtol=1e-12)

    # And the in-plane norm alone is not it.
    assert not np.isclose(float(np.linalg.norm(kelvin_plastic)), increment, rtol=1e-3)


def test_kelvin_does_not_make_the_plastic_gauge_the_identity() -> None:
    """The expectation Kelvin does *not* satisfy, pinned so nobody assumes it.

    Kelvin removes the metric from contractions, which is what dissipation
    needs. It does not remove it from the equivalent plastic strain, because a
    plane-stress plastic triple with `eps_zz` fixed by incompressibility is not
    an orthonormal subspace of the 3D deviatoric space. Writing
    `np.linalg.norm(z)` for `p_eq` after migrating to Kelvin would be a silent
    error of exactly the kind the migration exists to prevent.
    """

    from fem_inhouse.core.kelvin import PLANE_STRESS_PLASTIC_GAUGE

    gauge = PLANE_STRESS_PLASTIC_GAUGE
    assert not np.allclose(gauge, np.eye(3))
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(gauge)), [2.0 / 3.0, 2.0 / 3.0, 2.0], rtol=1e-13
    )

    generator = np.random.default_rng(9)
    triples = generator.normal(size=(200, 3)) * 1.0e-3
    through_gauge = np.sqrt(np.einsum("pi,ij,pj->p", triples, gauge, triples))
    np.testing.assert_allclose(equivalent_plastic_strain(triples), through_gauge, rtol=1e-13)
