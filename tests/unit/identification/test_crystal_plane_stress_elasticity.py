from __future__ import annotations

import numpy as np

from fem_inhouse.core.crystal_orientation import rotation_from_euler_bunge_deg
from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.identification.crystal_plane_stress_elasticity import (
    cubic_stiffness_from_engineering_constants,
    rotated_plane_stress_stiffness,
    tensor_to_voigt,
    voigt_to_tensor,
)

#: Crystal-frame constants declared by `mfront/Fcc316LForestRubinSrix.mfront`.
CRYSTAL = (99950.31055900622, 0.3881987577639752, 122000.0)


def test_the_declared_constants_give_the_expected_cubic_stiffness() -> None:
    """The elastic reference must not drift away from the constitutive law.

    These are read from the same three numbers the FCC behaviour declares, so a
    change to the law shows up here rather than silently producing a second,
    inconsistent elasticity for the identification side.
    """

    stiffness = cubic_stiffness_from_engineering_constants(*CRYSTAL)
    assert np.isclose(stiffness[0, 0], 197_000.0, rtol=1e-6)
    assert np.isclose(stiffness[0, 1], 125_000.0, rtol=1e-6)
    assert np.isclose(stiffness[3, 3], 122_000.0, rtol=1e-12)
    zener = 2 * stiffness[3, 3] / (stiffness[0, 0] - stiffness[0, 1])
    assert 3.3 < zener < 3.5


def test_the_voigt_and_tensor_forms_round_trip() -> None:
    stiffness = cubic_stiffness_from_engineering_constants(*CRYSTAL)
    np.testing.assert_allclose(tensor_to_voigt(voigt_to_tensor(stiffness)), stiffness, atol=0.0)


def test_a_cubic_symmetry_leaves_the_stiffness_unchanged() -> None:
    """Rotating a cubic crystal by one of its own symmetries must change nothing.

    This is the cheapest check that the four-index rotation and the Voigt
    conventions agree: a wrong factor on the shear rows survives most tests but
    not this one.
    """

    stiffness = cubic_stiffness_from_engineering_constants(*CRYSTAL)
    tensor = voigt_to_tensor(stiffness)
    rotation = rotation_from_euler_bunge_deg(90.0, 0.0, 0.0)
    rotated = np.einsum(
        "pi,qj,rk,sl,pqrs->ijkl", rotation, rotation, rotation, rotation, tensor
    )
    np.testing.assert_allclose(
        tensor_to_voigt(rotated), stiffness, rtol=0.0, atol=1e-6 * abs(stiffness).max()
    )


def test_an_isotropic_crystal_condenses_to_the_plane_stress_matrix_at_any_orientation() -> None:
    """The whole rotate-then-condense chain, checked against the existing path.

    An isotropic material has no orientation, so whatever Euler angles are fed
    in, the condensation must return exactly `plane_stress_elasticity`. That
    exercises the 3D rotation, the Voigt mapping and the Schur complement
    together, against a function the rest of the repository already trusts.
    """

    young, poisson = 205_000.0, 0.30
    isotropic = cubic_stiffness_from_engineering_constants(
        young, poisson, young / (2 * (1 + poisson))
    )
    angles = np.array([[0.0, 0.0, 0.0], [37.0, 21.0, 63.0], [180.0, 90.0, 45.0]])
    condensed = rotated_plane_stress_stiffness(isotropic, angles)
    expected = plane_stress_elasticity(young, poisson)
    for index in range(angles.shape[0]):
        np.testing.assert_allclose(condensed[index], expected, rtol=1e-12, atol=1e-6)


def test_the_condensation_is_not_a_plain_in_plane_rotation() -> None:
    """An out-of-plane orientation must couple normal and shear in the plane.

    Rotating a 3x3 in-plane matrix instead of condensing the rotated 3D tensor
    is the natural shortcut and it is wrong: it cannot produce the normal/shear
    coupling that an out-of-plane crystal axis creates. The off-diagonal entries
    below are what distinguishes the two.
    """

    stiffness = cubic_stiffness_from_engineering_constants(*CRYSTAL)
    aligned = rotated_plane_stress_stiffness(stiffness, np.array([[0.0, 0.0, 0.0]]))[0]
    tilted = rotated_plane_stress_stiffness(stiffness, np.array([[45.0, 54.7356, 0.0]]))[0]

    assert abs(aligned[0, 2]) < 1e-8 * abs(aligned).max()
    assert abs(tilted[0, 2]) > 0.05 * abs(tilted).max()
    # And the in-plane stiffness genuinely changes with orientation.
    assert tilted[0, 0] > 2.0 * aligned[0, 0] / 1.5
