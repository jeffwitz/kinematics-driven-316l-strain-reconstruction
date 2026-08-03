"""Reduced integration CPS4R and its stiffness hourglass control.

Section 15 of the specification. Everything here is analytical: no MFront, no
solver, so the element algebra is pinned independently of the rest.
"""

from __future__ import annotations

import numpy as np
import pytest

from fem_inhouse.core.element import (
    CPS4_QUADRATURE,
    CPS4R_QUADRATURE,
    HOURGLASS_PATTERN,
    QuadratureRule,
    hourglass_stiffness,
    plane_stress_elasticity,
    precompute_element,
    quadrature_for,
    strain_displacement_matrix,
    validate_reference_tangent,
)
from fem_inhouse.core.mesh import StructuredMesh

YOUNG_MPA = 205_000.0
POISSON = 0.3
C11, C12, C44 = 197_000.0, 125_000.0, 122_000.0


@pytest.fixture
def mesh() -> StructuredMesh:
    return StructuredMesh(0.002, 0.002, 0.001, 1.0)


@pytest.fixture
def isotropic() -> np.ndarray:
    return plane_stress_elasticity(YOUNG_MPA, POISSON)


def _coordinates(mesh: StructuredMesh) -> np.ndarray:
    return mesh.coords[mesh.conn[0]]


def _rank(matrix: np.ndarray) -> int:
    eigenvalues = np.linalg.eigvalsh(matrix)
    return int((eigenvalues > 1e-9 * eigenvalues.max()).sum())


def _one_point_material_stiffness(mesh: StructuredMesh, tangent: np.ndarray) -> np.ndarray:
    centre, determinant = strain_displacement_matrix(_coordinates(mesh), 0.0, 0.0)
    return 4.0 * determinant * (centre.T @ tangent @ centre)


def _nodal_field(mesh: StructuredMesh, x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    field = np.zeros(8)
    field[0::2] = x_values
    field[1::2] = y_values
    return field


def _rotated_cubic_plane_stress(rotation: np.ndarray) -> np.ndarray:
    """Cubic stiffness rotated to the global frame, then condensed.

    The reference a crystal backend has to supply: the isotropic plane-stress
    matrix would be simply wrong for an oriented single crystal.
    """

    root_two = np.sqrt(2.0)

    def kelvin_to_tensor(vector: np.ndarray) -> np.ndarray:
        shear = vector[3:] / root_two
        return np.array(
            [
                [vector[0], shear[0], shear[1]],
                [shear[0], vector[1], shear[2]],
                [shear[1], shear[2], vector[2]],
            ]
        )

    def tensor_to_kelvin(tensor: np.ndarray) -> np.ndarray:
        return np.array(
            [
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                root_two * tensor[0, 1],
                root_two * tensor[0, 2],
                root_two * tensor[1, 2],
            ]
        )

    crystal = np.zeros((6, 6))
    crystal[:3, :3] = C12
    np.fill_diagonal(crystal[:3, :3], C11)
    np.fill_diagonal(crystal[3:, 3:], 2.0 * C44)

    rotated = np.zeros((6, 6))
    for column in range(6):
        unit = np.zeros(6)
        unit[column] = 1.0
        strain_crystal = tensor_to_kelvin(rotation @ kelvin_to_tensor(unit) @ rotation.T)
        stress_crystal = crystal @ strain_crystal
        rotated[:, column] = tensor_to_kelvin(
            rotation.T @ kelvin_to_tensor(stress_crystal) @ rotation
        )

    in_plane = np.array([0, 1, 3])
    out_of_plane = np.array([2, 4, 5])
    caa = rotated[np.ix_(in_plane, in_plane)]
    cab = rotated[np.ix_(in_plane, out_of_plane)]
    cba = rotated[np.ix_(out_of_plane, in_plane)]
    cbb = rotated[np.ix_(out_of_plane, out_of_plane)]
    condensed_kelvin = caa - cab @ np.linalg.solve(cbb, cba)
    # Kelvin to engineering shear on the last row and column.
    scale = np.array([1.0, 1.0, 1.0 / root_two])
    return condensed_kelvin * scale[:, None] * scale[None, :]


def _rotation_about_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------
# 15.1 - the quadrature rules
# --------------------------------------------------------------------------


def test_the_two_rules_have_the_expected_points() -> None:
    assert CPS4_QUADRATURE.point_count == 4
    assert CPS4R_QUADRATURE.point_count == 1
    assert CPS4R_QUADRATURE.points.tolist() == [[0.0, 0.0]]
    assert CPS4R_QUADRATURE.weights.tolist() == [4.0]


def test_both_rules_integrate_the_parent_area() -> None:
    for rule in (CPS4_QUADRATURE, CPS4R_QUADRATURE):
        assert rule.weights.sum() == pytest.approx(4.0)


def test_a_rule_that_misses_the_parent_area_is_refused() -> None:
    """A wrong weight would rescale every element silently."""

    with pytest.raises(ValueError, match="sum to the parent area 4"):
        QuadratureRule(np.zeros((1, 2)), np.array([1.0]))


def test_an_unknown_formulation_names_the_available_ones() -> None:
    with pytest.raises(ValueError, match="cps4, cps4r"):
        quadrature_for("cps8")


# --------------------------------------------------------------------------
# 15.3 - the rank of the element stiffness
# --------------------------------------------------------------------------


def test_the_unstabilised_one_point_stiffness_has_rank_three(
    mesh: StructuredMesh, isotropic: np.ndarray
) -> None:
    """Three constant-strain modes seen, five modes invisible.

    Three of those five are rigid body motions and must stay free; the other
    two are the hourglass modes and must not.
    """

    material = _one_point_material_stiffness(mesh, isotropic)

    assert _rank(material) == 3


def test_the_stabilised_stiffness_has_rank_five(
    mesh: StructuredMesh, isotropic: np.ndarray
) -> None:
    operators = precompute_element(mesh, isotropic, formulation="cps4r")

    assert _rank(operators.elastic_stiffness) == 5


def test_only_the_rigid_body_modes_remain_in_the_kernel(
    mesh: StructuredMesh, isotropic: np.ndarray
) -> None:
    operators = precompute_element(mesh, isotropic, formulation="cps4r")
    coordinates = _coordinates(mesh)

    eigenvalues, vectors = np.linalg.eigh(operators.elastic_stiffness)
    kernel = vectors[:, eigenvalues <= 1e-9 * eigenvalues.max()]

    assert kernel.shape[1] == 3
    rigid = np.column_stack(
        [
            _nodal_field(mesh, np.ones(4), np.zeros(4)),
            _nodal_field(mesh, np.zeros(4), np.ones(4)),
            _nodal_field(mesh, -coordinates[:, 1], coordinates[:, 0]),
        ]
    )
    # The kernel is exactly the rigid-body space: every rigid mode is in it and
    # nothing else is, which is what makes the element solvable once supported.
    residual = rigid - kernel @ (kernel.T @ rigid)
    assert np.abs(residual).max() < 1e-9 * np.abs(rigid).max()


# --------------------------------------------------------------------------
# 15.4 - the affine patch test
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "gradient"),
    [
        ("uniform tension", np.array([[1e-3, 0.0], [0.0, 0.0]])),
        ("biaxial", np.array([[1e-3, 0.0], [0.0, -3e-4]])),
        ("uniform shear", np.array([[0.0, 2e-4], [2e-4, 0.0]])),
        ("rigid rotation", np.array([[0.0, -1e-4], [1e-4, 0.0]])),
    ],
)
def test_an_affine_field_produces_no_hourglass_response(
    mesh: StructuredMesh, isotropic: np.ndarray, label: str, gradient: np.ndarray
) -> None:
    """The property that makes the stabilisation admissible.

    A one-point rule integrates a constant-strain field exactly, so an affine
    field contributes identically to the four-point and the one-point stiffness
    and therefore nothing to their difference. Any hourglass force here would be
    an artificial stiffness added to a physical deformation mode.
    """

    operators = precompute_element(mesh, isotropic, formulation="cps4r")
    coordinates = _coordinates(mesh)
    displacement = _nodal_field(
        mesh, coordinates @ gradient[0], coordinates @ gradient[1]
    )

    force = operators.hourglass_stiffness @ displacement
    energy = 0.5 * displacement @ operators.hourglass_stiffness @ displacement

    reference = np.abs(operators.elastic_stiffness @ displacement).max()
    assert np.abs(force).max() < 1e-12 * max(reference, 1.0)
    assert abs(energy) < 1e-12


def test_a_rigid_translation_produces_no_hourglass_response(
    mesh: StructuredMesh, isotropic: np.ndarray
) -> None:
    operators = precompute_element(mesh, isotropic, formulation="cps4r")
    displacement = _nodal_field(mesh, np.full(4, 1e-4), np.full(4, -2e-4))

    assert np.abs(operators.hourglass_stiffness @ displacement).max() < 1e-9


# --------------------------------------------------------------------------
# 15.5 - the pure hourglass modes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1])
def test_the_hourglass_mode_is_invisible_to_the_material_point(
    mesh: StructuredMesh, isotropic: np.ndarray, axis: int
) -> None:
    """Zero strain at the centre, zero material energy, positive stabilisation.

    Without the stabilisation this mode would cost nothing and the element would
    deform freely into it.
    """

    operators = precompute_element(mesh, isotropic, formulation="cps4r")
    displacement = np.zeros(8)
    displacement[axis::2] = HOURGLASS_PATTERN

    centre_strain = operators.strain_displacement[0] @ displacement
    material = _one_point_material_stiffness(mesh, isotropic)

    assert np.abs(centre_strain).max() < 1e-15
    assert abs(0.5 * displacement @ material @ displacement) < 1e-9
    assert 0.5 * displacement @ operators.hourglass_stiffness @ displacement > 0.0
    assert np.abs(operators.hourglass_stiffness @ displacement).max() > 0.0


def test_the_two_hourglass_modes_cost_the_same_on_a_square(
    mesh: StructuredMesh, isotropic: np.ndarray
) -> None:
    operators = precompute_element(mesh, isotropic, formulation="cps4r")
    energies = []
    for axis in (0, 1):
        displacement = np.zeros(8)
        displacement[axis::2] = HOURGLASS_PATTERN
        energies.append(0.5 * displacement @ operators.hourglass_stiffness @ displacement)

    assert energies[0] == pytest.approx(energies[1], rel=1e-12)


# --------------------------------------------------------------------------
# 15.6 - elastic equivalence at beta = 1
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "tangent_factory"),
    [
        ("isotropic", lambda: plane_stress_elasticity(YOUNG_MPA, POISSON)),
        ("cubic, identity orientation", lambda: _rotated_cubic_plane_stress(np.eye(3))),
        ("cubic, rotated 30 degrees", lambda: _rotated_cubic_plane_stress(_rotation_about_z(30.0))),
    ],
)
def test_the_reduced_element_reproduces_the_full_one_at_unit_scale(
    mesh: StructuredMesh, label: str, tangent_factory
) -> None:
    """Exact, not approximate, and for anisotropic references too.

    K_cps4r = K^1pt + 1 * (K^4pt - K^1pt) = K^4pt identically, so this is an
    algebraic identity rather than a numerical coincidence. It is worth
    asserting anyway: it fails the moment the stabilisation is built from a
    different operator than the material one.
    """

    tangent = tangent_factory()
    full = precompute_element(mesh, tangent)
    reduced = precompute_element(mesh, tangent, formulation="cps4r", hourglass_scale=1.0)

    deviation = np.abs(full.elastic_stiffness - reduced.elastic_stiffness).max()
    assert deviation / np.abs(full.elastic_stiffness).max() < 1e-13


def test_a_smaller_scale_softens_the_hourglass_modes_only(
    mesh: StructuredMesh, isotropic: np.ndarray
) -> None:
    full = precompute_element(mesh, isotropic)
    soft = precompute_element(mesh, isotropic, formulation="cps4r", hourglass_scale=0.25)

    hourglass = np.zeros(8)
    hourglass[0::2] = HOURGLASS_PATTERN
    stiff = precompute_element(mesh, isotropic, formulation="cps4r", hourglass_scale=1.0)

    assert 0.5 * hourglass @ soft.hourglass_stiffness @ hourglass == pytest.approx(
        0.25 * (0.5 * hourglass @ stiff.hourglass_stiffness @ hourglass), rel=1e-12
    )
    # The material part is untouched: only the invisible modes are scaled.
    coordinates = _coordinates(mesh)
    affine = _nodal_field(mesh, coordinates[:, 0] * 1e-3, np.zeros(4))
    assert soft.elastic_stiffness @ affine == pytest.approx(
        full.elastic_stiffness @ affine, abs=1e-9
    )


# --------------------------------------------------------------------------
# Positivity and validation
# --------------------------------------------------------------------------


def test_the_hourglass_stiffness_is_symmetric_positive_semi_definite(
    mesh: StructuredMesh, isotropic: np.ndarray
) -> None:
    stabilisation = precompute_element(
        mesh, isotropic, formulation="cps4r"
    ).hourglass_stiffness

    assert np.abs(stabilisation - stabilisation.T).max() == 0.0
    eigenvalues = np.linalg.eigvalsh(stabilisation)
    assert eigenvalues.min() > -1e-10 * eigenvalues.max()
    # Exactly the two hourglass modes are stiffened; everything else is free.
    assert _rank(stabilisation) == 2


@pytest.mark.parametrize("scale", [0.0, -0.5, 1.5])
def test_an_out_of_range_scale_is_refused(
    mesh: StructuredMesh, isotropic: np.ndarray, scale: float
) -> None:
    with pytest.raises(ValueError, match="0 < beta <= 1"):
        precompute_element(mesh, isotropic, formulation="cps4r", hourglass_scale=scale)


def test_an_unsymmetric_reference_tangent_is_refused() -> None:
    """A crystal algorithmic tangent is unsymmetric; an elastic reference is not.

    Accepting one would build a stabilisation that is not a stabilisation.
    """

    tangent = plane_stress_elasticity(YOUNG_MPA, POISSON)
    tangent[0, 2] += 0.1 * tangent[0, 0]

    with pytest.raises(ValueError, match="not symmetric"):
        validate_reference_tangent(tangent)


def test_a_non_positive_definite_reference_tangent_is_refused() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        validate_reference_tangent(np.diag([1.0, 1.0, -1.0]))


def test_a_misshapen_reference_tangent_is_refused() -> None:
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        validate_reference_tangent(np.eye(6))


def test_round_off_asymmetry_is_absorbed_rather_than_refused() -> None:
    """A rotated cubic stiffness is symmetric only to round-off."""

    tangent = _rotated_cubic_plane_stress(_rotation_about_z(30.0))
    tangent[0, 1] += 1e-13 * tangent[0, 0]

    symmetric = validate_reference_tangent(tangent)

    assert np.abs(symmetric - symmetric.T).max() == 0.0


def test_the_full_formulation_refuses_a_stabilisation_parameter(
    mesh: StructuredMesh, isotropic: np.ndarray
) -> None:
    """Silently ignoring it would let a user believe CPS4 was stabilised."""

    with pytest.raises(ValueError, match="no meaning for"):
        precompute_element(mesh, isotropic, reference_tangent=isotropic)


# --------------------------------------------------------------------------
# The anisotropic reference actually matters
# --------------------------------------------------------------------------


def test_an_anisotropic_reference_gives_a_different_stabilisation(
    mesh: StructuredMesh, isotropic: np.ndarray
) -> None:
    """Justifies section 11: the isotropic matrix is not a safe default.

    Building the stabilisation from plane_stress_elasticity for an oriented
    cubic crystal would stiffen the hourglass modes by the wrong amount, in a
    way no global energy ratio would obviously reveal.
    """

    coordinates = _coordinates(mesh)
    from_isotropic = hourglass_stiffness(coordinates, isotropic)
    from_cubic = hourglass_stiffness(
        coordinates, _rotated_cubic_plane_stress(_rotation_about_z(30.0))
    )

    relative = np.abs(from_cubic - from_isotropic).max() / np.abs(from_isotropic).max()
    assert relative > 0.1


def test_the_operators_report_their_formulation(
    mesh: StructuredMesh, isotropic: np.ndarray
) -> None:
    full = precompute_element(mesh, isotropic)
    reduced = precompute_element(mesh, isotropic, formulation="cps4r")

    assert full.formulation == "cps4"
    assert full.gauss_point_count == 4
    assert full.hourglass_stiffness is None
    assert reduced.formulation == "cps4r"
    assert reduced.gauss_point_count == 1
    assert reduced.strain_displacement.shape == (1, 3, 8)
    assert reduced.jacobian_determinants.shape == (1,)
