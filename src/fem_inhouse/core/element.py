"""Bilinear plane-stress element operations, fully and reduced integrated."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from fem_inhouse.core.mesh import StructuredMesh

GAUSS_COORDINATE = 1.0 / np.sqrt(3.0)
GAUSS_POINTS = np.array(
    [
        [-GAUSS_COORDINATE, -GAUSS_COORDINATE],
        [GAUSS_COORDINATE, -GAUSS_COORDINATE],
        [GAUSS_COORDINATE, GAUSS_COORDINATE],
        [-GAUSS_COORDINATE, GAUSS_COORDINATE],
    ]
)
GAUSS_WEIGHTS = np.ones(4)
GAUSS_POINT_COUNT = 4

#: Node-ordered sign pattern of the bilinear hourglass mode.
#:
#: With the element node numbering used by this mesh, (1, -1, 1, -1) is the
#: displacement pattern whose strain vanishes at the element centre. It is the
#: mode a single central integration point cannot see, and it applies to the x
#: and y displacements independently, giving the two hourglass modes.
HOURGLASS_PATTERN = np.array([1.0, -1.0, 1.0, -1.0])

#: Relative tolerance on the symmetry of a reference tangent.
#:
#: Loose enough for the round-off of a rotated cubic stiffness, tight enough
#: that a genuinely unsymmetric operator is refused rather than symmetrised.
REFERENCE_TANGENT_SYMMETRY_TOLERANCE = 1e-10

#: Relative tolerance on negative eigenvalues of the hourglass stiffness.
HOURGLASS_POSITIVITY_TOLERANCE = 1e-10

ElementFormulation = Literal["cps4", "cps4r"]


@dataclass(frozen=True, slots=True)
class QuadratureRule:
    """Integration points and weights on the parent square."""

    points: NDArray
    weights: NDArray

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("quadrature points must have shape (n, 2)")
        if self.weights.shape != (self.points.shape[0],):
            raise ValueError("one weight is required per quadrature point")
        total = float(self.weights.sum())
        if not np.isclose(total, 4.0):
            raise ValueError(
                f"quadrature weights must sum to the parent area 4, got {total}"
            )

    @property
    def point_count(self) -> int:
        return len(self.weights)


CPS4_QUADRATURE = QuadratureRule(GAUSS_POINTS, GAUSS_WEIGHTS)
#: One central point carrying the whole parent area.
CPS4R_QUADRATURE = QuadratureRule(np.zeros((1, 2)), np.array([4.0]))

QUADRATURE_RULES: dict[str, QuadratureRule] = {
    "cps4": CPS4_QUADRATURE,
    "cps4r": CPS4R_QUADRATURE,
}


def quadrature_for(formulation: str) -> QuadratureRule:
    try:
        return QUADRATURE_RULES[formulation]
    except KeyError:
        known = ", ".join(sorted(QUADRATURE_RULES))
        raise ValueError(
            f"unknown element_formulation {formulation!r}; available: {known}"
        ) from None


@dataclass(frozen=True, slots=True)
class ElementOperators:
    """Constant operators shared by every element of a regular mesh.

    `hourglass_stiffness` is `None` for the fully integrated CPS4 and an 8x8
    matrix for the reduced CPS4R. There is no third case: a reduced element is
    never exposed without stabilisation.
    """

    elastic_stiffness: NDArray
    strain_displacement: NDArray
    jacobian_determinants: NDArray
    formulation: ElementFormulation = "cps4"
    gauss_points: NDArray = field(default_factory=lambda: GAUSS_POINTS)
    gauss_weights: NDArray = field(default_factory=lambda: GAUSS_WEIGHTS)
    hourglass_stiffness: NDArray | None = None

    @property
    def gauss_point_count(self) -> int:
        return len(self.gauss_weights)


def shape_function_derivatives(xi: float, eta: float) -> NDArray:
    """Derivatives of the four shape functions in natural coordinates."""

    return 0.25 * np.array(
        [
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
        ]
    )


def strain_displacement_matrix(
    coordinates: NDArray,
    xi: float,
    eta: float,
) -> tuple[NDArray, float]:
    """Return engineering-shear B and the positive Jacobian determinant."""

    natural_derivatives = shape_function_derivatives(xi, eta)
    jacobian = natural_derivatives @ coordinates
    determinant = float(np.linalg.det(jacobian))
    if determinant <= 0:
        raise ValueError("CPS4 element has a non-positive Jacobian")
    physical_derivatives = np.linalg.solve(jacobian, natural_derivatives)
    matrix = np.zeros((3, 8))
    for node in range(4):
        matrix[0, 2 * node] = physical_derivatives[0, node]
        matrix[1, 2 * node + 1] = physical_derivatives[1, node]
        matrix[2, 2 * node] = physical_derivatives[1, node]
        matrix[2, 2 * node + 1] = physical_derivatives[0, node]
    return matrix, determinant


def plane_stress_elasticity(young_modulus_mpa: float, poisson_ratio: float) -> NDArray:
    """Engineering-shear isotropic plane-stress elasticity matrix."""

    if young_modulus_mpa <= 0:
        raise ValueError("young_modulus_mpa must be positive")
    if not -1 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must satisfy -1 < nu < 0.5")
    factor = young_modulus_mpa / (1 - poisson_ratio**2)
    return factor * np.array(
        [
            [1, poisson_ratio, 0],
            [poisson_ratio, 1, 0],
            [0, 0, (1 - poisson_ratio) / 2],
        ]
    )


def validate_reference_tangent(tangent: NDArray) -> NDArray:
    """Check a stabilisation tangent and return a symmetrised copy, in MPa.

    Symmetrising is only ever applied to round-off. A genuinely unsymmetric
    operator is refused: it would mean the caller passed an algorithmic tangent
    where an elastic reference belongs, and the resulting hourglass stiffness
    would not be a stabilisation at all.
    """

    matrix = np.asarray(tangent, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"reference tangent must have shape (3, 3), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("reference tangent must be finite")
    magnitude = float(np.abs(matrix).max())
    if magnitude == 0.0:
        raise ValueError("reference tangent must not be identically zero")
    asymmetry = float(np.abs(matrix - matrix.T).max()) / magnitude
    if asymmetry > REFERENCE_TANGENT_SYMMETRY_TOLERANCE:
        raise ValueError(
            f"reference tangent is not symmetric: relative asymmetry {asymmetry:.3e} "
            f"exceeds {REFERENCE_TANGENT_SYMMETRY_TOLERANCE:.1e}"
        )
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if eigenvalues.min() <= 0.0:
        raise ValueError(
            f"reference tangent must be positive definite; smallest eigenvalue "
            f"{eigenvalues.min():.3e}"
        )
    return symmetric


def hourglass_stiffness(
    coordinates: NDArray,
    reference_tangent: NDArray,
    *,
    hourglass_scale: float = 1.0,
) -> NDArray:
    """Stiffness that restores what one central integration point cannot see.

    Following Flanagan and Belytschko, the stabilisation is the difference
    between the fully integrated and the one-point stiffness of the SAME
    reference operator:

        K_hg = beta (K_ref^4pt - K_ref^1pt).

    Two properties come free from writing it this way, and neither needs a
    hand-built projector. Any displacement field the one-point rule integrates
    exactly -- rigid body motion and every affine field, whose strain is
    constant -- contributes identically to both terms and therefore nothing to
    the difference, so the stabilisation is orthogonal to them by construction.
    And at beta = 1 with a constant tangent the element stiffness collapses back
    to the fully integrated one exactly, not approximately.

    This is stiffness control, not viscous: a viscous term would make the answer
    depend on the increment size, which is unacceptable for a quasi-static
    solve whose increment count is itself under study.
    """

    if not 0.0 < hourglass_scale <= 1.0:
        raise ValueError(
            f"hourglass_scale must satisfy 0 < beta <= 1, got {hourglass_scale}"
        )
    tangent = validate_reference_tangent(reference_tangent)

    full = np.zeros((8, 8))
    for point, weight in zip(CPS4_QUADRATURE.points, CPS4_QUADRATURE.weights, strict=True):
        matrix, determinant = strain_displacement_matrix(coordinates, point[0], point[1])
        full += weight * determinant * (matrix.T @ tangent @ matrix)

    centre, centre_determinant = strain_displacement_matrix(coordinates, 0.0, 0.0)
    reduced = (
        float(CPS4R_QUADRATURE.weights[0])
        * centre_determinant
        * (centre.T @ tangent @ centre)
    )

    stabilisation = hourglass_scale * (full - reduced)
    stabilisation = 0.5 * (stabilisation + stabilisation.T)

    eigenvalues = np.linalg.eigvalsh(stabilisation)
    largest = float(eigenvalues.max())
    if largest <= 0.0:
        raise ValueError("hourglass stiffness has no positive eigenvalue")
    smallest = float(eigenvalues.min())
    if smallest < -HOURGLASS_POSITIVITY_TOLERANCE * largest:
        raise ValueError(
            f"hourglass stiffness is not positive semi-definite: smallest eigenvalue "
            f"{smallest:.3e} against largest {largest:.3e}"
        )
    return stabilisation


def precompute_element(
    mesh: StructuredMesh,
    elasticity: NDArray,
    *,
    formulation: ElementFormulation = "cps4",
    hourglass_scale: float = 1.0,
    reference_tangent: NDArray | None = None,
) -> ElementOperators:
    """Compute the constant element operators once for a regular mesh.

    `reference_tangent` is the operator the stabilisation is built from, and it
    is ignored by the fully integrated formulation. It defaults to `elasticity`,
    which is right for an isotropic J2 law and wrong for an oriented cubic
    crystal; a crystal backend must supply its own in-plane reference.
    """

    rule = quadrature_for(formulation)
    coordinates = mesh.coords[mesh.conn[0]]
    stiffness = np.zeros((8, 8))
    matrices = np.zeros((rule.point_count, 3, 8))
    determinants = np.zeros(rule.point_count)
    for index, (point, weight) in enumerate(zip(rule.points, rule.weights, strict=True)):
        matrix, determinant = strain_displacement_matrix(coordinates, point[0], point[1])
        stiffness += weight * (matrix.T @ elasticity @ matrix) * determinant
        matrices[index] = matrix
        determinants[index] = determinant

    stabilisation: NDArray | None = None
    if formulation == "cps4r":
        stabilisation = hourglass_stiffness(
            coordinates,
            elasticity if reference_tangent is None else reference_tangent,
            hourglass_scale=hourglass_scale,
        )
        # The elastic stiffness the solver starts from must already carry the
        # stabilisation, or the first elastic predictor would see a singular
        # element and the hourglass modes would be free.
        stiffness = stiffness + stabilisation
    elif reference_tangent is not None:
        raise ValueError(
            "reference_tangent is a stabilisation parameter and has no meaning for "
            "the fully integrated cps4 formulation"
        )

    return ElementOperators(
        stiffness,
        matrices,
        determinants,
        formulation=formulation,
        gauss_points=rule.points,
        gauss_weights=rule.weights,
        hourglass_stiffness=stabilisation,
    )
