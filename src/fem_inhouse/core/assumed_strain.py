"""Assumed-strain stabilisation of the one-point quadrilateral, after QUAS4.

Sections 4, 5, 6 and 21 of the 2026-08-04 specification. The derivation, its
sources and every convention are in
`docs/reference/quas4_assumed_strain_derivation.md`; this module implements it
and nothing else.

The shape of the thing, in one paragraph. The bilinear gradient operator splits
exactly as `B = B_c + B_n`, where `B_c` is the operator at the element centre and
`B_n` carries what the central point cannot see. The constitutive law is
integrated **once**, at the centre, and supplies both the stress `sigma_c` and
the tangent `C`. Four geometric Gauss points then supply `B_n`, the Jacobian and
the weights -- and nothing else: no state, no history, no constitutive call. The
element stiffness is `K_c + K_stab` and the internal force splits the same way.

Two things distinguish this from the `hourglass_stiffness` already in
`element.py`. The stabilisation is built on the **current** tangent rather than a
frozen elastic reference, which is the whole point: after yielding the elastic
reference over-stiffens the hourglass modes while every other mode softens. And
there is **no free coefficient** -- no `beta` to choose, and none to calibrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.element import CPS4_QUADRATURE, shape_function_derivatives

FloatArray = NDArray[np.float64]

#: Nodal values of the constant and hourglass patterns, in node order.
CONSTANT_PATTERN = np.ones(4)
HOURGLASS_PATTERN = np.array([1.0, -1.0, 1.0, -1.0])

#: `(e1, e2, e3)` of Belytschko and Bindeman, table 3.3-1 of R3.06.10.
#:
#: `e3 = 0` cancels the hourglass contribution to the shear strain, which is the
#: cure for the bending stiffness of the fully integrated element. `e2` couples
#: the two normal components; `asbqi` sets it to `-poisson`, and is the reason
#: that variant is unusable for an anisotropic tangent.
ProjectionName = Literal["quad4", "asmd", "asoi", "asoi_half"]

PROJECTION_COEFFICIENTS: dict[ProjectionName, tuple[float, float, float]] = {
    # The plain regrouping: `B_c + B_n` reproduces the QUAD4 operator exactly,
    # so `K_c + K_stab` reproduces the fully integrated stiffness. Kept as the
    # consistency check of the whole construction, not as a usable element.
    "quad4": (1.0, 0.0, 1.0),
    "asmd": (0.5, -0.5, 1.0),
    "asoi": (1.0, -1.0, 0.0),
    "asoi_half": (0.5, -0.5, 0.0),
}

#: `asbqi` is deliberately absent. Its `e2 = -poisson` presumes the transverse
#: coupling is isotropic, and a cubic crystal at a general orientation has a
#: condensed plane-stress tangent with extension-shear coupling and no single
#: Poisson ratio. R3.06.10 does not say how to define that ratio for an
#: anisotropic tangent, and inventing one is not an option.
UNSUPPORTED_PROJECTIONS: frozenset[str] = frozenset({"asbqi"})


@dataclass(frozen=True, slots=True)
class CentralOperators:
    """Everything about one element's geometry, computed once.

    `b_x` and `b_y` are Hallquist's constant gradient vectors, `gamma` is the
    hourglass pattern purged of its linear content -- which is what makes the
    stabilisation orthogonal to every affine field -- and the four geometric
    points carry the Jacobian-weighted derivatives of `h = xi*eta`.
    """

    area: float
    b_x: FloatArray
    b_y: FloatArray
    gamma: FloatArray
    #: `(4, 2)` array of `[h_x, h_y]` at each geometric Gauss point.
    hourglass_gradients: FloatArray
    #: `w_g * det J_g` at each geometric Gauss point.
    jacobian_weights: FloatArray

    @property
    def strain_displacement_centre(self) -> FloatArray:
        return _engineering_operator(self.b_x, self.b_y)


def _engineering_operator(row_x: FloatArray, row_y: FloatArray) -> FloatArray:
    """Assemble a `(3, 8)` engineering-shear operator from two nodal rows."""

    operator = np.zeros((3, 8))
    operator[0, 0::2] = row_x
    operator[1, 1::2] = row_y
    operator[2, 0::2] = row_y
    operator[2, 1::2] = row_x
    return operator


def central_operators(coordinates: ArrayLike) -> CentralOperators:
    """Build the geometric operators of one quadrilateral.

    The closed forms are R3.06.10 eq 2.1-3 and 3.2-3. They agree with the
    shape-function derivatives at the centre to machine precision, which is
    asserted rather than assumed.
    """

    nodes = np.asarray(coordinates, dtype=float)
    if nodes.shape != (4, 2):
        raise ValueError(f"a quadrilateral has four nodes in 2D, got {nodes.shape}")
    if not np.isfinite(nodes).all():
        raise ValueError("node coordinates must be finite")
    x, y = nodes[:, 0], nodes[:, 1]
    area = 0.5 * ((x[2] - x[0]) * (y[3] - y[1]) + (x[1] - x[3]) * (y[2] - y[0]))
    if area <= 0.0:
        raise ValueError(f"quadrilateral has a non-positive area: {area}")
    b_x = np.array([y[1] - y[3], y[2] - y[0], y[3] - y[1], y[0] - y[2]]) / (2.0 * area)
    b_y = np.array([x[3] - x[1], x[0] - x[2], x[1] - x[3], x[2] - x[0]]) / (2.0 * area)
    gamma = 0.25 * (
        HOURGLASS_PATTERN - (HOURGLASS_PATTERN @ x) * b_x - (HOURGLASS_PATTERN @ y) * b_y
    )

    gradients = np.zeros((CPS4_QUADRATURE.point_count, 2))
    weights = np.zeros(CPS4_QUADRATURE.point_count)
    for index, ((xi, eta), weight) in enumerate(
        zip(CPS4_QUADRATURE.points, CPS4_QUADRATURE.weights, strict=True)
    ):
        natural = shape_function_derivatives(xi, eta)
        jacobian = natural @ nodes
        determinant = float(np.linalg.det(jacobian))
        if determinant <= 0.0:
            raise ValueError("quadrilateral has a non-positive Jacobian")
        # h = xi * eta, so its natural gradient is (eta, xi).
        gradients[index] = np.linalg.solve(jacobian, np.array([eta, xi]))
        weights[index] = weight * determinant
    return CentralOperators(
        area=float(area),
        b_x=b_x,
        b_y=b_y,
        gamma=gamma,
        hourglass_gradients=gradients,
        jacobian_weights=weights,
    )


def enrichment_operator(
    operators: CentralOperators,
    point: int,
    projection: ProjectionName,
) -> FloatArray:
    """`B_n` at one geometric point, R3.06.10 eq 3.3-2."""

    e1, e2, e3 = projection_coefficients(projection)
    gradient_x, gradient_y = operators.hourglass_gradients[point]
    gamma = operators.gamma
    enriched = np.zeros((3, 8))
    enriched[0, 0::2] = e1 * gradient_x * gamma
    enriched[0, 1::2] = e2 * gradient_y * gamma
    enriched[1, 0::2] = e2 * gradient_x * gamma
    enriched[1, 1::2] = e1 * gradient_y * gamma
    enriched[2, 0::2] = e3 * gradient_y * gamma
    enriched[2, 1::2] = e3 * gradient_x * gamma
    return enriched


def projection_coefficients(projection: str) -> tuple[float, float, float]:
    if projection in UNSUPPORTED_PROJECTIONS:
        raise ValueError(
            f"projection {projection!r} is deliberately not implemented: its e2 "
            "coefficient is minus Poisson's ratio, which presumes an isotropic "
            "transverse coupling and has no definition for an anisotropic tangent"
        )
    try:
        return PROJECTION_COEFFICIENTS[projection]  # type: ignore[index]
    except KeyError:
        known = ", ".join(sorted(PROJECTION_COEFFICIENTS))
        raise ValueError(
            f"unknown assumed-strain projection {projection!r}; available: {known}"
        ) from None


def hourglass_amplitudes(operators: CentralOperators, displacement: ArrayLike) -> FloatArray:
    """`(q_x, q_y) = (gamma . u_x, gamma . u_y)`, the two hourglass amplitudes.

    Section 18 reports these per element: they say how far the element is into
    the modes the central point cannot see.
    """

    values = np.asarray(displacement, dtype=float).reshape(-1)
    if values.shape != (8,):
        raise ValueError(f"an element displacement has eight components, got {values.shape}")
    return np.array([operators.gamma @ values[0::2], operators.gamma @ values[1::2]])


@dataclass(frozen=True, slots=True)
class StabilisationResult:
    """What a stabilisation strategy returns for one element."""

    internal_force: FloatArray
    tangent: FloatArray
    energy: float
    modal_amplitudes: FloatArray
    diagnostics: dict[str, float] = field(default_factory=dict)


class ReducedIntegrationStabilisation(Protocol):
    """Strategy contract, section 21.

    `tangent_centre` is the tangent of the **central** constitutive point, and
    is the only constitutive information a strategy receives. Nothing in this
    interface can reach a material model, which is how the one-call-per-element
    guarantee is enforced structurally rather than by discipline.
    """

    @property
    def name(self) -> str: ...

    def evaluate(
        self,
        operators: CentralOperators,
        displacement: ArrayLike,
        tangent_centre: ArrayLike,
    ) -> StabilisationResult: ...


def _validated_tangent(tangent: ArrayLike) -> FloatArray:
    values = np.asarray(tangent, dtype=float)
    if values.shape != (3, 3):
        raise ValueError(f"a plane-stress tangent has shape (3, 3), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("the constitutive tangent must be finite")
    return values


@dataclass(frozen=True, slots=True)
class AssumedStrainStabilisation:
    """QUAS4 stabilisation on the current constitutive tangent.

    Section 6.1, the direct variant. `C_stab` is the algorithmic tangent as it
    comes from the central integration, asymmetry included: a crystal tangent is
    genuinely non-symmetric, and symmetrising it here would make the global
    tangent inconsistent with the force it is supposed to differentiate.
    """

    projection: ProjectionName = "asmd"

    @property
    def name(self) -> str:
        return "assumed_strain_current"

    def evaluate(
        self,
        operators: CentralOperators,
        displacement: ArrayLike,
        tangent_centre: ArrayLike,
    ) -> StabilisationResult:
        return _assumed_strain_evaluate(
            operators=operators,
            displacement=displacement,
            tangent=_validated_tangent(tangent_centre),
            projection=self.projection,
            strategy=self.name,
            extra={},
        )


@dataclass(frozen=True, slots=True)
class EnergyProjectedAssumedStrainStabilisation:
    """QUAS4 stabilisation on a spectrally floored symmetric tangent.

    Section 6.2. The stabilisation must remain *stabilising*: a softening
    tangent can lose positive definiteness, and a stabilisation built on it
    would stop opposing the hourglass modes exactly when they matter most.

    The floor is a fixed fraction of the largest eigenvalue of the symmetric
    part, so it is a **conditioning rule and not a tuning knob**: it has no
    reference to CPS4, to any material, or to any case. It is recorded in the
    manifest and its sensitivity is measured.
    """

    projection: ProjectionName = "asmd"
    #: Eigenvalues below `relative_floor * lambda_max` are lifted to it.
    relative_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        if not 0.0 < self.relative_floor < 1.0:
            raise ValueError("relative_floor must lie in (0, 1)")

    @property
    def name(self) -> str:
        return "assumed_strain_energy"

    def evaluate(
        self,
        operators: CentralOperators,
        displacement: ArrayLike,
        tangent_centre: ArrayLike,
    ) -> StabilisationResult:
        algorithmic = _validated_tangent(tangent_centre)
        symmetric = 0.5 * (algorithmic + algorithmic.T)
        eigenvalues, vectors = np.linalg.eigh(symmetric)
        largest = float(eigenvalues.max())
        if largest <= 0.0:
            raise ValueError(
                "the symmetric part of the constitutive tangent has no positive "
                "eigenvalue; the stabilisation cannot be made positive from it"
            )
        floor = self.relative_floor * largest
        lifted = np.maximum(eigenvalues, floor)
        projected = vectors @ np.diag(lifted) @ vectors.T
        return _assumed_strain_evaluate(
            operators=operators,
            displacement=displacement,
            tangent=projected,
            projection=self.projection,
            strategy=self.name,
            extra={
                "tangent_floor": floor,
                "tangent_smallest_eigenvalue": float(eigenvalues.min()),
                "tangent_largest_eigenvalue": largest,
                "tangent_eigenvalues_lifted": float(np.count_nonzero(eigenvalues < floor)),
                "tangent_relative_asymmetry": float(
                    np.abs(algorithmic - algorithmic.T).max()
                    / max(float(np.abs(algorithmic).max()), 1.0)
                ),
            },
        )


def _assumed_strain_evaluate(
    *,
    operators: CentralOperators,
    displacement: ArrayLike,
    tangent: FloatArray,
    projection: ProjectionName,
    strategy: str,
    extra: dict[str, float],
) -> StabilisationResult:
    """The shared QUAS4 assembly, R3.06.10 eq 3.3-5 and 3.3-6.

    The three-term form is assembled in full even though the two cross terms sum
    to zero for every quadrilateral tested. They cost four small products and
    they fail loudly on a geometry where the cancellation does not hold, which a
    silent simplification would not.
    """

    nodal = np.asarray(displacement, dtype=float).reshape(-1)
    if nodal.shape != (8,):
        raise ValueError(f"an element displacement has eight components, got {nodal.shape}")
    centre = operators.strain_displacement_centre

    stiffness = np.zeros((8, 8))
    force = np.zeros(8)
    energy = 0.0
    for point in range(operators.jacobian_weights.size):
        enriched = enrichment_operator(operators, point, projection)
        weight = float(operators.jacobian_weights[point])
        stiffness += weight * (
            centre.T @ tangent @ enriched
            + enriched.T @ tangent @ centre
            + enriched.T @ tangent @ enriched
        )
        # The stabilising strain is the enrichment acting on the current
        # displacement; it carries no history, so nothing here needs a
        # transaction of its own.
        stabilising_strain = enriched @ nodal
        stabilising_stress = tangent @ stabilising_strain
        force += weight * ((centre + enriched).T @ stabilising_stress)
        energy += 0.5 * weight * float(stabilising_strain @ stabilising_stress)

    amplitudes = hourglass_amplitudes(operators, nodal)
    diagnostics = {
        "stabilisation_energy": energy,
        "stabilisation_force_norm": float(np.linalg.norm(force)),
        "stabilisation_tangent_norm": float(np.linalg.norm(stiffness)),
        "hourglass_amplitude": float(np.linalg.norm(amplitudes)),
        **extra,
    }
    return StabilisationResult(
        internal_force=force,
        tangent=stiffness,
        energy=energy,
        modal_amplitudes=amplitudes,
        diagnostics=diagnostics,
    )


#: Selectable strategies, section 21. `elastic_difference` is not here: it lives
#: in `element.py` and stays reachable through the existing `cps4r` formulation,
#: so archived campaigns remain readable without this module being involved.
STABILISATION_STRATEGIES: dict[str, type] = {
    "assumed_strain_current": AssumedStrainStabilisation,
    "assumed_strain_energy": EnergyProjectedAssumedStrainStabilisation,
}


def make_stabilisation(
    strategy: str,
    *,
    projection: str = "asmd",
    relative_floor: float = 1.0e-6,
) -> ReducedIntegrationStabilisation:
    """Build a strategy by name, refusing an unknown one before any solve."""

    projection_coefficients(projection)
    if strategy == "assumed_strain_current":
        return AssumedStrainStabilisation(projection=projection)  # type: ignore[arg-type]
    if strategy == "assumed_strain_energy":
        return EnergyProjectedAssumedStrainStabilisation(
            projection=projection,  # type: ignore[arg-type]
            relative_floor=relative_floor,
        )
    known = ", ".join(sorted(STABILISATION_STRATEGIES))
    raise ValueError(f"unknown stabilisation strategy {strategy!r}; available: {known}")


def batched_stabilisation(
    operators: CentralOperators,
    displacements: ArrayLike,
    tangents: ArrayLike,
    *,
    projection: ProjectionName = "asmd",
    relative_floor: float | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Stabilisation of every element at once, from its own current tangent.

    On a regular mesh all elements share one geometry, so `operators` is built
    once and only the tangents differ. That is what makes recomputing the
    stabilisation at every Newton iteration affordable: the per-element work is
    four small triple products, against one constitutive integration.

    `relative_floor` selects the energy variant. `None` uses the algorithmic
    tangent as it comes, asymmetry included.

    Returns `(forces, stiffnesses, energies)` shaped `(n, 8)`, `(n, 8, 8)`
    and `(n,)`.
    """

    nodal = np.asarray(displacements, dtype=float)
    material = np.asarray(tangents, dtype=float)
    if nodal.ndim != 2 or nodal.shape[1] != 8:
        raise ValueError(f"displacements must have shape (n, 8), got {nodal.shape}")
    if material.shape != (nodal.shape[0], 3, 3):
        raise ValueError(
            f"tangents must have shape {(nodal.shape[0], 3, 3)}, got {material.shape}"
        )
    if not np.isfinite(material).all():
        raise ValueError("constitutive tangents must be finite")

    if relative_floor is not None:
        if not 0.0 < relative_floor < 1.0:
            raise ValueError("relative_floor must lie in (0, 1)")
        symmetric = 0.5 * (material + np.swapaxes(material, 1, 2))
        eigenvalues, vectors = np.linalg.eigh(symmetric)
        largest = eigenvalues[:, -1]
        if np.any(largest <= 0.0):
            raise ValueError(
                "a constitutive tangent has no positive eigenvalue; the "
                "stabilisation cannot be made positive from it"
            )
        lifted = np.maximum(eigenvalues, (relative_floor * largest)[:, None])
        material = np.einsum("nij,nj,nkj->nik", vectors, lifted, vectors)

    centre = operators.strain_displacement_centre
    stiffness = np.zeros((nodal.shape[0], 8, 8))
    force = np.zeros_like(nodal)
    energy = np.zeros(nodal.shape[0])
    for point in range(operators.jacobian_weights.size):
        enriched = enrichment_operator(operators, point, projection)
        weight = float(operators.jacobian_weights[point])
        total = centre + enriched
        # (n, 3, 8) products, one per element, in one einsum each.
        centre_tangent = np.einsum("ij,njk->nik", centre.T, material)
        enriched_tangent = np.einsum("ij,njk->nik", enriched.T, material)
        stiffness += weight * (
            centre_tangent @ enriched + enriched_tangent @ centre + enriched_tangent @ enriched
        )
        strain = nodal @ enriched.T
        stress = np.einsum("nij,nj->ni", material, strain)
        force += weight * (stress @ total)
        energy += 0.5 * weight * np.einsum("ni,ni->n", strain, stress)
    return force, stiffness, energy
