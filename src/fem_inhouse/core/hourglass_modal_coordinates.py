"""Reduced coordinates in which the hourglass Jacobian defect is learned.

Sections 6, 7 and 8 of the 2026-08-04 Broyden specification.

The measured defect is precise: the physical element tangent is consistent to
`1.9e-6`, and the stabilisation tangent is wrong by `370 %` because it
differentiates `f_stab(u, C(u))` holding `C` fixed. A quasi-Newton correction
should therefore be learned **only** on the stabilisation, and in the smallest
space that carries it.

That space has five dimensions, not eight:

- the three central strains, which are what makes the constitutive tangent move;
- the two hourglass amplitudes, which are what produces the stabilising force.

The three rigid-body modes fall out automatically, which is why a correction
assembled as `H^T dG T` cannot put force on them -- a property this module makes
structural rather than checked after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.assumed_strain import CentralOperators

FloatArray = NDArray[np.float64]

#: Relative defect above which the stabilising force is not in the span of the
#: two hourglass modes, and the correction must not be applied to that element.
MODAL_PROJECTION_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class ModalCoordinates:
    """`T`, `H`, `L` and `T^+` for one element geometry.

    All four are pure geometry: they do not depend on the material, the state or
    the load, so on a regular mesh they are built once for the whole mesh.
    """

    #: `(2, 8)` -- nodal displacement to the two hourglass amplitudes.
    amplitude_operator: FloatArray
    #: `(5, 8)` -- nodal displacement to the reduced state `[eps_c; q]`.
    reduction: FloatArray
    #: `(8, 5)` -- right pseudo-inverse of `reduction`.
    reduction_pseudo_inverse: FloatArray
    #: `(2, 8)` -- nodal force to the generalised hourglass force.
    force_projector: FloatArray

    def reduced_state(self, displacement: ArrayLike) -> FloatArray:
        """`xi = T u`, the five coordinates the correction is learned in."""

        values = np.asarray(displacement, dtype=float)
        return values @ self.reduction.T

    def modal_force(self, stabilisation_force: ArrayLike) -> FloatArray:
        """`r = L f_stab`, the stabilising force in its two modes."""

        values = np.asarray(stabilisation_force, dtype=float)
        return values @ self.force_projector.T

    def modal_projection_defect(self, stabilisation_force: ArrayLike) -> FloatArray:
        """`|f - H^T L f| / |f|`, per element.

        The stabilising force *should* live entirely in the span of the two
        hourglass modes; section 7 requires this to be verified rather than
        assumed, and an element that fails it is excluded from the correction
        instead of being corrected on a projection that loses part of the force.
        """

        values = np.atleast_2d(np.asarray(stabilisation_force, dtype=float))
        rebuilt = self.modal_force(values) @ self.amplitude_operator
        residual = np.linalg.norm(values - rebuilt, axis=1)
        return residual / (np.linalg.norm(values, axis=1) + 1e-300)

    def reduced_jacobian(self, stabilisation_tangent: ArrayLike) -> FloatArray:
        """`G_0 = L K_stab T^+`, shaped `(..., 2, 5)`.

        The derivative of the generalised hourglass force with respect to the
        five reduced coordinates, as currently known -- that is, missing the term
        the Broyden correction exists to learn.
        """

        tangent = np.asarray(stabilisation_tangent, dtype=float)
        if tangent.shape[-2:] != (8, 8):
            raise ValueError(
                f"a stabilisation tangent has trailing shape (8, 8), got {tangent.shape}"
            )
        return (
            self.force_projector @ tangent @ self.reduction_pseudo_inverse
            if tangent.ndim == 2
            else np.einsum(
                "ij,njk,kl->nil",
                self.force_projector,
                tangent,
                self.reduction_pseudo_inverse,
            )
        )

    def expand_correction(self, reduced_correction: ArrayLike) -> FloatArray:
        """`H^T dG T`, shaped `(..., 8, 8)`.

        Three properties come from this form alone and need no enforcement: the
        correction produces only hourglass force, the rigid modes stay in its
        kernel, and an affine field is untouched -- because `gamma` is orthogonal
        to every affine field, so `H u_affine = 0`.
        """

        correction = np.asarray(reduced_correction, dtype=float)
        if correction.shape[-2:] != (2, 5):
            raise ValueError(
                f"a reduced correction has trailing shape (2, 5), got {correction.shape}"
            )
        if correction.ndim == 2:
            return self.amplitude_operator.T @ correction @ self.reduction
        return np.einsum(
            "ji,njk,kl->nil",
            self.amplitude_operator,
            correction,
            self.reduction,
        )


def modal_coordinates(
    operators: CentralOperators, *, length_scale: float | None = None
) -> ModalCoordinates:
    """Build the reduced-coordinate operators of one element geometry.

    `length_scale` divides the two amplitude rows, so the reduced state is
    `[eps_c ; q / L]` and all five coordinates are dimensionless. **This is not
    cosmetic.** Without it the first three entries are strains and the last two
    are lengths, the SVD sees the hourglass columns at `q / eps ~ h`, and on the
    campaign element (`h = 1.84e-3 mm`) that is three orders of magnitude. The
    consequence is measured in
    `tests/unit/core/test_limited_memory_broyden.py`: the same physical problem
    described in millimetres and in micrometres yields corrections differing by
    **41 %**, while the element stiffness it corrects is identical to `6e-16`.

    `None` selects `sqrt(area)`, the only length the element supplies. Passing
    `1.0` reproduces the unscaled coordinates the falsified campaign of
    `validation/cps4r_as_broyden_results.md` ran with.
    """

    scale = float(np.sqrt(operators.area)) if length_scale is None else float(length_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"length_scale must be finite and positive, got {scale}")
    gamma = operators.gamma / scale
    amplitude = np.zeros((2, 8))
    amplitude[0, 0::2] = gamma
    amplitude[1, 1::2] = gamma
    centre = operators.strain_displacement_centre
    reduction = np.vstack([centre, amplitude])

    # `T` has full row rank on a valid element, so the right pseudo-inverse is
    # the usual one; `pinv` is used rather than an explicit normal-equation
    # inverse so a nearly degenerate geometry degrades instead of exploding.
    pseudo_inverse = np.linalg.pinv(reduction)
    gram = amplitude @ amplitude.T
    projector = np.linalg.solve(gram, amplitude)
    return ModalCoordinates(
        amplitude_operator=amplitude,
        reduction=reduction,
        reduction_pseudo_inverse=pseudo_inverse,
        force_projector=projector,
    )
