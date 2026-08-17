"""A free tensor plastic increment from local coefficients, admissible after assembly.

```text
a  ->  v(x) = sum_j w_j(x) a_j  ->  P_H  ->  A  ->  J(u; u_obs)  ->  grad_a J
```

Three Kelvin coefficients per patch node instead of one scalar amplitude, so the
plastic increment is free to point anywhere in the dissipative half-space rather
than along the direction J2 dictates. That restriction --

```text
Delta eps^p(x) = Delta p(x) n_J2(sigma(x))
```

-- is precisely what this module removes. The previous milestone qualified the
plumbing around it and is kept; only the family changes.

Two things must not drift.

**The projection comes after assembly.** ``P_H(sum_j w_j a_j)``, never
``sum_j w_j P_H(a_j)``. The reason is *expressiveness*, not admissibility:
``H_sigma`` is a convex cone, so a partition of unity's non-negative blend of
admissible contributions is always admissible -- measured over 200 random
coefficient sets, the mode-wise blend never once left the half-space. What
mode-wise projection does instead is clip every contribution **in isolation**,
which shrinks the reachable family and makes the result depend on an arbitrary
decomposition into modes. The two orders differ by up to 62 % on the same
coefficients. Assembly first keeps the coefficients linear until the single
physical constraint is applied once, to the field that actually exists.

**Everything here is Kelvin.** ``[e_xx, e_yy, sqrt(2) e_xy]`` for strain and
``[s_xx, s_yy, sqrt(2) s_xy]`` for stress. The von Mises metric of
``core.constitutive`` is Voigt, so the J2 comparison arm converts explicitly and
asserts the result rather than trusting the shapes to line up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.constitutive import PLANE_STRESS_VON_MISES_METRIC, von_mises
from fem_inhouse.core.kelvin import KELVIN_SCALE_2D, PLANE_STRESS_PLASTIC_GAUGE
from fem_inhouse.identification.local_coefficient_inverse import _axis_operator
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)

FloatArray = NDArray[np.float64]


def plastic_gauge_norm(field: ArrayLike) -> float:
    """``sqrt(sum_p z_p^T Gp z_p)``, the norm whose value is the equivalent strain.

    A plain Euclidean norm would misweight shear by a factor of three against
    the normal components. Every error reported about a plastic field uses this
    and never ``np.linalg.norm``.
    """

    values = np.asarray(field, dtype=np.float64).reshape(-1, 3)
    return float(
        np.sqrt(max(np.einsum("pi,ij,pj->", values, PLANE_STRESS_PLASTIC_GAUGE, values), 0.0))
    )


def j2_flow_direction(kelvin_stress: ArrayLike) -> FloatArray:
    """The associated J2 direction of a Kelvin stress, returned in Kelvin strain.

    The repository's von Mises metric is Voigt, so the stress is converted down,
    the direction is taken there, and the resulting *engineering strain*
    direction is converted back up. Chaining the two conventions without this is
    the recurring error of the project and doubles the shear.
    """

    kelvin = np.asarray(kelvin_stress, dtype=np.float64).reshape(-1, 3)
    voigt = kelvin / KELVIN_SCALE_2D
    equivalent = von_mises(voigt)
    floor = 1e-12 * max(float(np.max(equivalent)), 1e-300)
    safe = np.where(equivalent > floor, equivalent, 1.0)
    engineering = (voigt @ PLANE_STRESS_VON_MISES_METRIC.T) / safe[:, None]
    engineering[equivalent <= floor] = 0.0
    return np.asarray(engineering / KELVIN_SCALE_2D, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class TensorLocalBasis:
    """``v(x) = sum_j w_j(x) a_j`` with three Kelvin components per patch node.

    Degree zero only, deliberately. Enriching a partition of unity with the
    polynomials it already reproduces gives a linearly dependent basis -- 23 of
    144 directions null at degree one, measured -- and the freedom this
    milestone needs is tensorial, not polynomial.
    """

    row_operator: FloatArray
    column_operator: FloatArray

    @classmethod
    def build(cls, nx: int, ny: int, patches: int) -> TensorLocalBasis:
        return cls(
            row_operator=_axis_operator(nx, patches, 0)[:, :, 0],
            column_operator=_axis_operator(ny, patches, 0)[:, :, 0],
        )

    @property
    def coefficient_shape(self) -> tuple[int, int, int]:
        return (self.row_operator.shape[1], self.column_operator.shape[1], 3)

    @property
    def coefficient_count(self) -> int:
        return int(np.prod(self.coefficient_shape))

    def assemble(self, coefficients: ArrayLike) -> FloatArray:
        values = np.asarray(coefficients, dtype=np.float64).reshape(self.coefficient_shape)
        return np.einsum(
            "xi,yj,ijc->xyc", self.row_operator, self.column_operator, values, optimize=True
        )

    def assemble_transpose(self, field_dual: ArrayLike) -> FloatArray:
        dual = np.asarray(field_dual, dtype=np.float64)
        return np.einsum(
            "xyc,xi,yj->ijc", dual, self.row_operator, self.column_operator, optimize=True
        )


@dataclass(frozen=True, slots=True)
class DissipativeProjection:
    """``P_H(v) = v + ReLU(-sigma^T v) / (sigma^T sigma) sigma``, branch D exactly.

    Only the genuinely anti-dissipative component is removed; every direction
    tangent to the stress survives. This is emphatically not the J2 arm --
    nothing pushes the increment towards ``n_J2`` -- and the half-space is the
    complete thermodynamic condition rather than an in-plane shadow of one,
    because ``sigma_zz = 0`` makes the in-plane Kelvin dot product equal to the
    full three-dimensional ``sigma : eps^p``.

    Incompressibility is **not** projected and must not be. It reads
    ``tr_3 = 0``, so ``e_zz = -(e_xx + e_yy)`` and the in-plane triple stays
    free; demanding a vanishing in-plane trace would force ``e_zz = 0``, a
    plane-strain plasticity this specimen does not have.
    """

    stress: FloatArray

    def __post_init__(self) -> None:
        if self.stress.ndim != 2 or self.stress.shape[1] != 3:
            raise ValueError("stress must have shape (points, 3)")

    @property
    def _denominator(self) -> FloatArray:
        square = np.einsum("pi,pi->p", self.stress, self.stress)
        # Where the stress vanishes the half-space is the whole space; a
        # near-zero denominator would otherwise manufacture an enormous
        # correction out of nothing.
        floor = 1e-12 * max(float(square.mean()), 1e-300)
        return np.maximum(square, floor)

    def apply(self, field: ArrayLike) -> tuple[FloatArray, FloatArray]:
        values = np.asarray(field, dtype=np.float64).reshape(-1, 3)
        overlap = np.einsum("pi,pi->p", self.stress, values)
        active = overlap < 0.0
        correction = np.where(active, -overlap, 0.0) / self._denominator
        return values + correction[:, None] * self.stress, active

    def jacobian_action(self, direction: ArrayLike, active: ArrayLike) -> FloatArray:
        """``dP/dv``, which equals its own transpose; kept separate for clarity."""

        return self.transpose_action(direction, active)

    def transpose_action(self, dual: ArrayLike, active: ArrayLike) -> FloatArray:
        """``(dP/dv)^T``, which equals ``dP/dv``: both branches are symmetric.

        Inactive points pass through; active ones see ``I - s s^T / (s^T s)``,
        the orthogonal projector off the stress direction.
        """

        values = np.asarray(dual, dtype=np.float64).reshape(-1, 3)
        mask = np.asarray(active, dtype=bool)
        overlap = np.einsum("pi,pi->p", self.stress, values)
        scale = np.where(mask, overlap, 0.0) / self._denominator
        return values - scale[:, None] * self.stress

    def dissipation(self, field: ArrayLike) -> FloatArray:
        values = np.asarray(field, dtype=np.float64).reshape(-1, 3)
        return np.einsum("pi,pi->p", self.stress, values)


@dataclass(frozen=True, slots=True)
class TensorInverseEvaluation:
    """One forward pass and everything the registered gates need to read."""

    objective: float
    displacement: FloatArray
    plastic_field: FloatArray
    active_fraction: float
    minimum_dissipation: float


class TensorLocalInverse:
    """Objective and exact adjoint gradient for the free tensor family."""

    def __init__(
        self,
        *,
        operator: TensorPlasticObservabilityOperator,
        basis: TensorLocalBasis,
        projection: DissipativeProjection,
        observed_displacement: ArrayLike,
        scalar_j2_family: bool = False,
        amplitude_channel: int = 1,
    ) -> None:
        self.operator = operator
        self.basis = basis
        self.projection = projection
        self.observed = np.asarray(observed_displacement, dtype=np.float64)
        self.scalar_j2_family = scalar_j2_family
        # Which assembled component the restricted family reads as `Delta p`.
        # It defaults to `yy`, the one a tensile start makes positive: reading a
        # channel that starts negative would pin the arm at `Delta p = 0` with a
        # zero gradient, and the comparison would report a spectacular
        # separation that measures nothing but a dead ReLU.
        self.amplitude_channel = amplitude_channel
        self.grid_shape = (operator.grid.nx, operator.grid.ny)
        self.subcells = operator.kinematics.material_point_count // (
            operator.grid.nx * operator.grid.ny
        )
        # The operator works in gauge coordinates `z` with `eps^p = z H^-1/2`,
        # so the physical Kelvin field is mapped through the inverse. Both
        # matrices are symmetric, which is why the adjoint below reuses the same
        # one without a transpose.
        self._to_gauge = np.linalg.inv(operator.inverse_gauge_root)
        self._flow = (
            j2_flow_direction(projection.stress) if scalar_j2_family else None
        )

    # -- the forward chain ------------------------------------------------

    def plastic_from(self, coefficients: ArrayLike) -> tuple[
        FloatArray, FloatArray, FloatArray
    ]:
        """``a -> v -> P_H(v)``, assembly first and admissibility second."""

        field = self.basis.assemble(coefficients)
        raw = np.repeat(field[:, :, None, :], self.subcells, axis=2).reshape(-1, 3)
        if self._flow is not None:
            # The comparison arm: one scalar amplitude per point along the J2
            # direction, non-negative. Same mechanics, same objective, strictly
            # less freedom -- which is the whole point of running it.
            amplitude = np.maximum(raw[:, self.amplitude_channel], 0.0)
            raw = amplitude[:, None] * self._flow
        projected, active = self.projection.apply(raw)
        return projected, active, field

    def _displacement(self, plastic: FloatArray) -> FloatArray:
        return np.asarray(
            self.operator.matvec((plastic @ self._to_gauge).reshape(-1))
        ).reshape(self.observed.shape)

    def _displacement_transpose(self, dual: FloatArray) -> FloatArray:
        return np.asarray(
            self.operator.rmatvec(dual.reshape(-1))
        ).reshape(-1, 3) @ self._to_gauge

    def evaluate(self, coefficients: ArrayLike) -> TensorInverseEvaluation:
        plastic, active, _ = self.plastic_from(coefficients)
        displacement = self._displacement(plastic)
        residual = displacement - self.observed
        dissipation = self.projection.dissipation(plastic)
        return TensorInverseEvaluation(
            objective=0.5 * float(np.sum(residual**2)),
            displacement=displacement,
            plastic_field=plastic,
            active_fraction=float(np.mean(active)),
            minimum_dissipation=float(np.min(dissipation)),
        )

    # -- the adjoint ------------------------------------------------------

    def gradient(self, coefficients: ArrayLike) -> tuple[
        TensorInverseEvaluation, FloatArray
    ]:
        """``dJ/da``, exactly, with no linear solve beyond the one ``A^T`` does."""

        plastic, active, _ = self.plastic_from(coefficients)
        displacement = self._displacement(plastic)
        residual = displacement - self.observed
        dual = self._displacement_transpose(residual)
        dual = self.projection.transpose_action(dual, active)
        if self._flow is not None:
            field = self.basis.assemble(coefficients)
            raw = np.repeat(field[:, :, None, :], self.subcells, axis=2).reshape(-1, 3)
            scalar = np.einsum("pi,pi->p", dual, self._flow)
            scalar = np.where(raw[:, self.amplitude_channel] > 0.0, scalar, 0.0)
            dual = np.zeros_like(dual)
            dual[:, self.amplitude_channel] = scalar
        per_pixel = dual.reshape(*self.grid_shape, self.subcells, 3).sum(axis=2)
        dissipation = self.projection.dissipation(plastic)
        evaluation = TensorInverseEvaluation(
            objective=0.5 * float(np.sum(residual**2)),
            displacement=displacement,
            plastic_field=plastic,
            active_fraction=float(np.mean(active)),
            minimum_dissipation=float(np.min(dissipation)),
        )
        return evaluation, self.basis.assemble_transpose(per_pixel)

    def sensitivity_column(
        self, coefficients: ArrayLike, direction: ArrayLike
    ) -> FloatArray:
        """``du/da`` applied to one direction, exactly rather than by differences.

        The chain is linear in `a` wherever the projection's activity does not
        change, so a difference quotient would be exact there and wrong across a
        kink. Applying the Jacobian removes the question.
        """

        _, active, _ = self.plastic_from(coefficients)
        field = self.basis.assemble(direction)
        raw = np.repeat(field[:, :, None, :], self.subcells, axis=2).reshape(-1, 3)
        return self._displacement(self.projection.jacobian_action(raw, active))
