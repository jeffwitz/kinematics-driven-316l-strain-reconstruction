"""The inverse chain with local plastic coefficients as the variables.

```text
a  ->  Delta p(x) = sum_j w_j(x) sum_k a_jk phi_jk(x)  ->  P_H  ->  equilibrium  ->  J(u; u_obs)
```

No network. The coefficients are the optimisation variables directly, so the
whole inverse chain can be qualified before a generator is introduced and any
failure has exactly one place to be.

Nothing here re-implements mechanics. The forward solve is
``solve_fixed_plastic_increment_equilibrium`` and the adjoint is
``ExperimentalMechanicalOracleLinearisation.jacobian_transpose_action``, which
already returns the displacement *and* plastic duals from one application. What
this module adds is the parameterisation, the admissible projection, and the
chain rule joining them.

The measured result that this design rests on is that the coefficient count does
not multiply the number of global solves -- 64 against 4096 gave identical 8
Newton and ~170 Krylov. A rich local representation is therefore free on the
mechanical side, and `q` may grow without the cost model changing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator, gmres

from fem_inhouse.core.driven_j2 import DrivenJ2PlaneStressBatch
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import DiscreteKinematics2D
from fem_inhouse.workflows.experimental_mechanical_oracle import (
    ExperimentalMechanicalOracleLinearisation,
    FixedIncrementEquilibriumResult,
    solve_fixed_plastic_increment_equilibrium,
)

FloatArray = NDArray[np.float64]


def _legendre(degree: int, coordinate: FloatArray) -> FloatArray:
    """Shifted Legendre polynomials on the patch support ``[-1, 1]``.

    Monomials would do, but they are badly conditioned past degree two and the
    coefficients then stop being comparable between modes, which matters when a
    gradient step treats them as one vector.
    """

    if degree == 0:
        return np.ones_like(coordinate)
    if degree == 1:
        return coordinate
    previous, current = np.ones_like(coordinate), coordinate
    for order in range(2, degree + 1):
        previous, current = current, (
            (2 * order - 1) * coordinate * current - (order - 1) * previous
        ) / order
    return current


def _axis_operator(pixels: int, patches: int, degree: int) -> FloatArray:
    """``W[x, j, k] = w_j(x) phi_k(x - x_j)`` for one axis.

    Keeping the construction separable is not a micro-optimisation. The dense
    ``(pixels^2, patches^2 (degree+1)^2)`` alternative is 34 GB at 1024 square
    with 4096 patches, which is what killed an earlier attempt.
    """

    if patches < 2:
        raise ValueError("patches must be at least 2")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    coordinate = np.linspace(0.0, patches - 1.0, pixels)
    nodes = np.arange(patches, dtype=np.float64)
    offset = coordinate[:, None] - nodes[None, :]
    weight = np.clip(1.0 - np.abs(offset), 0.0, None)
    total = weight.sum(axis=1, keepdims=True)
    if not np.all(total > 0.0):
        raise RuntimeError("the partition of unity has a hole")
    weight = weight / total
    operator = np.empty((pixels, patches, degree + 1), dtype=np.float64)
    for order in range(degree + 1):
        operator[:, :, order] = weight * _legendre(order, offset)
    return operator


@dataclass(frozen=True, slots=True)
class SeparableLocalBasis:
    """``Delta p(x) = sum_j w_j(x) sum_k a_jk phi_jk(x)``, never assembled densely.

    At ``degree = 0`` this reduces **exactly** to the bilinear partition of unity
    the nonlinear bench already uses, which is the check that richer local modes
    did not silently change the meaning of the old coefficients.
    """

    row_operator: FloatArray
    column_operator: FloatArray

    @classmethod
    def build(cls, nx: int, ny: int, patches: int, degree: int) -> SeparableLocalBasis:
        return cls(
            row_operator=_axis_operator(nx, patches, degree),
            column_operator=_axis_operator(ny, patches, degree),
        )

    @property
    def coefficient_shape(self) -> tuple[int, int, int, int]:
        patches = self.row_operator.shape[1]
        modes = self.row_operator.shape[2]
        return (patches, patches, modes, modes)

    @property
    def coefficient_count(self) -> int:
        return int(np.prod(self.coefficient_shape))

    def assemble(self, coefficients: ArrayLike) -> FloatArray:
        values = np.asarray(coefficients, dtype=np.float64).reshape(
            self.coefficient_shape
        )
        return np.einsum(
            "xik,yjl,ijkl->xy",
            self.row_operator,
            self.column_operator,
            values,
            optimize=True,
        )

    def assemble_transpose(self, field_dual: ArrayLike) -> FloatArray:
        dual = np.asarray(field_dual, dtype=np.float64)
        return np.einsum(
            "xy,xik,yjl->ijkl",
            dual,
            self.row_operator,
            self.column_operator,
            optimize=True,
        )


@dataclass(frozen=True, slots=True)
class AdmissibleProjection:
    """``P_H``: the two bounds that make the increment thermodynamically usable.

    ``Delta p >= 0`` is what makes the dissipation sign-definite. The upper bound
    is the wall where associated J2 relaxes the deviatoric stress onto the origin
    and the local equation stops having a solution; the material exposes it in
    closed form, and an optimiser should project onto it rather than discover it
    as an integration failure.
    """

    safety: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 < self.safety < 1.0:
            raise ValueError("safety must lie strictly between 0 and 1")

    def apply(
        self, field: FloatArray, upper_bound: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Return the projected field and the pass-through mask of the chain rule."""

        cap = self.safety * upper_bound
        projected = np.clip(field, 0.0, cap)
        interior = (field > 0.0) & (field < cap)
        return projected, interior.astype(np.float64)


@dataclass(frozen=True, slots=True)
class InverseEvaluation:
    """One forward solve, its objective, and everything the gates need to read."""

    objective: float
    displacement: FloatArray
    plastic_increment: FloatArray
    newton_iterations: int
    krylov_iterations: int
    clipped_fraction: float
    admissibility_margin: float


class LocalCoefficientInverse:
    """Objective and exact adjoint gradient in the local-coefficient variables."""

    def __init__(
        self,
        *,
        material: DrivenJ2PlaneStressBatch,
        kinematics: DiscreteKinematics2D,
        grid: StructuredGrid2D,
        basis: SeparableLocalBasis,
        boundary_displacement: ArrayLike,
        observed_displacement: ArrayLike,
        projection: AdmissibleProjection | None = None,
        preconditioner: LinearOperator | None = None,
        observation_weight: ArrayLike | None = None,
        equilibrium_rms_tolerance: float = 1.0e-10,
        maximum_krylov_iterations: int = 3000,
        adjoint_relative_tolerance: float = 1.0e-12,
    ) -> None:
        self.material = material
        self.kinematics = kinematics
        self.basis = basis
        self.projection = projection or AdmissibleProjection()
        self.preconditioner = preconditioner
        self.boundary_displacement = np.asarray(boundary_displacement, dtype=np.float64)
        self.observed_displacement = np.asarray(observed_displacement, dtype=np.float64)
        if self.observed_displacement.shape != self.boundary_displacement.shape:
            raise ValueError("observed and boundary displacement shapes must agree")
        self.observation_weight = (
            None
            if observation_weight is None
            else np.asarray(observation_weight, dtype=np.float64)
        )
        self.equilibrium_rms_tolerance = equilibrium_rms_tolerance
        self.maximum_krylov_iterations = maximum_krylov_iterations
        self.adjoint_relative_tolerance = adjoint_relative_tolerance
        self._warm_displacement: FloatArray | None = None

        # Taken explicitly rather than off the kinematics: the protocol does
        # not carry a grid, only the concrete classes do.
        self.grid_shape = (grid.nx, grid.ny)
        self.subcells = kinematics.material_point_count // (grid.nx * grid.ny)
        self.interior_shape = (grid.nx - 1, grid.ny - 1, 2)
        self.interior_count = int(np.prod(self.interior_shape))

    # -- the forward chain ------------------------------------------------

    def _upper_bound(self) -> FloatArray:
        """``Delta p_max`` at the warm displacement, reduced over the subcells.

        Taken at the previous iterate rather than the current one, which keeps
        the projection non-circular: the bound needed to build the increment
        cannot depend on the solve that increment drives.
        """

        reference = (
            self.boundary_displacement
            if self._warm_displacement is None
            else self._warm_displacement
        )
        strain = self.kinematics.strain_samples(reference).reshape(-1, 3)
        bound = self.material.maximum_admissible_equivalent_plastic_increment(strain)
        return bound.reshape(*self.grid_shape, self.subcells).min(axis=2)

    def _increment_from(self, coefficients: ArrayLike) -> tuple[
        FloatArray, FloatArray, FloatArray
    ]:
        raw = self.basis.assemble(coefficients)
        bound = self._upper_bound()
        projected, mask = self.projection.apply(raw, bound)
        return projected, mask, bound

    def evaluate(self, coefficients: ArrayLike) -> tuple[
        InverseEvaluation, FixedIncrementEquilibriumResult, FloatArray
    ]:
        projected, mask, bound = self._increment_from(coefficients)
        increment = np.repeat(projected[:, :, None], self.subcells, axis=2)
        result = solve_fixed_plastic_increment_equilibrium(
            material=self.material,
            kinematics=self.kinematics,
            boundary_displacement=self.boundary_displacement,
            equivalent_plastic_increment=increment,
            initial_displacement=self._warm_displacement,
            equilibrium_rms_tolerance=self.equilibrium_rms_tolerance,
            maximum_krylov_iterations=self.maximum_krylov_iterations,
            preconditioner=self.preconditioner,
        )
        self._warm_displacement = result.displacement.copy()
        residual = result.displacement - self.observed_displacement
        if self.observation_weight is not None:
            residual = residual * self.observation_weight
        # The boundary is imposed, so it carries no information and must not
        # enter the objective: a misfit there would be identically zero and
        # would only dilute the interior norm.
        interior = residual[1:-1, 1:-1]
        objective = 0.5 * float(np.sum(interior**2))
        positive = bound > 0.0
        margin = (
            float(np.max(projected[positive] / bound[positive])) if positive.any() else 0.0
        )
        evaluation = InverseEvaluation(
            objective=objective,
            displacement=result.displacement,
            plastic_increment=projected,
            newton_iterations=result.newton_iterations,
            krylov_iterations=int(sum(result.krylov_iterations)),
            clipped_fraction=float(1.0 - mask.mean()),
            admissibility_margin=margin,
        )
        return evaluation, result, mask

    # -- the adjoint ------------------------------------------------------

    def _transpose_operator(
        self, linearisation: ExperimentalMechanicalOracleLinearisation
    ) -> LinearOperator:
        shape = linearisation.displacement_shape

        def action(vector: FloatArray) -> FloatArray:
            dual = np.zeros(shape, dtype=np.float64)
            dual[1:-1, 1:-1] = vector.reshape(self.interior_shape)
            displacement_gradient, _ = linearisation.jacobian_transpose_action(dual)
            return np.asarray(displacement_gradient[1:-1, 1:-1].ravel())

        return LinearOperator(
            (self.interior_count, self.interior_count), matvec=action, dtype=np.float64
        )

    def gradient(self, coefficients: ArrayLike) -> tuple[InverseEvaluation, FloatArray]:
        """``dJ/da`` by adjoint: one linear solve at the converged state.

        The forward problem is ``F(u, p) = 0`` and the objective is ``J(u)``, so

        ```text
        lambda solves  (dF/du)^T lambda = dJ/du,     dJ/dp = -(dF/dp)^T lambda
        ```

        and ``jacobian_transpose_action`` returns both duals from one pass. A
        converged state needs a single linear solve here, not a replayed Newton
        loop.
        """

        evaluation, result, mask = self.evaluate(coefficients)
        linearisation = result.linearisation

        residual = evaluation.displacement - self.observed_displacement
        if self.observation_weight is not None:
            residual = residual * self.observation_weight**2
        right_hand_side = np.zeros_like(residual)
        right_hand_side[1:-1, 1:-1] = residual[1:-1, 1:-1]

        adjoint, info = gmres(
            self._transpose_operator(linearisation),
            right_hand_side[1:-1, 1:-1].ravel(),
            rtol=self.adjoint_relative_tolerance,
            atol=0.0,
            maxiter=self.maximum_krylov_iterations,
            M=self.preconditioner,
        )
        if info != 0 or not np.isfinite(adjoint).all():
            raise RuntimeError(f"the adjoint solve did not converge (info={info})")

        dual = np.zeros(linearisation.displacement_shape, dtype=np.float64)
        dual[1:-1, 1:-1] = adjoint.reshape(self.interior_shape)
        _, plastic_dual = linearisation.jacobian_transpose_action(dual)

        # Back through the subcell broadcast, then through P_H, then the basis.
        per_pixel = -np.asarray(plastic_dual).reshape(
            *self.grid_shape, self.subcells
        ).sum(axis=2)
        gradient = self.basis.assemble_transpose(per_pixel * mask)
        return evaluation, gradient

    # -- identifiability --------------------------------------------------

    def sensitivity_matrix(self, coefficients: ArrayLike) -> tuple[
        InverseEvaluation, FloatArray
    ]:
        """The parameter-to-observable map ``du/da``, one column per coefficient.

        This is the instrument that decides identifiability, and it decides it
        without an optimiser. A descent that stalls can always be blamed on the
        optimiser; a singular-value spectrum cannot. Directions whose singular
        value sits far below the leading one are combinations of local
        coefficients that the displacement simply does not see, and no amount of
        network capacity or iteration count recovers them.

        Costs one linearised solve per coefficient, so it is meant for the small
        grids where a spectrum is affordable, not for production.
        """

        evaluation, result, mask = self.evaluate(coefficients)
        linearisation = result.linearisation
        operator = LinearOperator(
            (self.interior_count, self.interior_count),
            matvec=lambda vector: self._forward_action(linearisation, vector),
            dtype=np.float64,
        )
        columns = np.empty((self.interior_count, self.basis.coefficient_count))
        seed = np.zeros(self.basis.coefficient_count)
        for index in range(self.basis.coefficient_count):
            seed[:] = 0.0
            seed[index] = 1.0
            field = self.basis.assemble(seed.reshape(self.basis.coefficient_shape))
            plastic = np.repeat((field * mask)[:, :, None], self.subcells, axis=2)
            zero = np.zeros(linearisation.displacement_shape, dtype=np.float64)
            right_hand_side = -np.asarray(
                linearisation.jacobian_action(zero, plastic)
            )[1:-1, 1:-1].ravel()
            solution, info = gmres(
                operator, right_hand_side, rtol=self.adjoint_relative_tolerance,
                atol=0.0, maxiter=self.maximum_krylov_iterations, M=self.preconditioner,
            )
            if info != 0 or not np.isfinite(solution).all():
                raise RuntimeError(f"sensitivity column {index} did not converge")
            columns[:, index] = solution
        return evaluation, columns

    def _forward_action(
        self, linearisation: ExperimentalMechanicalOracleLinearisation, vector: FloatArray
    ) -> FloatArray:
        displacement = np.zeros(linearisation.displacement_shape, dtype=np.float64)
        displacement[1:-1, 1:-1] = vector.reshape(self.interior_shape)
        zero = np.zeros(linearisation.plastic_increment_shape, dtype=np.float64)
        action = linearisation.jacobian_action(displacement, zero)
        return np.asarray(action[1:-1, 1:-1].ravel())
