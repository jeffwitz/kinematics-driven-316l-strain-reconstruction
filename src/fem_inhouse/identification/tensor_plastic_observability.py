"""Matrix-free observability of the tensor plastic increment.

The dense qualification prototype assembles ``B``, ``G = B^T C``, ``K`` and
``S = G H^-1 G^T`` explicitly. That is affordable at M20 and impossible beyond:
at M100 the forcing operator alone is ``19602 x 60000``, about 9.4 GB, and the
dense factorisations are cubic on top of that.

None of those matrices is needed. The operator whose singular values answer the
question is

```text
A = W_D M_D K^-1 G H^-1/2,      G = B^T C,      K = B^T C B,
```

and every factor is either a local 3x3 block, an existing field operator, or
one elastic solve. Its non-zero singular values are those of the dense ``T L``
of the prototype: with ``phi = H^-1 G^T S^-1 f`` one has ``G phi = f`` and
``G^T T^T T f = lambda G^T S^-1 f``, and ``G^T`` is injective because ``G`` has
full row rank, so the two spectra coincide.

`K` is the one factor that is genuinely a matrix, but a sparse one: the strain
of a pixel reads its four corner nodes, so a node couples only inside its
``3 x 3`` neighbourhood. Colouring that stencil recovers the whole sparse
operator in eighteen applications, whatever the mesh size, without any
knowledge of the element internals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.sparse as sparse
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator, factorized

from fem_inhouse.core.element import plane_stress_elasticity
from fem_inhouse.core.kelvin import (
    KELVIN_SCALE_2D,
    PLANE_STRESS_PLASTIC_GAUGE,
    stiffness_from_engineering,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.kinematics import TwoSubcellDiagnostic2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior

FloatArray = NDArray[np.float64]


class FieldOperator(Protocol):
    """A self-adjointly-usable nodal field map: the DIC transfer or whitener."""

    def apply(self, values: ArrayLike) -> FloatArray: ...

    def adjoint(self, values: ArrayLike) -> FloatArray: ...


#: Half-width of the nodal coupling of the two-sub-cell strain operator.
_STENCIL_RADIUS = 1
_COLOURS = 2 * _STENCIL_RADIUS + 1


def inverse_gauge_square_root(point_count: int) -> FloatArray:
    """Return ``H^-1/2`` for the Kelvin gauge ``H = G / point_count``.

    ``G`` is the plane-stress plastic gauge: the norm it induces is the
    equivalent plastic strain, so a unit coordinate vector is a unit RMS of
    ``p_eq``. Kelvin does not make ``G`` the identity -- with ``eps_zz`` fixed by
    incompressibility the plane-stress triple is not an orthonormal subspace of
    the 3D deviatoric space -- but it does make every *contraction* in this
    module a plain dot product, which is what the dissipation constraint needs.
    """

    eigenvalues, vectors = np.linalg.eigh(PLANE_STRESS_PLASTIC_GAUGE)
    root = (vectors / np.sqrt(eigenvalues)) @ vectors.T
    return np.asarray(root * np.sqrt(point_count), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class TensorPlasticObservabilityOperator:
    """``A = W_D M_D K^-1 B^T C H^-1/2`` and its adjoint, without dense algebra."""

    grid: StructuredGrid2D
    kinematics: TwoSubcellDiagnostic2D
    elasticity: FloatArray
    inverse_gauge_root: FloatArray
    quadrature_weight: float
    solve_stiffness: Callable[[FloatArray], FloatArray]
    transfer: FieldOperator
    whitener: FieldOperator

    @classmethod
    def build(
        cls,
        grid: StructuredGrid2D,
        *,
        young_modulus_mpa: float,
        poisson_ratio: float,
        transfer: FieldOperator,
        whitener: FieldOperator,
        point_elasticity: ArrayLike | None = None,
    ) -> TensorPlasticObservabilityOperator:
        """Build the operator, optionally with a per-point elastic reference.

        `point_elasticity` carries `(points, 3, 3)` plane-stress stiffnesses, so
        the reference model against which the mechanical defect is measured can
        be the real crystallographic elasticity rather than a homogeneous
        isotropic one. Everything else -- the gauge, the measurement chain, the
        boundary conditions -- is unchanged, which is what makes the two runs
        comparable.
        """

        kinematics = TwoSubcellDiagnostic2D(grid)
        # Everything inside this module is Kelvin. The engineering convention
        # survives only where an external interface imposes it: `strain` returns
        # it and `divergence_from_sample_stress` expects Voigt stress, so the
        # conversion happens at those two calls and nowhere else.
        if point_elasticity is None:
            elasticity = np.broadcast_to(
                stiffness_from_engineering(
                    plane_stress_elasticity(young_modulus_mpa, poisson_ratio)
                ),
                (kinematics.material_point_count, 3, 3),
            ).copy()
        else:
            elasticity = np.asarray(point_elasticity, dtype=np.float64)
            if elasticity.shape != (kinematics.material_point_count, 3, 3):
                raise ValueError(
                    "point_elasticity must have shape "
                    f"{(kinematics.material_point_count, 3, 3)}"
                )
        weight = float(kinematics.sample_quadrature_weight)
        stiffness = _assemble_sparse_stiffness(grid, kinematics, elasticity, weight)
        return cls(
            grid=grid,
            kinematics=kinematics,
            elasticity=elasticity,
            inverse_gauge_root=inverse_gauge_square_root(kinematics.material_point_count),
            quadrature_weight=weight,
            solve_stiffness=factorized(stiffness.tocsc()),
            transfer=transfer,
            whitener=whitener,
        )

    @property
    def plastic_size(self) -> int:
        return self.kinematics.material_point_count * 3

    @property
    def observation_size(self) -> int:
        return int(np.prod(self.grid.node_shape) * 2)

    @property
    def free_size(self) -> int:
        return 2 * (self.grid.nx - 1) * (self.grid.ny - 1)

    def _strain_transpose(self, stress: FloatArray) -> FloatArray:
        """``B_K^T`` applied to a Kelvin stress.

        `B_K = B / S`, so `B_K^T sigma_K = B^T (sigma_K / S)` -- the Kelvin
        stress is converted back to Voigt and handed to the existing operator.
        The stiffness `K = B_K^T C_K B_K` is unchanged by the migration, since
        the two scalings cancel: `(B/S)^T (S C S) (B/S) = B^T C B`.
        """

        voigt = stress.reshape(-1, 3) / KELVIN_SCALE_2D
        nodal = self.kinematics.divergence_from_sample_stress(
            voigt.reshape((self.grid.nx, self.grid.ny, 2, 3))
        )
        return -pack_interior(nodal) / self.quadrature_weight

    def kelvin_strain(self, displacement: ArrayLike) -> FloatArray:
        """The kinematics returns engineering shear; divide it into Kelvin."""

        engineering = np.asarray(
            self.kinematics.strain(displacement), dtype=np.float64
        ).reshape(-1, 3)
        return engineering / KELVIN_SCALE_2D

    def matvec(self, values: ArrayLike) -> FloatArray:
        vector = np.asarray(values, dtype=np.float64).reshape(-1, 3)
        plastic = vector @ self.inverse_gauge_root
        stress = np.einsum("pi,pij->pj", plastic, self.elasticity)
        displacement = unpack_interior(
            self.solve_stiffness(self._strain_transpose(stress.reshape(-1))), self.grid
        )
        observed = self.whitener.apply(self.transfer.apply(displacement))
        return np.asarray(observed, dtype=np.float64).reshape(-1)

    def kelvin_response(self, plastic: ArrayLike) -> FloatArray:
        """Kelvin strain produced by a Kelvin plastic field, without observation."""

        stress = np.einsum(
            "pi,pij->pj", np.asarray(plastic, dtype=np.float64).reshape(-1, 3), self.elasticity
        )
        displacement = unpack_interior(
            self.solve_stiffness(self._strain_transpose(stress.reshape(-1))), self.grid
        )
        return self.kelvin_strain(displacement).reshape(-1)

    def rmatvec(self, values: ArrayLike) -> FloatArray:
        field = np.asarray(values, dtype=np.float64).reshape((*self.grid.node_shape, 2))
        dual = self.transfer.adjoint(self.whitener.adjoint(field))
        displacement = unpack_interior(
            self.solve_stiffness(pack_interior(np.asarray(dual, dtype=np.float64))), self.grid
        )
        strain = self.kelvin_strain(displacement)
        stress = np.einsum("pi,pij->pj", strain, self.elasticity)
        return (stress @ self.inverse_gauge_root).reshape(-1)

    def as_linear_operator(self) -> LinearOperator:
        return LinearOperator(
            (self.observation_size, self.plastic_size),
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=np.float64,
        )

    def plastic_mode(self, right_singular_vector: ArrayLike) -> FloatArray:
        """Map a right singular vector back to a plastic field, ``phi = H^-1/2 x``."""

        vector = np.asarray(right_singular_vector, dtype=np.float64).reshape(-1, 3)
        return (vector @ self.inverse_gauge_root).reshape(-1)


def _assemble_sparse_stiffness(
    grid: StructuredGrid2D,
    kinematics: TwoSubcellDiagnostic2D,
    elasticity: FloatArray,
    weight: float,
) -> sparse.coo_matrix:
    """Recover ``K = B^T C B`` exactly, by colouring its ``3 x 3`` node stencil.

    A pixel's strain reads its four corner nodes, so an interior node couples
    only to the eight around it. Probing with one unit per congruence class
    modulo three therefore isolates a single source per target, and eighteen
    applications reproduce the operator exactly at any mesh size. The result is
    checked against a random product by the caller's validation, not assumed.
    """

    interior = (grid.nx - 1, grid.ny - 1)
    free_size = 2 * interior[0] * interior[1]
    index = np.arange(free_size, dtype=np.int64).reshape(*interior, 2)
    rows: list[NDArray] = []
    columns: list[NDArray] = []
    values: list[NDArray] = []

    for offset_x in range(_COLOURS):
        for offset_y in range(_COLOURS):
            for component in range(2):
                probe = np.zeros((*interior, 2), dtype=np.float64)
                probe[offset_x::_COLOURS, offset_y::_COLOURS, component] = 1.0
                engineering = np.asarray(
                    kinematics.strain(unpack_interior(probe.reshape(-1), grid)),
                    dtype=np.float64,
                ).reshape(-1, 3)
                stress = np.einsum(
                    "pi,pij->pj", engineering / KELVIN_SCALE_2D, elasticity
                )
                nodal = kinematics.divergence_from_sample_stress(
                    (stress / KELVIN_SCALE_2D).reshape((grid.nx, grid.ny, 2, 3))
                )
                response = (-pack_interior(nodal) / weight).reshape(*interior, 2)

                # For every target node, the unique source of this colour that
                # can reach it lies within one cell.
                for shift_x in (-1, 0, 1):
                    for shift_y in (-1, 0, 1):
                        targets_x = np.arange(interior[0])
                        targets_y = np.arange(interior[1])
                        sources_x = targets_x + shift_x
                        sources_y = targets_y + shift_y
                        keep_x = (
                            (sources_x >= 0)
                            & (sources_x < interior[0])
                            & (sources_x % _COLOURS == offset_x)
                        )
                        keep_y = (
                            (sources_y >= 0)
                            & (sources_y < interior[1])
                            & (sources_y % _COLOURS == offset_y)
                        )
                        if not keep_x.any() or not keep_y.any():
                            continue
                        grid_x, grid_y = np.meshgrid(
                            targets_x[keep_x], targets_y[keep_y], indexing="ij"
                        )
                        source_grid_x, source_grid_y = np.meshgrid(
                            sources_x[keep_x], sources_y[keep_y], indexing="ij"
                        )
                        for target_component in range(2):
                            entry = response[grid_x, grid_y, target_component].reshape(-1)
                            rows.append(index[grid_x, grid_y, target_component].reshape(-1))
                            columns.append(
                                index[source_grid_x, source_grid_y, component].reshape(-1)
                            )
                            values.append(entry)

    matrix = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(free_size, free_size),
    )
    return matrix.tocsr()
