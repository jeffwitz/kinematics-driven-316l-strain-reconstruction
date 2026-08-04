"""Nodal-to-pixel kinematics and exact adjoint divergences."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.spectral2d.green import ReferenceOperatorSymbols
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.transforms import TransformPlan2D

FloatArray = NDArray[np.float64]


def _nodal_displacement(values: ArrayLike, grid: StructuredGrid2D) -> FloatArray:
    displacement = np.asarray(values, dtype=np.float64)
    expected = (*grid.node_shape, 2)
    if displacement.shape != expected:
        raise ValueError(f"expected displacement shape {expected}, got {displacement.shape}")
    return displacement


def _stress(values: ArrayLike, expected: tuple[int, ...]) -> FloatArray:
    stress = np.asarray(values, dtype=np.float64)
    if stress.shape != expected:
        raise ValueError(f"expected stress shape {expected}, got {stress.shape}")
    return stress


@runtime_checkable
class DiscreteKinematics2D(Protocol):
    """Spatial contract shared by pixel and sub-pixel discretisations."""

    @property
    def material_point_count(self) -> int: ...

    @property
    def points_per_pixel(self) -> int: ...

    def strain(self, nodal_displacement: ArrayLike) -> FloatArray: ...

    def divergence(self, stress: ArrayLike) -> FloatArray: ...

    def reference_operator_symbols(
        self, transform_plan: TransformPlan2D
    ) -> ReferenceOperatorSymbols: ...


class CellCenteredOnePoint2D:
    """2D HEX1 analogue: four nodal values and one material state per pixel.

    The neutral name describes data placement, not improved stability.  This
    stencil retains the near-hourglass risk of the published HEX1 scheme.
    """

    def __init__(self, grid: StructuredGrid2D) -> None:
        self.grid = grid

    @property
    def material_point_count(self) -> int:
        return self.grid.nx * self.grid.ny

    @property
    def points_per_pixel(self) -> int:
        return 1

    def reference_operator_symbols(
        self, transform_plan: TransformPlan2D
    ) -> ReferenceOperatorSymbols:
        """Return closed-form symbols of the four-node quadrilateral stencil."""
        return _closed_form_symbols(self.grid, transform_plan, averaged=True)

    def strain(self, nodal_displacement: ArrayLike) -> FloatArray:
        u = _nodal_displacement(nodal_displacement, self.grid)
        ux = u[..., 0]
        uy = u[..., 1]
        exx = (ux[1:, :-1] + ux[1:, 1:] - ux[:-1, :-1] - ux[:-1, 1:]) / (2.0 * self.grid.spacing_x)
        eyy = (uy[:-1, 1:] + uy[1:, 1:] - uy[:-1, :-1] - uy[1:, :-1]) / (2.0 * self.grid.spacing_y)
        gamma = (ux[:-1, 1:] + ux[1:, 1:] - ux[:-1, :-1] - ux[1:, :-1]) / (
            2.0 * self.grid.spacing_y
        ) + (uy[1:, :-1] + uy[1:, 1:] - uy[:-1, :-1] - uy[:-1, 1:]) / (2.0 * self.grid.spacing_x)
        return np.stack((exx, eyy, gamma), axis=-1)

    def divergence(self, stress: ArrayLike) -> FloatArray:
        sigma = _stress(stress, (*self.grid.pixel_shape, 3))
        result = np.zeros((*self.grid.node_shape, 2), dtype=np.float64)
        area = self.grid.spacing_x * self.grid.spacing_y
        sx = sigma[..., 0] * area / (2.0 * self.grid.spacing_x)
        sy = sigma[..., 1] * area / (2.0 * self.grid.spacing_y)
        st = sigma[..., 2] * area
        # This is -B^T with the same four-corner coefficients as strain().
        result[:-1, :-1, 0] += sx
        result[:-1, 1:, 0] += sx
        result[1:, :-1, 0] -= sx
        result[1:, 1:, 0] -= sx
        result[:-1, :-1, 1] += sy
        result[1:, :-1, 1] += sy
        result[:-1, 1:, 1] -= sy
        result[1:, 1:, 1] -= sy
        shear_x = st / (2.0 * self.grid.spacing_y)
        shear_y = st / (2.0 * self.grid.spacing_x)
        result[:-1, :-1, 0] += shear_x
        result[1:, :-1, 0] += shear_x
        result[:-1, 1:, 0] -= shear_x
        result[1:, 1:, 0] -= shear_x
        result[:-1, :-1, 1] += shear_y
        result[:-1, 1:, 1] += shear_y
        result[1:, :-1, 1] -= shear_y
        result[1:, 1:, 1] -= shear_y
        return result


class TwoSubcellDiagnostic2D:
    """Two independent constant-strain triangles per pixel."""

    def __init__(self, grid: StructuredGrid2D) -> None:
        self.grid = grid

    @property
    def material_point_count(self) -> int:
        return 2 * self.grid.nx * self.grid.ny

    @property
    def points_per_pixel(self) -> int:
        return 2

    def reference_operator_symbols(
        self, transform_plan: TransformPlan2D
    ) -> ReferenceOperatorSymbols:
        """Return closed-form symbols of the two-triangle stencil."""
        return _closed_form_symbols(self.grid, transform_plan, averaged=False)

    def strain(self, nodal_displacement: ArrayLike) -> FloatArray:
        u = _nodal_displacement(nodal_displacement, self.grid)
        result = np.empty((*self.grid.pixel_shape, 2, 3), dtype=np.float64)
        bl, br = u[:-1, :-1], u[1:, :-1]
        tl, tr = u[:-1, 1:], u[1:, 1:]
        result[..., 0, 0] = (br[..., 0] - bl[..., 0]) / self.grid.spacing_x
        result[..., 0, 1] = (tl[..., 1] - bl[..., 1]) / self.grid.spacing_y
        result[..., 0, 2] = (tl[..., 0] - bl[..., 0]) / self.grid.spacing_y + (
            br[..., 1] - bl[..., 1]
        ) / self.grid.spacing_x
        result[..., 1, 0] = (tr[..., 0] - tl[..., 0]) / self.grid.spacing_x
        result[..., 1, 1] = (tr[..., 1] - br[..., 1]) / self.grid.spacing_y
        result[..., 1, 2] = (tr[..., 0] - br[..., 0]) / self.grid.spacing_y + (
            tr[..., 1] - tl[..., 1]
        ) / self.grid.spacing_x
        return result

    def divergence(self, stress: ArrayLike) -> FloatArray:
        sigma = _stress(stress, (*self.grid.pixel_shape, 2, 3))
        result = np.zeros((*self.grid.node_shape, 2), dtype=np.float64)
        area = 0.5 * self.grid.spacing_x * self.grid.spacing_y
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                bl = (i, j)
                br = (i + 1, j)
                tl = (i, j + 1)
                tr = (i + 1, j + 1)
                s1, s2 = sigma[i, j]
                result[bl][0] += area * (s1[0] / self.grid.spacing_x + s1[2] / self.grid.spacing_y)
                result[br][0] += area * (-s1[0] / self.grid.spacing_x)
                result[tl][0] += area * (-s1[2] / self.grid.spacing_y)
                result[bl][1] += area * (s1[1] / self.grid.spacing_y + s1[2] / self.grid.spacing_x)
                result[br][1] += area * (-s1[2] / self.grid.spacing_x)
                result[tl][1] += area * (-s1[1] / self.grid.spacing_y)
                result[tr][0] += area * (-s2[0] / self.grid.spacing_x - s2[2] / self.grid.spacing_y)
                result[tl][0] += area * (s2[0] / self.grid.spacing_x)
                result[br][0] += area * (s2[2] / self.grid.spacing_y)
                result[tr][1] += area * (-s2[1] / self.grid.spacing_y - s2[2] / self.grid.spacing_x)
                result[tl][1] += area * (s2[2] / self.grid.spacing_x)
                result[br][1] += area * (s2[1] / self.grid.spacing_y)
        return result


def _closed_form_symbols(
    grid: StructuredGrid2D, transform_plan: TransformPlan2D, *, averaged: bool
) -> ReferenceOperatorSymbols:
    frequencies_x = np.asarray(transform_plan.frequencies_x, dtype=np.float64)
    frequencies_y = np.asarray(transform_plan.frequencies_y, dtype=np.float64)
    shape = (frequencies_x.size, frequencies_y.size)
    directional_x = np.broadcast_to(
        4.0 * grid.spacing_y / grid.spacing_x * np.sin(frequencies_x[:, None] / 2.0) ** 2,
        shape,
    ).copy()
    directional_y = np.broadcast_to(
        4.0 * grid.spacing_x / grid.spacing_y * np.sin(frequencies_y[None, :] / 2.0) ** 2,
        shape,
    ).copy()
    if averaged:
        directional_x *= np.cos(frequencies_y[None, :] / 2.0) ** 2
        directional_y *= np.cos(frequencies_x[:, None] / 2.0) ** 2
    return ReferenceOperatorSymbols(
        laplacian=directional_x + directional_y,
        directional_x=directional_x,
        directional_y=directional_y,
    )


def _modal_symbols(
    operator: CellCenteredOnePoint2D | TwoSubcellDiagnostic2D, transform_plan: TransformPlan2D
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Probe normal and transverse scalar contributions in the DST basis."""
    shape = (len(transform_plan.frequencies_x), len(transform_plan.frequencies_y))
    directional_x = np.empty(shape, dtype=np.float64)
    directional_y = np.empty(shape, dtype=np.float64)
    laplacian = np.empty(shape, dtype=np.float64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            mode = np.zeros(shape, dtype=np.float64)
            mode[i, j] = 1.0
            interior = transform_plan.inverse_displacement(mode)
            displacement = np.zeros((*operator.grid.node_shape, 2), dtype=np.float64)

            displacement[1:-1, 1:-1, 0] = interior
            strain_x = operator.strain(displacement)
            stress_x = np.zeros_like(strain_x)
            stress_x[..., 0] = strain_x[..., 0]
            response_x_normal = -operator.divergence(stress_x)[1:-1, 1:-1, 0]
            stress_x[..., 2] = strain_x[..., 2]
            response_x = -operator.divergence(stress_x)[1:-1, 1:-1, 0]

            displacement.fill(0.0)
            displacement[1:-1, 1:-1, 1] = interior
            strain_y = operator.strain(displacement)
            stress_y = np.zeros_like(strain_y)
            stress_y[..., 1] = strain_y[..., 1]
            response_y_normal = -operator.divergence(stress_y)[1:-1, 1:-1, 1]
            stress_y[..., 2] = strain_y[..., 2]
            response_y = -operator.divergence(stress_y)[1:-1, 1:-1, 1]

            transformed_x = transform_plan.forward_displacement(response_x)
            transformed_x_normal = transform_plan.forward_displacement(response_x_normal)
            transformed_y = transform_plan.forward_displacement(response_y)
            transformed_y_normal = transform_plan.forward_displacement(response_y_normal)
            laplacian[i, j] = transformed_x[i, j]
            directional_x[i, j] = transformed_x[i, j]
            directional_x[i, j] = transformed_x_normal[i, j]
            directional_y[i, j] = transformed_y_normal[i, j]
            if not np.isclose(laplacian[i, j], transformed_y[i, j], rtol=1.0e-11, atol=1.0e-12):
                raise ValueError("the discrete scalar Laplacian is not component-independent")
    return laplacian, directional_x, directional_y
