"""Discrete Green operators and plane-stress reference projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReferenceOperatorSymbols:
    """Spectral symbols assembled from the actual discrete gradient-adjoint."""

    laplacian: FloatArray
    directional_x: FloatArray
    directional_y: FloatArray

    def __post_init__(self) -> None:
        shapes = {self.laplacian.shape, self.directional_x.shape, self.directional_y.shape}
        if len(shapes) != 1:
            raise ValueError("reference operator symbols must have identical shapes")
        if not all(np.isfinite(values).all() for values in self.as_tuple()):
            raise ValueError("reference operator symbols must be finite")

    def as_tuple(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        return self.laplacian, self.directional_x, self.directional_y


@dataclass(frozen=True, slots=True)
class GreenDiagnostics:
    minimum_denominator_x: float
    minimum_denominator_y: float
    null_modes: int


class _DiagonalGreen2D:
    def __init__(
        self,
        symbols: ReferenceOperatorSymbols,
        *,
        lambda_0: float,
        mu_0: float,
        symbol_null_tolerance: float = 1.0e-12,
        mode: Literal["b0", "two_mu"] = "b0",
    ) -> None:
        if mu_0 <= 0.0 or lambda_0 + mu_0 <= 0.0:
            raise ValueError("reference Lamé parameters must satisfy mu>0 and lambda+mu>0")
        if symbol_null_tolerance <= 0.0:
            raise ValueError("symbol_null_tolerance must be positive")
        self.symbols = symbols
        self.lambda_0 = float(lambda_0)
        self.mu_0 = float(mu_0)
        self.symbol_null_tolerance = float(symbol_null_tolerance)
        self.mode = mode
        self._denominator_x = np.empty_like(symbols.laplacian)
        self._denominator_y = np.empty_like(symbols.laplacian)
        self._null_mask = np.empty(symbols.laplacian.shape, dtype=bool)
        self.update_parameters(lambda_0=lambda_0, mu_0=mu_0)

    @property
    def diagnostics(self) -> GreenDiagnostics:
        return self._diagnostics

    def update_parameters(self, *, lambda_0: float, mu_0: float) -> None:
        """Update denominators without recreating the transform plan."""

        if not np.isfinite(lambda_0) or not np.isfinite(mu_0):
            raise ValueError("reference Lamé parameters must be finite")
        if mu_0 <= 0.0 or lambda_0 + mu_0 <= 0.0:
            raise ValueError("reference Lamé parameters must satisfy mu>0 and lambda+mu>0")
        self.lambda_0 = float(lambda_0)
        self.mu_0 = float(mu_0)
        np.multiply(2.0 * mu_0, self.symbols.laplacian, out=self._denominator_x)
        if self.mode == "b0":
            np.add(
                self._denominator_x,
                lambda_0 * self.symbols.directional_x,
                out=self._denominator_x,
            )
            np.multiply(2.0 * mu_0, self.symbols.laplacian, out=self._denominator_y)
            np.add(
                self._denominator_y,
                lambda_0 * self.symbols.directional_y,
                out=self._denominator_y,
            )
        else:
            self._denominator_y[...] = self._denominator_x
        self._update_diagnostics()

    def _update_diagnostics(self) -> None:
        scale = max(
            1.0,
            float(np.max(np.abs(self._denominator_x))),
            float(np.max(np.abs(self._denominator_y))),
        )
        null = self.symbol_null_tolerance * scale
        self._null_mask[...] = (np.abs(self._denominator_x) <= null) | (
            np.abs(self._denominator_y) <= null
        )
        self._diagnostics = GreenDiagnostics(
            minimum_denominator_x=float(np.min(np.abs(self._denominator_x))),
            minimum_denominator_y=float(np.min(np.abs(self._denominator_y))),
            null_modes=int(np.count_nonzero(self._null_mask)),
        )

    def apply(self, transformed_polarization: ArrayLike) -> FloatArray:
        polarization = np.asarray(transformed_polarization, dtype=np.float64)
        if polarization.shape[: self._denominator_x.ndim] != self._denominator_x.shape:
            raise ValueError("polarization leading shape must match reference operator symbols")
        if polarization.shape[-1] != 2:
            raise ValueError("polarization must have two displacement components")
        result = np.empty_like(polarization)
        self.apply_into(polarization, result)
        return result

    def apply_into(self, transformed_polarization: ArrayLike, destination: FloatArray) -> None:
        polarization = np.asarray(transformed_polarization, dtype=np.float64)
        if polarization.shape[: self._denominator_x.ndim] != self._denominator_x.shape:
            raise ValueError("polarization leading shape must match reference operator symbols")
        if polarization.shape[-1] != 2:
            raise ValueError("polarization must have two displacement components")
        scale_x = max(1.0, float(np.max(np.abs(self._denominator_x))))
        scale_y = max(1.0, float(np.max(np.abs(self._denominator_y))))
        safe_x = np.abs(self._denominator_x) > self.symbol_null_tolerance * scale_x
        safe_y = np.abs(self._denominator_y) > self.symbol_null_tolerance * scale_y
        if destination.shape != polarization.shape:
            raise ValueError("destination shape must match polarization shape")
        destination[...] = 0.0
        np.divide(
            -polarization[..., 0], self._denominator_x, out=destination[..., 0], where=safe_x
        )
        np.divide(
            -polarization[..., 1], self._denominator_y, out=destination[..., 1], where=safe_y
        )

    def reference_force(self, transformed_displacement: ArrayLike) -> FloatArray:
        """Apply the exact negative reference operator in transform space."""
        displacement = np.asarray(transformed_displacement, dtype=np.float64)
        if displacement.shape[: self._denominator_x.ndim] != self._denominator_x.shape:
            raise ValueError("displacement leading shape must match reference operator symbols")
        if displacement.shape[-1] != 2:
            raise ValueError("displacement must have two components")
        result = np.zeros_like(displacement)
        result[..., 0] = -self._denominator_x * displacement[..., 0]
        result[..., 1] = -self._denominator_y * displacement[..., 1]
        return result


class B0Green2D(_DiagonalGreen2D):
    """Production diagonal Green operator for non-periodic Dirichlet data."""

    def __init__(
        self,
        symbols: ReferenceOperatorSymbols,
        *,
        lambda_0: float,
        mu_0: float,
        symbol_null_tolerance: float = 1.0e-12,
    ) -> None:
        super().__init__(
            symbols,
            lambda_0=lambda_0,
            mu_0=mu_0,
            symbol_null_tolerance=symbol_null_tolerance,
            mode="b0",
        )


class TwoMuGreen2D(_DiagonalGreen2D):
    """Control Green operator retaining only the ``2 mu`` part of ``B0``."""

    def __init__(
        self,
        symbols: ReferenceOperatorSymbols,
        *,
        mu_0: float,
        symbol_null_tolerance: float = 1.0e-12,
    ) -> None:
        super().__init__(
            symbols,
            lambda_0=0.0,
            mu_0=mu_0,
            symbol_null_tolerance=symbol_null_tolerance,
            mode="two_mu",
        )


def isotropic_plane_stress_matrix(lambda_0: float, mu_0: float) -> FloatArray:
    """Return the engineering-component isotropic plane-stress matrix."""

    return np.array(
        [
            [lambda_0 + 2.0 * mu_0, lambda_0, 0.0],
            [lambda_0, lambda_0 + 2.0 * mu_0, 0.0],
            [0.0, 0.0, mu_0],
        ],
        dtype=np.float64,
    )


def project_isotropic_plane_stress_tangent(
    tangent: ArrayLike, *, tolerance: float = 1.0e-12
) -> tuple[float, float, float]:
    """Project a symmetric reference tangent onto isotropic plane stress."""

    matrix = np.asarray(tangent, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("reference tangent must have shape (3, 3)")
    if not np.isfinite(matrix).all():
        raise ValueError("reference tangent must be finite")
    kelvin_scale = np.diag([1.0, 1.0, np.sqrt(2.0)])
    kelvin = kelvin_scale @ matrix @ kelvin_scale
    symmetric = 0.5 * (kelvin + kelvin.T)
    basis_lambda = kelvin_scale @ isotropic_plane_stress_matrix(1.0, 0.0) @ kelvin_scale
    basis_mu = kelvin_scale @ isotropic_plane_stress_matrix(0.0, 1.0) @ kelvin_scale
    coefficients, *_ = np.linalg.lstsq(
        np.column_stack((basis_lambda.reshape(-1), basis_mu.reshape(-1))),
        symmetric.reshape(-1),
        rcond=None,
    )
    lambda_0, mu_0 = (float(value) for value in coefficients)
    scale = max(1.0, float(np.linalg.norm(symmetric)))
    mu_0 = max(mu_0, tolerance * scale)
    lambda_0 = max(lambda_0, tolerance * scale - mu_0)
    projected = kelvin_scale @ isotropic_plane_stress_matrix(lambda_0, mu_0) @ kelvin_scale
    error = float(np.linalg.norm(symmetric - projected) / scale)
    return lambda_0, mu_0, error
