"""Admissible nine-parameter SRIX coordinates for observability studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fem_inhouse.core.single_crystal_presets import CubicElasticity
from fem_inhouse.core.srix_parameters import SrixParameterSet

FloatArray = NDArray[np.float64]
SRIX9_NAMES = (
    "log(C11-C12)",
    "log(C11+2C12)",
    "log(C44)",
    "log(tau0)",
    "log(R)",
    "log(Q)",
    "log(b)",
    "log(C)",
    "log(d)",
)


@dataclass(frozen=True, slots=True)
class SrixTheta9:
    """Complete local SRIX parameter vector with stable cubic elasticity."""

    c11_mpa: float
    c12_mpa: float
    c44_mpa: float
    tau0_mpa: float
    r_mpa: float
    q_mpa: float
    b: float
    c_mpa: float
    d: float

    def __post_init__(self) -> None:
        values = self.as_physical_array()
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("all positive SRIX coordinates must be finite")
        if self.c11_mpa <= self.c12_mpa:
            raise ValueError("cubic stability requires C11 > C12")
        if self.c11_mpa + 2.0 * self.c12_mpa <= 0.0:
            raise ValueError("cubic stability requires C11 + 2 C12 > 0")

    def as_physical_array(self) -> FloatArray:
        return np.asarray(
            (self.c11_mpa, self.c12_mpa, self.c44_mpa, self.tau0_mpa,
             self.r_mpa, self.q_mpa, self.b, self.c_mpa, self.d),
            dtype=np.float64,
        )

    def log_coordinates(self) -> FloatArray:
        return np.log(
            np.asarray(
                (
                    self.c11_mpa - self.c12_mpa,
                    self.c11_mpa + 2.0 * self.c12_mpa,
                    self.c44_mpa,
                    self.tau0_mpa,
                    self.r_mpa,
                    self.q_mpa,
                    self.b,
                    self.c_mpa,
                    self.d,
                ),
                dtype=np.float64,
            )
        )

    def as_runtime_overrides(self) -> dict[str, float]:
        """Return explicit names accepted by ``resolve_srix_parameters``."""

        return {
            "C11_mpa": self.c11_mpa,
            "C12_mpa": self.c12_mpa,
            "C44_mpa": self.c44_mpa,
            "tau0_mpa": self.tau0_mpa,
            "R_mpa": self.r_mpa,
            "Q_mpa": self.q_mpa,
            "b": self.b,
            "C_mpa": self.c_mpa,
            "d": self.d,
        }

    @classmethod
    def from_log_coordinates(cls, eta: ArrayLike) -> SrixTheta9:
        values = np.asarray(eta, dtype=np.float64)
        if values.shape != (9,) or not np.isfinite(values).all():
            raise ValueError("eta must contain nine finite coordinates")
        a, bulk, c44, tau0, r, q, b, c, d = np.exp(values)
        c11 = (bulk + 2.0 * a) / 3.0
        c12 = (bulk - a) / 3.0
        return cls(float(c11), float(c12), float(c44), float(tau0), float(r),
                   float(q), float(b), float(c), float(d))

    @classmethod
    def from_parameter_set(cls, parameter_set: SrixParameterSet) -> SrixTheta9:
        elasticity = parameter_set.elasticity
        return cls(
            elasticity.c11_mpa,
            elasticity.c12_mpa,
            elasticity.c44_mpa,
            parameter_set.tau0_mpa,
            parameter_set.overstress_modulus_mpa,
            parameter_set.q_mpa,
            parameter_set.b,
            parameter_set.c_mpa,
            parameter_set.d,
        )

    @property
    def elasticity(self) -> CubicElasticity:
        return CubicElasticity(self.c11_mpa, self.c12_mpa, self.c44_mpa)
