"""Typed, unit-explicit parameter coordinates for nonlocal identification."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, sqrt
from typing import Any


def _positive_finite(name: str, value: float) -> float:
    converted = float(value)
    if not isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return converted


@dataclass(frozen=True, slots=True)
class NonlocalIdentificationPoint:
    """One canonical point in ``(alpha, ell)`` and ``(H_chi, A_chi)``.

    Lengths are accepted in micrometres at the user boundary and stored in
    millimetres. The local point is canonical: ``alpha == 0`` always gives
    undefined length and logarithmic coordinates, preventing duplicate local
    calculations for several meaningless lengths.
    """

    alpha: float
    h_ref_mpa: float
    length_scale_mm: float | None

    def __post_init__(self) -> None:
        alpha = float(self.alpha)
        h_ref = _positive_finite("h_ref_mpa", self.h_ref_mpa)
        if not isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        if alpha == 0.0:
            object.__setattr__(self, "alpha", 0.0)
            object.__setattr__(self, "h_ref_mpa", h_ref)
            object.__setattr__(self, "length_scale_mm", None)
            return
        if self.length_scale_mm is None:
            raise ValueError("a positive-alpha point requires length_scale_mm")
        length = _positive_finite("length_scale_mm", self.length_scale_mm)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "h_ref_mpa", h_ref)
        object.__setattr__(self, "length_scale_mm", length)

    @classmethod
    def from_alpha_and_length_um(
        cls,
        *,
        alpha: float,
        length_scale_um: float | None,
        h_ref_mpa: float,
    ) -> NonlocalIdentificationPoint:
        """Build a canonical point from the public coordinates."""

        if float(alpha) == 0.0:
            return cls(alpha=0.0, h_ref_mpa=h_ref_mpa, length_scale_mm=None)
        if length_scale_um is None:
            raise ValueError("a positive-alpha point requires length_scale_um")
        length_um = _positive_finite("length_scale_um", length_scale_um)
        return cls(
            alpha=alpha,
            h_ref_mpa=h_ref_mpa,
            length_scale_mm=length_um / 1_000.0,
        )

    @property
    def is_local(self) -> bool:
        """Whether this is the unique local baseline."""

        return self.alpha == 0.0

    @property
    def h_chi_mpa(self) -> float:
        """Micromorphic coupling modulus in MPa."""

        return self.alpha * self.h_ref_mpa

    @property
    def length_scale_um(self) -> float | None:
        """Spatial length in micrometres, or ``None`` for the local point."""

        if self.length_scale_mm is None:
            return None
        return self.length_scale_mm * 1_000.0

    @property
    def a_chi_mpa_mm2(self) -> float | None:
        """Gradient coefficient ``H_chi * ell**2`` in MPa mm²."""

        if self.length_scale_mm is None:
            return None
        return self.h_chi_mpa * self.length_scale_mm**2

    @property
    def a_chi_mpa_um2(self) -> float | None:
        """Gradient coefficient in MPa µm²."""

        value = self.a_chi_mpa_mm2
        return None if value is None else value * 1_000_000.0

    @property
    def theta_h(self) -> float | None:
        """Natural logarithm of ``H_chi / MPa``."""

        return None if self.is_local else log(self.h_chi_mpa)

    @property
    def theta_a(self) -> float | None:
        """Natural logarithm of ``A_chi / (MPa mm²)``."""

        value = self.a_chi_mpa_mm2
        return None if value is None else log(value)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record with explicit units."""

        return {
            "alpha": self.alpha,
            "h_ref_mpa": self.h_ref_mpa,
            "h_chi_mpa": self.h_chi_mpa,
            "length_scale_mm": self.length_scale_mm,
            "length_scale_um": self.length_scale_um,
            "a_chi_mpa_mm2": self.a_chi_mpa_mm2,
            "a_chi_mpa_um2": self.a_chi_mpa_um2,
            "theta_h_log_mpa": self.theta_h,
            "theta_a_log_mpa_mm2": self.theta_a,
            "is_local": self.is_local,
        }


def from_h_chi_and_a_chi(
    *,
    h_chi_mpa: float,
    a_chi_mpa_mm2: float,
    h_ref_mpa: float,
) -> NonlocalIdentificationPoint:
    """Convert exactly from ``(H_chi, A_chi)`` to the public coordinates."""

    h_chi = _positive_finite("h_chi_mpa", h_chi_mpa)
    a_chi = _positive_finite("a_chi_mpa_mm2", a_chi_mpa_mm2)
    h_ref = _positive_finite("h_ref_mpa", h_ref_mpa)
    return NonlocalIdentificationPoint(
        alpha=h_chi / h_ref,
        h_ref_mpa=h_ref,
        length_scale_mm=sqrt(a_chi / h_chi),
    )
