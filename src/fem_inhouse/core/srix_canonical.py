"""Closed-form Forest-Rubin results, and the diagnostics sections 8 and 9 ask for.

Everything here is independent of 316L. It describes the *model*, so it can
falsify the implementation without any material data being settled.

The centrepiece is the `[001]` tension plateau, which the SRIX flow rule admits
in closed form once hardening is switched off, and which the implementation
reproduces to machine precision. The rest is the bookkeeping that says whether a
computed increment is thermodynamically admissible and how hard the flow rule was
pushed to produce it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

#: Number of octahedral systems with a non-zero Schmid factor in `[001]`
#: tension. The other four are exactly unloaded by symmetry.
ACTIVE_SYSTEMS_001 = 8

#: `|m| = 1/sqrt(6)` for each of those eight.
SCHMID_FACTOR_001 = 1.0 / math.sqrt(6.0)

#: `sqrt(6) / 8`. Slip rate per active system divided by the equivalent strain
#: rate in `[001]` tension: eight systems each contribute `|m|` to the axial
#: plastic rate, so `eqrate = (8 / sqrt(6)) * gammadot`.
SLIP_PER_EQUIVALENT_RATE_001 = math.sqrt(6.0) / 8.0


def uniaxial_001_plateau_stress(*, tau0_mpa: float, overstress_modulus_mpa: float) -> float:
    r"""Axial stress of a `[001]` tension without hardening, in closed form.

    With `Q = 0` (or `b = 0`) and `C = 0` the critical resistance stays at
    `tau0` and the back stress at zero, so the SRIX flow rule

    .. math:: \dot\gamma_s = \dot{\bar\varepsilon}\,
              \frac{\langle |\tau_s| - \tau_0\rangle}{R}

    closes on itself. Eight systems are active, each with `|m| = 1/sqrt(6)`, and
    together they produce the axial plastic rate, so
    `gammadot = (sqrt(6)/8) * eqrate`. Substituting and cancelling `eqrate` --
    which is what makes the law rate independent -- leaves

    .. math:: \sigma = \sqrt 6\,\tau_0 + \tfrac{6}{8} R.

    The elastic constants do not appear: the plateau is reached once every
    active system is flowing, and where it sits depends only on the threshold
    and on the overstress modulus.

    Reproduced by the implementation to a relative `1e-16`.
    """

    if tau0_mpa <= 0.0:
        raise ValueError("tau0_mpa must be positive")
    if overstress_modulus_mpa <= 0.0:
        raise ValueError("overstress_modulus_mpa must be positive")
    return math.sqrt(6.0) * tau0_mpa + 0.75 * overstress_modulus_mpa


def uniaxial_001_relative_overstress(
    *, tau0_mpa: float, overstress_modulus_mpa: float
) -> float:
    r"""`(|tau| - tau_0) / tau_0` on the plateau above.

    Equal to the registered overstress ratio `O_R = (sqrt(6)/8) R / tau_0`, which
    is not a coincidence: `O_R` was defined to be exactly this number, so the
    dimensionless label attached to a parameter set is the relative overstress
    the model will actually run at in the reference configuration.
    """

    plateau = uniaxial_001_plateau_stress(
        tau0_mpa=tau0_mpa, overstress_modulus_mpa=overstress_modulus_mpa
    )
    return (plateau * SCHMID_FACTOR_001 - tau0_mpa) / tau0_mpa


def uniaxial_001_slip_per_system(*, axial_plastic_strain: float) -> float:
    """Slip on each of the eight active systems, for a given axial plastic strain."""

    return SLIP_PER_EQUIVALENT_RATE_001 * axial_plastic_strain


@dataclass(frozen=True, slots=True)
class OverstressDiagnostic:
    """Section 9.2. How hard the flow rule was pushed, per increment.

    `eta_s = <|tau_s - X_s| - r_s> / max(r_s, eps)` is the relative overstress.
    It is a *descriptive* quantity: the flow rule is linear in it, so a large
    value is not an error, it means the increment demanded a lot of slip. The
    fractions above one, five and ten percent are reported so a campaign can say
    where it ran, not so a run can be rejected.
    """

    maximum: float
    q99: float
    q95: float
    mean_active: float
    fraction_above_1pc: float
    fraction_above_5pc: float
    fraction_above_10pc: float
    active_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "maximum": self.maximum,
            "q99": self.q99,
            "q95": self.q95,
            "mean_active": self.mean_active,
            "fraction_above_1pc": self.fraction_above_1pc,
            "fraction_above_5pc": self.fraction_above_5pc,
            "fraction_above_10pc": self.fraction_above_10pc,
            "active_count": self.active_count,
        }


def overstress_diagnostic(
    *,
    resolved_stress_mpa: ArrayLike,
    back_stress_mpa: ArrayLike,
    critical_resistance_mpa: ArrayLike,
    floor_mpa: float = 1e-12,
) -> OverstressDiagnostic:
    """Compute `eta_s` over the slip systems and summarise it."""

    tau = np.asarray(resolved_stress_mpa, dtype=float)
    back = np.asarray(back_stress_mpa, dtype=float)
    resistance = np.asarray(critical_resistance_mpa, dtype=float)
    if not (tau.shape == back.shape == resistance.shape):
        raise ValueError("resolved stress, back stress and resistance must share a shape")
    if floor_mpa <= 0.0:
        raise ValueError("floor_mpa must be positive")
    overstress = np.maximum(np.abs(tau - back) - resistance, 0.0)
    eta = overstress / np.maximum(resistance, floor_mpa)
    active = eta > 0.0
    activated = eta[active]
    return OverstressDiagnostic(
        maximum=float(eta.max(initial=0.0)),
        q99=float(np.quantile(eta, 0.99)),
        q95=float(np.quantile(eta, 0.95)),
        mean_active=float(activated.mean()) if activated.size else 0.0,
        fraction_above_1pc=float(np.mean(eta > 0.01)),
        fraction_above_5pc=float(np.mean(eta > 0.05)),
        fraction_above_10pc=float(np.mean(eta > 0.10)),
        active_count=int(active.sum()),
    )


def slip_dissipation_increments(
    *,
    resolved_stress_mpa: ArrayLike,
    back_stress_mpa: ArrayLike,
    slip_increment: ArrayLike,
) -> FloatArray:
    r"""Per-system dissipation increment `(tau_s - X_s) \Delta\gamma_s`.

    Section 9.1 requires this to be non-negative on every system. It is not an
    assumption of the model but a consequence of its flow rule: the sign of
    `Dgamma_s` is `sign(tau_s - X_s)` by construction, so the product is
    `|tau_s - X_s| * |Dgamma_s| >= 0`. A negative value therefore signals an
    implementation defect -- a sign error, or a converged state that does not
    satisfy the flow rule -- and not merely an unusual material.
    """

    tau = np.asarray(resolved_stress_mpa, dtype=float)
    back = np.asarray(back_stress_mpa, dtype=float)
    increment = np.asarray(slip_increment, dtype=float)
    if not (tau.shape == back.shape == increment.shape):
        raise ValueError("stresses and slip increments must share a shape")
    return (tau - back) * increment


@dataclass(frozen=True, slots=True)
class EnergyBalance:
    """Section 9.1's global bookkeeping, in MPa (energy per unit volume)."""

    total_work: float
    elastic_energy: float
    stored_isotropic: float
    stored_kinematic: float
    plastic_dissipation: float

    @property
    def residual(self) -> float:
        """Work not accounted for by the four terms above."""

        return self.total_work - (
            self.elastic_energy
            + self.stored_isotropic
            + self.stored_kinematic
            + self.plastic_dissipation
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "total_work": self.total_work,
            "elastic_energy": self.elastic_energy,
            "stored_isotropic": self.stored_isotropic,
            "stored_kinematic": self.stored_kinematic,
            "plastic_dissipation": self.plastic_dissipation,
            "residual": self.residual,
        }


def kinematic_stored_energy(*, back_strain: ArrayLike, c_mpa: float) -> float:
    r"""Armstrong-Frederick stored energy, `sum_s (C/2) a_s^2`.

    With `X_s = C a_s`, this is the recoverable part of the kinematic hardening.
    The dynamic recovery term `-d a_s |dgamma_s|` dissipates the rest, which is
    why the kinematic contribution to the energy balance is *not* the integral
    of `X_s dgamma_s`.
    """

    if c_mpa < 0.0:
        raise ValueError("c_mpa must be nonnegative")
    values = np.asarray(back_strain, dtype=float)
    return float(0.5 * c_mpa * np.sum(values**2))
