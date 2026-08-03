"""Immutable single-crystal parameter sets from the literature.

Frozen on purpose. A parameter set is a citation: changing a value silently
would make every result computed with it unattributable. Presets are registered
once, never mutated, and a set that is incomplete says so rather than being
quietly filled with plausible numbers.

The target law is the Meric-Cailletaud FCC single crystal of the TFEL gallery,
`MericCailletaudSingleCrystalViscoPlasticity`: small strain, `StandardElasticity`
brick, 12 octahedral slip systems `<0,1,-1>{1,1,1}`, viscoplastic flow with
Norton exponent `n` and strength `K`, saturating isotropic hardening `Q, b` and
Armstrong-Frederick kinematic hardening `C, d`.

## The interaction matrix has seven coefficients, and slot six is colinear

MFront's FCC interaction matrix takes **exactly seven** coefficients. Decoded by
compiling a probe with distinct values and reading the generated `him`:

| slot | cells of the 12x12 | class |
|---:|---:|---|
| 1 | 12, the diagonal | self |
| 2 | 24 | coplanar |
| 3, 4, 5, 7 | 24 each | junctions |
| 6 | **12**, one partner per system | **colinear** |

Slot 6 having a single partner per system identifies it as the colinear
interaction, and it is the one literature reports as by far the strongest.

A source quoting **six** coefficients uses a different classification, and the
mapping onto these seven slots is not determined by the numbers alone. Such a
set is stored verbatim and marked incomplete.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

#: Number of coefficients MFront expects for an FCC interaction matrix.
FCC_INTERACTION_COEFFICIENTS = 7

#: Slot index, one-based, carrying the colinear interaction.
FCC_COLINEAR_SLOT = 6


@dataclass(frozen=True, slots=True)
class CubicElasticity:
    """Cubic single-crystal stiffness, in MPa.

    Engineering constants are derived rather than stored, so they cannot drift
    away from the stiffnesses they come from.
    """

    c11_mpa: float
    c12_mpa: float
    c44_mpa: float

    def __post_init__(self) -> None:
        if min(self.c11_mpa, self.c44_mpa) <= 0.0:
            raise ValueError("c11 and c44 must be positive")
        if self.c11_mpa <= self.c12_mpa:
            raise ValueError("cubic stability requires c11 > c12")

    @property
    def young_modulus_100_mpa(self) -> float:
        """Young's modulus along <100>, the soft direction."""

        return 1.0 / self._compliance_11

    @property
    def poisson_ratio_100(self) -> float:
        return -self._compliance_12 / self._compliance_11

    @property
    def shear_modulus_mpa(self) -> float:
        return self.c44_mpa

    @property
    def zener_anisotropy(self) -> float:
        """`2 C44 / (C11 - C12)`; unity would be isotropic."""

        return 2.0 * self.c44_mpa / (self.c11_mpa - self.c12_mpa)

    @property
    def _determinant(self) -> float:
        return (self.c11_mpa - self.c12_mpa) * (self.c11_mpa + 2.0 * self.c12_mpa)

    @property
    def _compliance_11(self) -> float:
        return (self.c11_mpa + self.c12_mpa) / self._determinant

    @property
    def _compliance_12(self) -> float:
        return -self.c12_mpa / self._determinant


@dataclass(frozen=True, slots=True)
class SingleCrystalPreset:
    """One published parameter set, complete or not.

    `None` means the source does not give the value. It is never replaced by a
    default: a missing coefficient is a reason to refuse to run, not a reason to
    invent one.
    """

    identifier: str
    material: str
    provenance: str
    elasticity: CubicElasticity
    interaction_matrix: tuple[float, ...]
    interaction_convention: str
    norton_exponent: float | None = None
    norton_strength_mpa: float | None = None
    initial_crss_mpa: float | None = None
    isotropic_saturation_mpa: float | None = None
    isotropic_rate: float | None = None
    kinematic_modulus_mpa: float | None = None
    kinematic_recovery: float | None = None
    notes: str = ""

    def missing(self) -> tuple[str, ...]:
        """Everything the Meric-Cailletaud law needs and this set lacks."""

        gaps: list[str] = []
        required = {
            "norton_exponent": self.norton_exponent,
            "norton_strength_mpa": self.norton_strength_mpa,
            "initial_crss_mpa": self.initial_crss_mpa,
            "isotropic_saturation_mpa": self.isotropic_saturation_mpa,
            "isotropic_rate": self.isotropic_rate,
            "kinematic_modulus_mpa": self.kinematic_modulus_mpa,
            "kinematic_recovery": self.kinematic_recovery,
        }
        gaps.extend(name for name, value in required.items() if value is None)
        if len(self.interaction_matrix) != FCC_INTERACTION_COEFFICIENTS:
            gaps.append(
                f"interaction_matrix has {len(self.interaction_matrix)} coefficients, "
                f"MFront FCC needs {FCC_INTERACTION_COEFFICIENTS}"
            )
        return tuple(gaps)

    @property
    def complete(self) -> bool:
        return not self.missing()

    @property
    def colinear_coefficient(self) -> float | None:
        """The colinear interaction, if the set uses the MFront convention."""

        if len(self.interaction_matrix) != FCC_INTERACTION_COEFFICIENTS:
            return None
        return self.interaction_matrix[FCC_COLINEAR_SLOT - 1]

    def mfront_parameters(self) -> dict[str, Any]:
        """Parameters for the gallery law, or a refusal naming what is absent."""

        gaps = self.missing()
        if gaps:
            raise ValueError(
                f"preset {self.identifier!r} is incomplete and cannot drive the "
                f"Meric-Cailletaud law; missing: {', '.join(gaps)}"
            )
        return {
            "n": self.norton_exponent,
            "K": self.norton_strength_mpa,
            "tau0": self.initial_crss_mpa,
            "Q": self.isotropic_saturation_mpa,
            "b": self.isotropic_rate,
            "C": self.kinematic_modulus_mpa,
            "d": self.kinematic_recovery,
        }

    def elastic_brick_options(self) -> dict[str, float]:
        """`StandardElasticity` options, from the cubic stiffnesses."""

        young = self.young_modulus_100_mpa
        poisson = self.elasticity.poisson_ratio_100
        shear = self.elasticity.shear_modulus_mpa
        return {
            "young_modulus1": young,
            "young_modulus2": young,
            "young_modulus3": young,
            "poisson_ratio12": poisson,
            "poisson_ratio23": poisson,
            "poisson_ratio13": poisson,
            "shear_modulus12": shear,
            "shear_modulus23": shear,
            "shear_modulus13": shear,
        }

    @property
    def young_modulus_100_mpa(self) -> float:
        return self.elasticity.young_modulus_100_mpa


#: Registered sets. This mapping is the record; nothing writes to it at runtime.
SINGLE_CRYSTAL_PRESETS: dict[str, SingleCrystalPreset] = {}


def _register(preset: SingleCrystalPreset) -> SingleCrystalPreset:
    if preset.identifier in SINGLE_CRYSTAL_PRESETS:
        raise ValueError(f"preset {preset.identifier!r} is already registered")
    SINGLE_CRYSTAL_PRESETS[preset.identifier] = preset
    return preset


GUILHEM_NASRI_316L = _register(
    SingleCrystalPreset(
        identifier="316l_guilhem2013_nasri2018",
        material="316L austenitic stainless steel",
        provenance="Guilhem 2013 and Nasri 2018, as supplied 2026-08-03",
        elasticity=CubicElasticity(c11_mpa=197_000.0, c12_mpa=125_000.0, c44_mpa=122_000.0),
        interaction_matrix=(1.0, 1.0, 0.6, 1.8, 1.6, 12.3, 1.6),
        interaction_convention="mfront_fcc_7",
        norton_exponent=11.0,
        norton_strength_mpa=12.0,
        initial_crss_mpa=40.0,
        isotropic_saturation_mpa=10.0,
        isotropic_rate=3.0,
        kinematic_modulus_mpa=40_000.0,
        kinematic_recovery=1_500.0,
        notes=(
            "Complete. The seven interaction coefficients match the TFEL gallery "
            "defaults, with the colinear interaction at 12.3 in slot six. tau0 of "
            "40 MPa is consistent with the repository's own yield map: its mean "
            "164.8 MPa divided by a Taylor factor of 3.06 gives 53.9 MPa, the same "
            "order."
        ),
    )
)

GUERY_316LN_D50 = _register(
    SingleCrystalPreset(
        identifier="316ln_guery2016_d50",
        material="316LN austenitic stainless steel, d50 grain size",
        provenance="Guery 2016, as supplied 2026-08-03",
        interaction_matrix=(1.0, 1.0, 0.6, 12.3, 1.6, 1.3),
        interaction_convention="six_class_unmapped",
        elasticity=CubicElasticity(c11_mpa=207_000.0, c12_mpa=133_000.0, c44_mpa=117_000.0),
        norton_exponent=10.0,
        norton_strength_mpa=25.0,
        kinematic_modulus_mpa=10_400.0,
        kinematic_recovery=340.0,
        notes=(
            "INCOMPLETE, deliberately left so. The source gives six interaction "
            "coefficients where MFront's FCC matrix takes seven, and the mapping is "
            "not determined by the numbers: 12.3 sits in slot four here and in slot "
            "six in the seven-coefficient convention, so the two orderings differ. "
            "The set also gives no tau0, Q or b, so the flow threshold and the "
            "isotropic hardening are absent. Nothing has been guessed; the original "
            "publication is needed."
        ),
    )
)


#: `8 / sqrt(6)`, the geometric factor of equation (16) of Forest and Rubin.
_SRIX_UNIAXIAL_FACTOR = 8.0 / math.sqrt(6.0)


def srix_overstress_modulus_from_meric(
    *,
    norton_strength_mpa: float,
    norton_exponent: float,
    reference_strain_rate: float,
) -> float:
    """Transpose a Meric-Cailletaud pair `(K, n)` into the SRIX modulus `R`.

    Equation (16) of Forest and Rubin (2016):

    ``R = (8 / sqrt(6)) K (sqrt(6) rate / 8) ** (1 / n)``

    It equates the overstress of the two models for the tension of a `[001]`
    single crystal at `reference_strain_rate`. Away from that configuration the
    two laws differ, by construction: this is an analytical transposition, not
    an identification, and nothing here makes `R` a measured property of any
    particular 316L.

    The name says `from_meric` because **this is one route to `R` and not the
    definition of `R`.** `R` is the overstress modulus of the SRIX flow rule and
    can be identified directly, from the width of the measured elastic-plastic
    transition, without any rate-dependent law in the way; that is the route
    registered in `validation/srix_316l_calibration_preregistration.md`. A value
    produced here carries the status `analytical_transposition`, a value fitted
    to data would carry `identified`, and the two must not be confused in a
    manifest.

    The reference strain rate is a required argument with no default. `R`
    carries the rate at which the rate-dependent law was frozen, so a default
    would silently attach an unstated experimental condition to every result.
    """

    if norton_strength_mpa <= 0.0:
        raise ValueError("norton_strength_mpa must be positive")
    if norton_exponent <= 0.0:
        raise ValueError("norton_exponent must be positive")
    if reference_strain_rate <= 0.0:
        raise ValueError("reference_strain_rate must be positive")
    scaled_rate = reference_strain_rate / _SRIX_UNIAXIAL_FACTOR
    return _SRIX_UNIAXIAL_FACTOR * norton_strength_mpa * scaled_rate ** (1.0 / norton_exponent)


def srix_reference_stress(
    *,
    norton_strength_mpa: float,
    norton_exponent: float,
    reference_strain_rate: float,
) -> float:
    """Deprecated alias of `srix_overstress_modulus_from_meric`.

    The old name said what the number is used for, not where it comes from, and
    reading it in a manifest gave no hint that the value was transposed from a
    rate-dependent law rather than measured.
    """

    warnings.warn(
        "srix_reference_stress is deprecated; use "
        "srix_overstress_modulus_from_meric, which names the Meric-Cailletaud "
        "transposition it performs",
        DeprecationWarning,
        stacklevel=2,
    )
    return srix_overstress_modulus_from_meric(
        norton_strength_mpa=norton_strength_mpa,
        norton_exponent=norton_exponent,
        reference_strain_rate=reference_strain_rate,
    )


@dataclass(frozen=True, slots=True)
class SrixPreset:
    """A SRIX parameter set obtained by transposing a Meric-Cailletaud one.

    Everything but the flow rule is inherited from `source`, so the two laws
    cannot drift apart: the elasticity, the interaction matrix and the hardening
    are read from the parent rather than copied.
    """

    identifier: str
    source_identifier: str
    reference_strain_rate: float
    provenance: str
    notes: str = ""

    def __post_init__(self) -> None:
        if self.reference_strain_rate <= 0.0:
            raise ValueError("reference_strain_rate must be positive")
        if not self.source.complete:
            raise ValueError(
                f"SRIX preset {self.identifier!r} derives from the incomplete preset "
                f"{self.source_identifier!r}; missing: {', '.join(self.source.missing())}"
            )

    @property
    def source(self) -> SingleCrystalPreset:
        """The Meric-Cailletaud set this one is transposed from."""

        return get_preset(self.source_identifier)

    @property
    def overstress_modulus_mpa(self) -> float:
        """`R`, recomputed from the parent every time rather than stored."""

        source = self.source
        assert source.norton_strength_mpa is not None
        assert source.norton_exponent is not None
        return srix_overstress_modulus_from_meric(
            norton_strength_mpa=source.norton_strength_mpa,
            norton_exponent=source.norton_exponent,
            reference_strain_rate=self.reference_strain_rate,
        )

    @property
    def elasticity(self) -> CubicElasticity:
        return self.source.elasticity

    @property
    def interaction_matrix(self) -> tuple[float, ...]:
        return self.source.interaction_matrix

    def mfront_parameters(self) -> dict[str, Any]:
        """Parameters of the SRIX law: the hardening set, with `R` for `(K, n)`."""

        inherited = self.source.mfront_parameters()
        for rate_dependent in ("K", "n"):
            del inherited[rate_dependent]
        return {"R": self.overstress_modulus_mpa, **inherited}

    def elastic_brick_options(self) -> dict[str, float]:
        return self.source.elastic_brick_options()

    def provenance_record(self) -> dict[str, Any]:
        """Everything needed to reproduce `R` and to attribute the parameters."""

        source = self.source
        return {
            "identifier": self.identifier,
            "source_preset": self.source_identifier,
            "source_provenance": source.provenance,
            "srix_reference": (
                "Forest and Rubin, European Journal of Mechanics A/Solids 55, "
                "278-288, 2016, doi:10.1016/j.euromechsol.2015.08.012, equation (16)"
            ),
            "hardening_reference": self.provenance,
            "norton_strength_mpa": source.norton_strength_mpa,
            "norton_exponent": source.norton_exponent,
            "reference_strain_rate": self.reference_strain_rate,
            "overstress_modulus_mpa": self.overstress_modulus_mpa,
            "status": (
                "analytical transposition of a rate-dependent parameter set, NOT an "
                "identification of 316L for the SRIX law"
            ),
        }


#: Registered SRIX sets, kept apart from the rate-dependent ones.
SRIX_PRESETS: dict[str, SrixPreset] = {}


def _register_srix(preset: SrixPreset) -> SrixPreset:
    if preset.identifier in SRIX_PRESETS:
        raise ValueError(f"SRIX preset {preset.identifier!r} is already registered")
    SRIX_PRESETS[preset.identifier] = preset
    return preset


def get_srix_preset(identifier: str) -> SrixPreset:
    """Return one registered SRIX preset, naming the alternatives if unknown."""

    try:
        return SRIX_PRESETS[identifier]
    except KeyError:
        known = ", ".join(sorted(SRIX_PRESETS))
        raise KeyError(f"unknown SRIX preset {identifier!r}; registered: {known}") from None


def get_preset(identifier: str) -> SingleCrystalPreset:
    """Return one registered preset, naming the alternatives if it is unknown."""

    try:
        return SINGLE_CRYSTAL_PRESETS[identifier]
    except KeyError:
        known = ", ".join(sorted(SINGLE_CRYSTAL_PRESETS))
        raise KeyError(f"unknown preset {identifier!r}; registered: {known}") from None


# Registered last: a SRIX preset validates its parent on construction, so
# get_preset must already exist by the time this runs.
FOREST_RUBIN_SRIX_316L = _register_srix(
    SrixPreset(
        identifier="316l_forest_rubin_srix_from_nasri2018",
        source_identifier="316l_guilhem2013_nasri2018",
        reference_strain_rate=1.0e-3,
        provenance=(
            "Nasri and others, Comptes Rendus Mecanique 346, 132-151, 2018, "
            "doi:10.1016/j.crme.2017.11.009"
        ),
        notes=(
            "R is a transposition, not a measurement. The reference strain rate of "
            "1e-3 per second is a placeholder chosen to make the number reproducible; "
            "it is NOT the rate of our DIC experiment, which has not been documented "
            "yet. Any campaign result depending on R must state the rate it used."
        ),
    )
)
