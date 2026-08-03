"""Registered SRIX parameter sets, their provenance, and their MFront names.

Sections 4, 5 and 6 of the 2026-08-03 specification.

Three things live here, and they are deliberately in one place.

**The values.** Every SRIX parameter -- the cubic elasticity, the overstress
modulus `R`, the slip threshold `tau0`, the isotropic pair `(Q, b)` and the
kinematic pair `(C, d)` -- is a registered, immutable set selected by
identifier. Sets are never mutated: a parameter set is a citation, and silently
changing a value would make every result computed with it unattributable.

**Where each value came from.** Not one provenance string for the set, but one
per *group of parameters*, because a single set routinely mixes an elasticity
measured on single crystals, a threshold taken as a literature prior, and a
modulus transposed analytically from a different flow rule. Presenting that
mixture as "the 316L parameters" is the specific error this module exists to
prevent. Each group carries one of five statuses, and a result computed with an
`exploratory` or `analytical_transposition` value must never be reported as a
material identification.

**How to hand them to MFront.** Every parameter of `Fcc316LForestRubinSrix` is
an MFront `@Parameter`, so a set is applied at run time through MGIS
`setParameter` and nothing needs recompiling. `mfront_overrides` returns the
mapping under the names MFront actually exports, which are not the names used
in the papers -- `R` is exported as `SrixOverstressModulus`, and the cubic
elasticity reaches the orthotropic `StandardElasticity` brick as engineering
constants rather than as `C11, C12, C44`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from fem_inhouse.core.single_crystal_presets import (
    CubicElasticity,
    get_srix_preset,
)

#: What is known about where a parameter came from.
#:
#: `identified` -- fitted to measurements of *this* material with *this* law.
#: `literature_measurement` -- measured and published for this material.
#: `literature_prior` -- a published value for a comparable material, adopted
#: as a starting point rather than established here.
#: `analytical_transposition` -- computed from a parameter of a *different*
#: constitutive law through a closed-form correspondence.
#: `exploratory` -- chosen to span a range in a sensitivity study; it carries no
#: claim about any material at all.
ParameterStatus = Literal[
    "identified",
    "literature_measurement",
    "literature_prior",
    "analytical_transposition",
    "exploratory",
]

_CLAIMS_MATERIAL_KNOWLEDGE: frozenset[str] = frozenset(
    {"identified", "literature_measurement"}
)

#: `sqrt(6) / 8`, the uniaxial factor of the Forest-Rubin correspondence.
#: Equation (16) relates the von Mises equivalent rate of a `[001]` tension to
#: the resolved rate on its active systems through this number.
UNIAXIAL_FACTOR = math.sqrt(6.0) / 8.0

#: Temperature every registered set is stated at. Room temperature; none of the
#: sources documents a temperature dependence for these parameters.
ROOM_TEMPERATURE_K = 293.15


@dataclass(frozen=True, slots=True)
class ParameterOrigin:
    """Where one group of parameters came from, and how far it can be trusted."""

    status: ParameterStatus
    reference: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.reference:
            raise ValueError("a parameter origin must cite something, even 'unpublished'")

    @property
    def claims_material_knowledge(self) -> bool:
        """True only for a value that says something about the real material."""

        return self.status in _CLAIMS_MATERIAL_KNOWLEDGE

    def record(self) -> dict[str, Any]:
        return {"status": self.status, "reference": self.reference, "note": self.note}


@dataclass(frozen=True, slots=True)
class SrixParameterSet:
    """One immutable, fully attributed SRIX parameter set."""

    identifier: str
    elasticity: CubicElasticity
    interaction_matrix: tuple[float, ...]
    overstress_modulus_mpa: float
    tau0_mpa: float
    q_mpa: float
    b: float
    c_mpa: float
    d: float
    elastic_origin: ParameterOrigin
    threshold_origin: ParameterOrigin
    isotropic_origin: ParameterOrigin
    kinematic_origin: ParameterOrigin
    overstress_origin: ParameterOrigin
    interaction_origin: ParameterOrigin
    overstress_method: str
    temperature_k: float = ROOM_TEMPERATURE_K
    #: Only meaningful when `R` was transposed from a rate-dependent law: it is
    #: the rate at which that law was frozen. `None` for any other route.
    reference_strain_rate: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("a parameter set must have an identifier")
        for name in ("overstress_modulus_mpa", "tau0_mpa", "q_mpa", "c_mpa"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("b", "d"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if len(self.interaction_matrix) != 7:
            raise ValueError(
                "an FCC interaction matrix has seven coefficients in MFront; got "
                f"{len(self.interaction_matrix)}"
            )
        if self.temperature_k <= 0.0:
            raise ValueError("temperature_k must be positive")
        if self.reference_strain_rate is not None and self.reference_strain_rate <= 0.0:
            raise ValueError("reference_strain_rate must be positive when set")
        transposed = self.overstress_origin.status == "analytical_transposition"
        if transposed and self.reference_strain_rate is None:
            # The transposition is only defined at a rate. Recording the value
            # without it would hide the operating point it was frozen at.
            raise ValueError(
                f"{self.identifier!r} transposes R but states no reference strain rate"
            )

    @property
    def overstress_ratio(self) -> float:
        """`O_R = (sqrt(6) / 8) R / tau0`, section 6.3.

        A dimensionless reading of how abrupt the elastic-plastic transition is
        at first yield: the overstress the flow rule needs to reach the
        equivalent rate, measured against the threshold it has to exceed. Small
        means a sharp corner, large means a rounded one.
        """

        return UNIAXIAL_FACTOR * self.overstress_modulus_mpa / self.tau0_mpa

    @property
    def origins(self) -> dict[str, ParameterOrigin]:
        return {
            "elasticity": self.elastic_origin,
            "tau0": self.threshold_origin,
            "isotropic_hardening": self.isotropic_origin,
            "kinematic_hardening": self.kinematic_origin,
            "overstress_modulus": self.overstress_origin,
            "interaction_matrix": self.interaction_origin,
        }

    @property
    def claims_material_identification(self) -> bool:
        """True only if **every** group is identified or measured on 316L.

        Read this before writing a sentence about the material. One transposed
        or exploratory group is enough to make the whole set a numerical study.
        """

        return all(origin.claims_material_knowledge for origin in self.origins.values())

    def weakest_statuses(self) -> tuple[str, ...]:
        """Groups that stop the set from being an identification, sorted."""

        return tuple(
            sorted(
                name
                for name, origin in self.origins.items()
                if not origin.claims_material_knowledge
            )
        )

    def mfront_overrides(self) -> dict[str, float]:
        """Values keyed by the names MFront exports, ready for `setParameter`.

        The names differ from the ones in the papers, in two ways that have
        both caused confusion already. `R` is exported under its entry name
        `SrixOverstressModulus`. And the cubic elasticity does not reach the
        behaviour as `C11, C12, C44`: the `StandardElasticity` brick is declared
        orthotropic, so it takes engineering constants, and the three cubic
        stiffnesses are converted to one Young's modulus, one Poisson ratio and
        one shear modulus repeated over the three axes. That conversion lives in
        `CubicElasticity`, so it cannot drift from the stiffnesses it comes from.
        """

        young = self.elasticity.young_modulus_100_mpa
        poisson = self.elasticity.poisson_ratio_100
        shear = self.elasticity.shear_modulus_mpa
        return {
            "SrixOverstressModulus": float(self.overstress_modulus_mpa),
            "tau0": float(self.tau0_mpa),
            "Q": float(self.q_mpa),
            "b": float(self.b),
            "C": float(self.c_mpa),
            "d": float(self.d),
            "YoungModulus1": float(young),
            "YoungModulus2": float(young),
            "YoungModulus3": float(young),
            "PoissonRatio12": float(poisson),
            "PoissonRatio23": float(poisson),
            "PoissonRatio13": float(poisson),
            "ShearModulus12": float(shear),
            "ShearModulus23": float(shear),
            "ShearModulus13": float(shear),
        }

    def provenance_record(self) -> dict[str, Any]:
        """Everything section 5 requires about the parameters themselves.

        The run-dependent half of the manifest -- MFront file hash, TFEL and
        MGIS versions, git commit -- is added by `srix_provenance` at solve
        time, because none of it is a property of the parameter set.
        """

        return {
            "identifier": self.identifier,
            "temperature_k": self.temperature_k,
            "reference_strain_rate": self.reference_strain_rate,
            "overstress_method": self.overstress_method,
            "claims_material_identification": self.claims_material_identification,
            "weakest_statuses": list(self.weakest_statuses()),
            "values": {
                "C11_mpa": self.elasticity.c11_mpa,
                "C12_mpa": self.elasticity.c12_mpa,
                "C44_mpa": self.elasticity.c44_mpa,
                "zener_anisotropy": self.elasticity.zener_anisotropy,
                "young_modulus_100_mpa": self.elasticity.young_modulus_100_mpa,
                "poisson_ratio_100": self.elasticity.poisson_ratio_100,
                "R_mpa": self.overstress_modulus_mpa,
                "tau0_mpa": self.tau0_mpa,
                "Q_mpa": self.q_mpa,
                "b": self.b,
                "C_mpa": self.c_mpa,
                "d": self.d,
                "overstress_ratio": self.overstress_ratio,
            },
            "units": {
                "C11_mpa": "MPa",
                "C12_mpa": "MPa",
                "C44_mpa": "MPa",
                "zener_anisotropy": "1",
                "young_modulus_100_mpa": "MPa",
                "poisson_ratio_100": "1",
                "R_mpa": "MPa",
                "tau0_mpa": "MPa",
                "Q_mpa": "MPa",
                "b": "1",
                "C_mpa": "MPa",
                "d": "1",
                "overstress_ratio": "1",
                "temperature_k": "K",
                "reference_strain_rate": "1/s",
            },
            "origins": {name: origin.record() for name, origin in self.origins.items()},
            "interaction_matrix": {
                "coefficients": list(self.interaction_matrix),
                "convention": (
                    "MFront FCC ordering, seven coefficients; slot 6 is the "
                    "colinear interaction. See "
                    "docs/reference/fcc_interaction_matrix_mapping.md"
                ),
            },
            "mfront_parameter_names": sorted(self.mfront_overrides()),
            "notes": self.notes,
        }


SRIX_PARAMETER_SETS: dict[str, SrixParameterSet] = {}


def _register(parameter_set: SrixParameterSet) -> SrixParameterSet:
    if parameter_set.identifier in SRIX_PARAMETER_SETS:
        raise ValueError(f"duplicate SRIX parameter set {parameter_set.identifier!r}")
    SRIX_PARAMETER_SETS[parameter_set.identifier] = parameter_set
    return parameter_set


def get_parameter_set(identifier: str) -> SrixParameterSet:
    try:
        return SRIX_PARAMETER_SETS[identifier]
    except KeyError:
        known = ", ".join(sorted(SRIX_PARAMETER_SETS))
        raise KeyError(
            f"unknown SRIX parameter set {identifier!r}; registered: {known}"
        ) from None


# ---------------------------------------------------------------------------
# Section 6.1 -- the historical set, kept exactly as it was.
#
# The numbers are read from the existing `SrixPreset` rather than retyped, so
# the two cannot drift apart and `R` is still recomputed from `(K, n)` through
# equation (16) every time.
# ---------------------------------------------------------------------------

_HISTORICAL_SOURCE = get_srix_preset("316l_forest_rubin_srix_from_nasri2018")
_NASRI = (
    "Nasri and others, Comptes Rendus Mecanique 346, 132-151, 2018, "
    "doi:10.1016/j.crme.2017.11.009"
)
_HISTORICAL_HARDENING = _HISTORICAL_SOURCE.source.mfront_parameters()

TRANSPOSED_FROM_NASRI2018 = _register(
    SrixParameterSet(
        identifier="316l_srix_transposed_from_nasri2018_rate_1e-3",
        elasticity=_HISTORICAL_SOURCE.elasticity,
        interaction_matrix=_HISTORICAL_SOURCE.interaction_matrix,
        overstress_modulus_mpa=_HISTORICAL_SOURCE.overstress_modulus_mpa,
        tau0_mpa=float(_HISTORICAL_HARDENING["tau0"]),
        q_mpa=float(_HISTORICAL_HARDENING["Q"]),
        b=float(_HISTORICAL_HARDENING["b"]),
        c_mpa=float(_HISTORICAL_HARDENING["C"]),
        d=float(_HISTORICAL_HARDENING["d"]),
        elastic_origin=ParameterOrigin(
            status="literature_prior",
            reference=_NASRI,
            note=(
                "Cubic constants of the Meric-Cailletaud set the SRIX one was "
                "transposed from; adopted, not measured here."
            ),
        ),
        threshold_origin=ParameterOrigin(status="literature_prior", reference=_NASRI),
        isotropic_origin=ParameterOrigin(status="literature_prior", reference=_NASRI),
        kinematic_origin=ParameterOrigin(status="literature_prior", reference=_NASRI),
        overstress_origin=ParameterOrigin(
            status="analytical_transposition",
            reference=(
                "Forest and Rubin, European Journal of Mechanics A/Solids 55, "
                "278-288, 2016, doi:10.1016/j.euromechsol.2015.08.012, equation (16)"
            ),
            note=(
                "Transposed from K = 12 MPa and n = 11 at a reference rate of "
                "1e-3 per second. The rate is a placeholder that makes the number "
                "reproducible; it is NOT the rate of the DIC experiment, which has "
                "not been documented."
            ),
        ),
        interaction_origin=ParameterOrigin(status="literature_prior", reference=_NASRI),
        overstress_method="meric_cailletaud_transposition_equation_16",
        reference_strain_rate=_HISTORICAL_SOURCE.reference_strain_rate,
        notes=(
            "The historical set. Reproduces every archived SRIX result. Not an "
            "identification of 316L for the SRIX law: the flow parameter is "
            "transposed and everything else is a prior."
        ),
    )
)


# ---------------------------------------------------------------------------
# Section 6.2 -- updated single-crystal elasticity, everything else inherited.
# ---------------------------------------------------------------------------

#: The specification supplies these three stiffnesses without a citation, so
#: none is invented here. Recording "primary source not supplied" is the honest
#: statement, and it is what stops the set being read as better attributed than
#: the historical one.
_UPDATED_ELASTIC_REFERENCE = (
    "Supplied by the 2026-08-03 specification as updated 316L single-crystal "
    "constants; primary publication not supplied and deliberately not invented"
)

UPDATED_ELASTICITY_PRIOR = _register(
    SrixParameterSet(
        identifier="316l_srix_updated_elasticity_prior",
        elasticity=CubicElasticity(c11_mpa=218300.0, c12_mpa=144800.0, c44_mpa=125400.0),
        interaction_matrix=TRANSPOSED_FROM_NASRI2018.interaction_matrix,
        overstress_modulus_mpa=TRANSPOSED_FROM_NASRI2018.overstress_modulus_mpa,
        tau0_mpa=38.33,
        q_mpa=TRANSPOSED_FROM_NASRI2018.q_mpa,
        b=TRANSPOSED_FROM_NASRI2018.b,
        c_mpa=TRANSPOSED_FROM_NASRI2018.c_mpa,
        d=TRANSPOSED_FROM_NASRI2018.d,
        elastic_origin=ParameterOrigin(
            status="literature_prior",
            reference=_UPDATED_ELASTIC_REFERENCE,
            note=(
                "Zener anisotropy 3.41 against 3.39 for the historical set, so the "
                "anisotropy is essentially unchanged and the stiffness level is not."
            ),
        ),
        threshold_origin=ParameterOrigin(
            status="literature_prior",
            reference=_UPDATED_ELASTIC_REFERENCE,
            note="38.33 MPa is adopted as a prior, not established here.",
        ),
        isotropic_origin=ParameterOrigin(
            status="literature_prior",
            reference=_NASRI,
            note="Inherited unchanged from the historical set; provisional.",
        ),
        kinematic_origin=ParameterOrigin(
            status="literature_prior",
            reference=_NASRI,
            note="Inherited unchanged from the historical set; provisional.",
        ),
        overstress_origin=TRANSPOSED_FROM_NASRI2018.overstress_origin,
        interaction_origin=ParameterOrigin(
            status="literature_prior",
            reference=_NASRI,
            note="Inherited unchanged from the historical set; provisional.",
        ),
        overstress_method="meric_cailletaud_transposition_equation_16",
        reference_strain_rate=TRANSPOSED_FROM_NASRI2018.reference_strain_rate,
        notes=(
            "Updated elasticity and threshold, everything else explicitly inherited "
            "from the historical set and provisional. Changing tau0 without "
            "re-identifying (Q, b, C, d) moves the whole hardening curve, so this "
            "set is for sensitivity work, not for reporting a material response."
        ),
    )
)


# ---------------------------------------------------------------------------
# Section 6.3 -- the exploratory R series.
# ---------------------------------------------------------------------------

#: Registered sweep of the overstress modulus. Everything but `R` is the
#: historical set, so a difference between two of these is attributable to `R`
#: alone. `18.7819100705` is included so the historical value sits inside the
#: sweep as its own point rather than being compared from outside it.
EXPLORATORY_OVERSTRESS_MODULI_MPA: tuple[float, ...] = (
    1.0,
    2.0,
    4.0,
    8.0,
    18.7819100705,
)


def _exploratory_identifier(value: float) -> str:
    return f"316l_srix_exploratory_r{value:g}".replace(".", "p")


EXPLORATORY_SETS: tuple[SrixParameterSet, ...] = tuple(
    _register(
        SrixParameterSet(
            identifier=_exploratory_identifier(value),
            elasticity=TRANSPOSED_FROM_NASRI2018.elasticity,
            interaction_matrix=TRANSPOSED_FROM_NASRI2018.interaction_matrix,
            overstress_modulus_mpa=value,
            tau0_mpa=TRANSPOSED_FROM_NASRI2018.tau0_mpa,
            q_mpa=TRANSPOSED_FROM_NASRI2018.q_mpa,
            b=TRANSPOSED_FROM_NASRI2018.b,
            c_mpa=TRANSPOSED_FROM_NASRI2018.c_mpa,
            d=TRANSPOSED_FROM_NASRI2018.d,
            elastic_origin=TRANSPOSED_FROM_NASRI2018.elastic_origin,
            threshold_origin=TRANSPOSED_FROM_NASRI2018.threshold_origin,
            isotropic_origin=TRANSPOSED_FROM_NASRI2018.isotropic_origin,
            kinematic_origin=TRANSPOSED_FROM_NASRI2018.kinematic_origin,
            overstress_origin=ParameterOrigin(
                status="exploratory",
                reference=(
                    "Sensitivity sweep registered in section 6.3 of the 2026-08-03 "
                    "specification"
                ),
                note=(
                    "Chosen to span the transition width, not measured and not "
                    "transposed. Carries no claim about any material."
                ),
            ),
            interaction_origin=TRANSPOSED_FROM_NASRI2018.interaction_origin,
            overstress_method="registered_sensitivity_sweep",
            notes=(
                "Material-point sensitivity only. Everything but R matches the "
                "historical set so a difference is attributable to R alone."
            ),
        )
    )
    for value in EXPLORATORY_OVERSTRESS_MODULI_MPA
)

#: Identifier of the set applied when a run selects none. The historical one,
#: so an unconfigured SRIX run reproduces every archived result.
DEFAULT_PARAMETER_SET = TRANSPOSED_FROM_NASRI2018.identifier


# ---------------------------------------------------------------------------
# Section 4 -- turning configuration into MFront parameter values.
# ---------------------------------------------------------------------------

#: Names accepted in an explicit `parameters` block, and where each one lands.
#: The configuration uses the names of the papers, with an explicit unit suffix
#: where there is one; MFront uses its own. Keeping the two vocabularies apart
#: is deliberate: a configuration file should not have to know that `R` is
#: exported as `SrixOverstressModulus`.
EXPLICIT_PARAMETER_NAMES: dict[str, str] = {
    "R_mpa": "SrixOverstressModulus",
    "tau0_mpa": "tau0",
    "Q_mpa": "Q",
    "b": "b",
    "C_mpa": "C",
    "d": "d",
}

#: The cubic stiffnesses are overridden as a group or not at all. Two of the
#: three describe no material, and converting a partial set would silently
#: complete it from whichever preset happened to be underneath.
ELASTIC_PARAMETER_NAMES: tuple[str, ...] = ("C11_mpa", "C12_mpa", "C44_mpa")


def resolve_srix_parameters(
    *,
    parameter_set: Any = None,
    explicit: Any = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Resolve configuration into MFront overrides and a provenance record.

    `parameter_set` selects a registered set by identifier; `explicit` overrides
    individual values on top of it. Both are optional and the default set is
    applied when neither is given, so **every** run records which numbers it
    used rather than leaving "the compiled defaults" implicit.

    Unknown names are refused here, before anything is loaded or solved.
    """

    if parameter_set is None:
        chosen = get_parameter_set(DEFAULT_PARAMETER_SET)
        selected_explicitly = False
    elif isinstance(parameter_set, str):
        chosen = get_parameter_set(parameter_set)
        selected_explicitly = True
    else:
        raise TypeError(
            f"parameter_set must be a registered identifier string, got "
            f"{type(parameter_set).__name__}"
        )

    overrides = chosen.mfront_overrides()
    record = chosen.provenance_record()
    record["selected_explicitly"] = selected_explicitly

    if explicit is None:
        record["explicit_overrides"] = {}
        return overrides, record
    if not isinstance(explicit, Mapping):
        raise TypeError("constitutive_options['parameters'] must be a mapping")

    supplied = dict(explicit)
    elastic = {name: supplied.pop(name) for name in ELASTIC_PARAMETER_NAMES if name in supplied}
    if elastic and len(elastic) != len(ELASTIC_PARAMETER_NAMES):
        missing = sorted(set(ELASTIC_PARAMETER_NAMES) - set(elastic))
        raise ValueError(
            "the cubic stiffnesses are overridden as a group; missing "
            f"{', '.join(missing)}"
        )
    unknown = sorted(name for name in supplied if name not in EXPLICIT_PARAMETER_NAMES)
    if unknown:
        known = ", ".join(sorted(EXPLICIT_PARAMETER_NAMES) + list(ELASTIC_PARAMETER_NAMES))
        raise ValueError(
            f"unknown SRIX parameter(s) {', '.join(unknown)}; accepted: {known}"
        )

    applied: dict[str, Any] = {}
    for name, value in supplied.items():
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"SRIX parameter {name!r} must be finite, got {value!r}")
        overrides[EXPLICIT_PARAMETER_NAMES[name]] = numeric
        applied[name] = numeric
    if elastic:
        stiffness = CubicElasticity(
            c11_mpa=float(elastic["C11_mpa"]),
            c12_mpa=float(elastic["C12_mpa"]),
            c44_mpa=float(elastic["C44_mpa"]),
        )
        young = stiffness.young_modulus_100_mpa
        poisson = stiffness.poisson_ratio_100
        shear = stiffness.shear_modulus_mpa
        for axis in (1, 2, 3):
            overrides[f"YoungModulus{axis}"] = young
        for pair in ("12", "23", "13"):
            overrides[f"PoissonRatio{pair}"] = poisson
            overrides[f"ShearModulus{pair}"] = shear
        applied.update({name: float(value) for name, value in elastic.items()})
        record["values"]["zener_anisotropy"] = stiffness.zener_anisotropy

    for name, value in applied.items():
        key = {"C11_mpa": "C11_mpa", "C12_mpa": "C12_mpa", "C44_mpa": "C44_mpa"}.get(
            name, name
        )
        record["values"][key] = value

    # An inline value carries no attribution: nothing here knows whether it was
    # measured, transposed or guessed. Recording it as exploratory is the only
    # honest status, and it demotes the whole set so the run cannot be reported
    # as an identification on the strength of a preset it no longer matches.
    inline = ParameterOrigin(
        status="exploratory",
        reference="supplied inline in constitutive_options['parameters']",
        note=(
            "Overridden at configuration time. No provenance is available for an "
            "inline value, so it cannot support a claim about the material."
        ),
    )
    overridden_groups = _groups_touched(applied)
    for group in overridden_groups:
        record["origins"][group] = inline.record()
    record["claims_material_identification"] = False
    record["weakest_statuses"] = sorted(
        set(record["weakest_statuses"]) | set(overridden_groups)
    )
    record["explicit_overrides"] = applied
    record["base_parameter_set"] = chosen.identifier
    record["identifier"] = f"{chosen.identifier}+inline"
    return overrides, record


def _behaviour_source_digest(mfront_source: Any) -> dict[str, Any]:
    """SHA-256 of the `.mfront` source, when it can be found on disk.

    The compiled library is what actually runs, but its digest changes with the
    compiler and the flags. The source digest is what says whether the *law*
    changed, and it is the one a reader can check against the repository.
    """

    from hashlib import sha256
    from pathlib import Path

    if mfront_source is None:
        return {"path": None, "sha256": None, "note": "source path not supplied"}
    path = Path(mfront_source)
    if not path.is_file():
        return {
            "path": str(path),
            "sha256": None,
            "note": "source file not found at solve time",
        }
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def _toolchain_versions() -> dict[str, Any]:
    """TFEL and MGIS versions, reported as unavailable rather than guessed."""

    versions: dict[str, Any] = {"tfel": None, "mgis": None}
    try:
        import mgis

        versions["mgis"] = getattr(mgis, "__version__", None)
    except ImportError:
        pass
    try:  # `tfel.__version__` is only present in the Python bindings
        import tfel

        versions["tfel"] = getattr(tfel, "__version__", None)
    except ImportError:
        pass
    if versions["tfel"] is None:
        import shutil
        import subprocess

        executable = shutil.which("tfel-config")
        if executable is not None:
            try:
                versions["tfel"] = subprocess.run(
                    [executable, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                ).stdout.strip()
            except (subprocess.SubprocessError, OSError):
                versions["tfel"] = None
    return versions


def _git_commit() -> str | None:
    """Current commit, or `None` outside a checkout. Never a fabricated value."""

    import subprocess
    from pathlib import Path

    repository = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def srix_provenance(
    *,
    parameter_set: Any = None,
    explicit: Any = None,
    mfront_source: Any = None,
) -> dict[str, Any]:
    """The complete section 5 record: the parameters plus the run that used them.

    Split from `resolve_srix_parameters` because the parameter half is a
    property of the registered set and reproducible anywhere, while this half
    describes one machine at one moment. Anything that cannot be determined is
    recorded as `None` with a note; nothing here is ever guessed, because a
    fabricated version string is worse than an absent one.
    """

    _, record = resolve_srix_parameters(parameter_set=parameter_set, explicit=explicit)
    record["run"] = {
        "mfront_source": _behaviour_source_digest(mfront_source),
        "toolchain": _toolchain_versions(),
        "git_commit": _git_commit(),
    }
    return record


def _groups_touched(applied: Mapping[str, float]) -> tuple[str, ...]:
    """Which provenance groups an explicit override invalidates."""

    mapping = {
        "R_mpa": "overstress_modulus",
        "tau0_mpa": "tau0",
        "Q_mpa": "isotropic_hardening",
        "b": "isotropic_hardening",
        "C_mpa": "kinematic_hardening",
        "d": "kinematic_hardening",
        "C11_mpa": "elasticity",
        "C12_mpa": "elasticity",
        "C44_mpa": "elasticity",
    }
    return tuple(sorted({mapping[name] for name in applied}))
