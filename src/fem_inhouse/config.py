"""Validated configuration objects for the supported 316L case study."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

#: Backends the solver implements directly, without going through a plugin.
BUILTIN_CONSTITUTIVE_BACKENDS = frozenset(
    {
        "python",
        "mfront",
        "mfront-native-plane-stress",
        "mfront-3d-condensed-plane-stress",
    }
)


def known_constitutive_backends() -> frozenset[str]:
    """Builtin backends plus every registered plugin identifier.

    Imported lazily: `constitutive_plugins` must not be a hard dependency of
    the configuration layer, and entry-point discovery must not run at import
    time of this module.
    """

    from fem_inhouse.core.constitutive_plugins import (
        CONSTITUTIVE_PLUGINS,
        load_constitutive_plugins,
    )

    load_constitutive_plugins()
    return BUILTIN_CONSTITUTIVE_BACKENDS | frozenset(CONSTITUTIVE_PLUGINS.identifiers())


@dataclass(frozen=True, slots=True)
class MaterialConfig:
    """Homogeneous material constants and legacy Abaqus-table settings.

    Stress-like quantities use MPa and strains are dimensionless.
    The table controls are ignored by the default analytical MFront backend.
    """

    young_modulus_mpa: float = 205_000.0
    poisson_ratio: float = 0.30
    hardening_exponent: float = 0.245
    plastic_strain_max: float = 0.2
    plastic_table_points: int = 1_000
    first_positive_plastic_strain: float = 1e-6

    def __post_init__(self) -> None:
        if self.young_modulus_mpa <= 0:
            raise ValueError("young_modulus_mpa must be positive")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must satisfy -1 < nu < 0.5")
        if self.hardening_exponent <= 0:
            raise ValueError("hardening_exponent must be positive")
        if self.plastic_strain_max <= 0:
            raise ValueError("plastic_strain_max must be positive")
        if self.plastic_table_points < 3:
            raise ValueError("plastic_table_points must be at least 3")
        if not 0 < self.first_positive_plastic_strain < self.plastic_strain_max:
            raise ValueError(
                "first_positive_plastic_strain must lie inside the plastic-strain range"
            )


@dataclass(frozen=True, slots=True)
class MeshConfig:
    """Structured pixel mesh settings."""

    nx: int
    ny: int
    base_pixel_size_mm: float = 0.001
    scale_factor: float = 1.84

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx and ny must be positive")
        if self.base_pixel_size_mm <= 0:
            raise ValueError("base_pixel_size_mm must be positive")
        if self.scale_factor <= 0:
            raise ValueError("scale_factor must be positive")

    @property
    def element_size_mm(self) -> float:
        return self.base_pixel_size_mm * self.scale_factor

    @property
    def physical_size_mm(self) -> tuple[float, float]:
        spacing = self.element_size_mm
        return self.nx * spacing, self.ny * spacing


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Nonlinear solution controls."""

    increments: int = 20
    max_newton_iterations: int = 15
    residual_tolerance: float = 1e-6
    minimum_step_divisor: int = 1_024
    #: `cps4` is the reference formulation and stays the default.
    #:
    #: `cps4r` integrates one material point per element instead of four, which
    #: is what makes crystal plasticity affordable, and always carries stiffness
    #: hourglass control. It is an approximation of `cps4`, exact only while the
    #: response stays elastic; after yielding the stabilisation remains built on
    #: the elastic reference, so the two formulations genuinely differ and the
    #: hourglass energy ratio is what says by how much.
    #: `cps4r_as` is the QUAS4 assumed-strain element: one constitutive point,
    #: stabilised on the CURRENT tangent and with no free coefficient. Not
    #: qualified; see validation/cps4r_assumed_strain_preregistration.md.
    element_formulation: Literal["cps4", "cps4r", "cps4r_as"] = "cps4"
    #: Assumed-strain strategy and projection, used only by `cps4r_as`.
    stabilisation_strategy: Literal[
        "assumed_strain_current",
        "assumed_strain_energy",
        "assumed_strain_energy_lagged",
    ] = "assumed_strain_energy"
    stabilisation_projection: Literal["quad4", "asmd", "asoi", "asoi_half"] = "asmd"
    #: Spectral floor of the energy variant, as a fraction of the largest
    #: eigenvalue. A conditioning rule, not a tuning knob.
    stabilisation_tangent_floor: float = 1.0e-6
    #: Quasi-Newton correction of the element Jacobian, used only by `cps4r_as`.
    #: `broyden` learns the missing `(df_stab/dC)(dC/du)` term from the secant
    #: pairs the iteration already produces. It changes the MATRIX only: the
    #: residual, the stresses and the converged solution are untouched, so the
    #: sole observable is the iteration count. Not qualified; see
    #: validation/cps4r_as_broyden_preregistration.md.
    jacobian_correction: Literal["none", "broyden"] = "none"
    #: Secant pairs kept per element, `1 <= m <= 5`. Five is the dimension of
    #: the reduced space, so more pairs carry no independent information.
    jacobian_correction_memory: int = 5
    #: Hourglass stiffness scale, `0 < beta <= 1`. At one, the reduced element
    #: reproduces the fully integrated one exactly in the elastic range.
    hourglass_scale: float = 1.0
    #: Warn above this hourglass-to-internal energy ratio; `None` disables it.
    hourglass_energy_warning_ratio: float | None = 0.01
    #: Fail above this ratio. `None`, the default, never fails the solve.
    hourglass_energy_failure_ratio: float | None = None
    require_pypardiso: bool = True
    hardening_mode: Literal["ludwik", "tabular"] = "ludwik"
    #: No longer a `Literal`: a registered plugin identifier must be accepted.
    #: Validity is still checked eagerly in `__post_init__`, against the builtin
    #: names plus whatever the plugin registry knows, so a typo fails when the
    #: configuration is read rather than minutes into a solve.
    constitutive_backend: str = "mfront"
    mfront_behaviour_id: str | None = None
    constitutive_options: dict[str, Any] = field(default_factory=dict)
    mfront_library: str = "build/mfront/src/libBehaviour.so"
    mfront_threads: int = 1
    local_plane_stress_tolerance_mpa: float = 1e-8
    local_plane_stress_relative_tolerance: float = 1e-10
    maximum_local_plane_stress_iterations: int = 15
    maximum_cbb_condition_number: float = 1e12
    newton_line_search: bool = False
    line_search_reduction: float = 0.5
    line_search_armijo_coefficient: float = 1e-4
    line_search_minimum_factor: float = 2.0**-12
    line_search_maximum_trials: int = 12
    boundary_history_predictor: Literal[
        "elastic",
        "secant-corrected-elastic",
    ] = "elastic"

    def __post_init__(self) -> None:
        if self.increments < 1:
            raise ValueError("increments must be positive")
        if self.max_newton_iterations < 1:
            raise ValueError("max_newton_iterations must be positive")
        if not 0 < self.residual_tolerance < 1:
            raise ValueError("residual_tolerance must lie in (0, 1)")
        if self.minimum_step_divisor < 2:
            raise ValueError("minimum_step_divisor must be at least 2")
        if self.hardening_mode not in {"ludwik", "tabular"}:
            raise ValueError("hardening_mode must be 'ludwik' or 'tabular'")
        if self.element_formulation not in ("cps4", "cps4r", "cps4r_as"):
            raise ValueError(
                f"unsupported element_formulation {self.element_formulation!r}; "
                "available: cps4, cps4r, cps4r_as"
            )
        if self.element_formulation != "cps4r_as":
            # Accepting an assumed-strain setting on a formulation that ignores
            # it would let a reader believe the run used it.
            if self.stabilisation_projection != "asmd":
                raise ValueError(
                    "stabilisation_projection only applies to the cps4r_as "
                    f"formulation, not to {self.element_formulation!r}"
                )
            if self.jacobian_correction != "none":
                raise ValueError(
                    "jacobian_correction repairs the assumed-strain hourglass "
                    "tangent and only applies to the cps4r_as formulation, not "
                    f"to {self.element_formulation!r}"
                )
        elif not 0.0 < self.stabilisation_tangent_floor < 1.0:
            raise ValueError("stabilisation_tangent_floor must lie in (0, 1)")
        if self.jacobian_correction not in ("none", "broyden"):
            raise ValueError(
                f"unknown jacobian_correction {self.jacobian_correction!r}; "
                "available: none, broyden"
            )
        if not 1 <= self.jacobian_correction_memory <= 5:
            raise ValueError(
                "jacobian_correction_memory must lie in 1..5, got "
                f"{self.jacobian_correction_memory}"
            )
        if self.element_formulation == "cps4":
            # Accepting a hourglass setting that does nothing would let a reader
            # believe the fully integrated element was being stabilised.
            if self.hourglass_scale != 1.0:
                raise ValueError(
                    "hourglass_scale has no meaning for the fully integrated cps4 "
                    "formulation; it is a cps4r stabilisation parameter"
                )
        elif not 0.0 < self.hourglass_scale <= 1.0:
            raise ValueError(
                f"hourglass_scale must satisfy 0 < beta <= 1, got {self.hourglass_scale}"
            )
        for name in ("hourglass_energy_warning_ratio", "hourglass_energy_failure_ratio"):
            ratio = getattr(self, name)
            if ratio is not None and not ratio > 0.0:
                raise ValueError(f"{name} must be positive when set")
        if not self.constitutive_backend:
            raise ValueError("constitutive_backend must not be empty")
        if self.constitutive_backend not in known_constitutive_backends():
            known = ", ".join(sorted(known_constitutive_backends()))
            raise ValueError(
                f"unsupported constitutive_backend {self.constitutive_backend!r}; "
                f"known: {known}"
            )
        if self.mfront_behaviour_id is not None and not self.mfront_behaviour_id:
            raise ValueError("mfront_behaviour_id must not be empty")
        if not all(isinstance(key, str) for key in self.constitutive_options):
            raise TypeError("constitutive_options keys must be strings")
        if not self.mfront_library:
            raise ValueError("mfront_library must not be empty")
        if not 0.0 < self.line_search_reduction < 1.0:
            raise ValueError("line_search_reduction must lie in (0, 1)")
        if not 0.0 < self.line_search_armijo_coefficient < 1.0:
            raise ValueError("line_search_armijo_coefficient must lie in (0, 1)")
        if not 0.0 < self.line_search_minimum_factor <= 1.0:
            raise ValueError("line_search_minimum_factor must lie in (0, 1]")
        if self.line_search_maximum_trials < 1:
            raise ValueError("line_search_maximum_trials must be positive")
        if self.boundary_history_predictor not in {
            "elastic",
            "secant-corrected-elastic",
        }:
            raise ValueError(
                "boundary_history_predictor must be 'elastic' or "
                "'secant-corrected-elastic'"
            )
        if isinstance(self.mfront_threads, bool) or not isinstance(self.mfront_threads, int):
            raise TypeError("mfront_threads must be an integer")
        if self.mfront_threads < 1:
            raise ValueError("mfront_threads must be at least 1")
        if self.local_plane_stress_tolerance_mpa <= 0:
            raise ValueError("local_plane_stress_tolerance_mpa must be positive")
        if self.local_plane_stress_relative_tolerance <= 0:
            raise ValueError("local_plane_stress_relative_tolerance must be positive")
        if self.maximum_local_plane_stress_iterations < 1:
            raise ValueError("maximum_local_plane_stress_iterations must be positive")
        if self.maximum_cbb_condition_number <= 1:
            raise ValueError("maximum_cbb_condition_number must be greater than one")


@dataclass(frozen=True, slots=True)
class NonlocalPlasticityConfig:
    """Micromorphic J2 coupling controls.

    The length is expressed in millimetres, the coupling modulus in MPa, and
    ``relative_tolerance`` controls the staggered ``p``--``chi`` fixed point.
    """

    enabled: bool = False
    length_scale_mm: float = 0.05888
    coupling_modulus_mpa: float = 0.0
    relaxation: float = 0.5
    relaxation_strategy: Literal["fixed", "aitken"] = "fixed"
    minimum_relaxation: float = 0.05
    maximum_relaxation: float = 0.8
    aitken_residual_growth_factor: float = 1.25
    relative_tolerance: float = 1e-6
    maximum_iterations: int = 15
    maximum_helmholtz_residual: float = 1e-10
    record_iteration_history: bool = False
    #: Which registered criterion drives the fixed point. The default is the
    #: historical PEEQ-Helmholtz coupling, so existing configurations are
    #: unchanged.
    criterion: str = "peeq_helmholtz"
    criterion_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.length_scale_mm <= 0:
            raise ValueError("length_scale_mm must be positive")
        if not self.criterion:
            raise ValueError("nonlocal criterion must not be empty")
        if not all(isinstance(key, str) for key in self.criterion_options):
            raise TypeError("criterion_options keys must be strings")
        if self.coupling_modulus_mpa < 0:
            raise ValueError("coupling_modulus_mpa must be nonnegative")
        if not 0 < self.relaxation <= 1:
            raise ValueError("relaxation must lie in (0, 1]")
        if self.relaxation_strategy not in {"fixed", "aitken"}:
            raise ValueError("relaxation_strategy must be 'fixed' or 'aitken'")
        if not 0 < self.minimum_relaxation <= self.maximum_relaxation <= 1:
            raise ValueError(
                "relaxation bounds must satisfy 0 < minimum <= maximum <= 1"
            )
        if (
            self.relaxation_strategy == "aitken"
            and not self.minimum_relaxation
            <= self.relaxation
            <= self.maximum_relaxation
        ):
            raise ValueError("relaxation must lie inside the configured bounds")
        if self.aitken_residual_growth_factor <= 1:
            raise ValueError("aitken_residual_growth_factor must be greater than one")
        if not 0 < self.relative_tolerance < 1:
            raise ValueError("relative_tolerance must lie in (0, 1)")
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")
        if self.maximum_helmholtz_residual <= 0:
            raise ValueError("maximum_helmholtz_residual must be positive")


@dataclass(frozen=True, slots=True)
class CaseStudyConfig:
    """Complete configuration for one structured case-study solve."""

    mesh: MeshConfig
    material: MaterialConfig = field(default_factory=MaterialConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    nonlocal_plasticity: NonlocalPlasticityConfig = field(
        default_factory=NonlocalPlasticityConfig
    )
