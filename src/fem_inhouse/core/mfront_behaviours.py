"""Declarative catalogue of MFront behaviours and their solver contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fem_inhouse.core.linear_solver import LinearSystemMatrixType

PlaneStressStrategy = Literal["native", "condensed_3d"]
VariableKind = Literal["scalar", "vector", "symmetric_tensor", "tensor"]


@dataclass(frozen=True, slots=True)
class MFrontVariableSpec:
    """One canonical field bound to an MFront entry name."""

    canonical_name: str
    entry_name: str
    kind: VariableKind = "scalar"
    required: bool = True

    def __post_init__(self) -> None:
        if not self.canonical_name or not self.entry_name:
            raise ValueError("MFront variable names must not be empty")


@dataclass(frozen=True, slots=True)
class MFrontBehaviourSpec:
    """Capabilities and canonical field bindings of one constitutive law."""

    identifier: str
    native_plane_stress_behaviour: str | None
    tridimensional_behaviour: str | None
    material_properties: tuple[MFrontVariableSpec, ...]
    external_state_variables: tuple[MFrontVariableSpec, ...] = ()
    internal_state_variables: tuple[MFrontVariableSpec, ...] = ()
    linear_system_matrix_type: LinearSystemMatrixType = "nonsymmetric"
    requires_rotation_matrix: bool = False
    bridge_profile: str = "custom"

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("MFront behaviour identifier must not be empty")
        if self.native_plane_stress_behaviour is None and self.tridimensional_behaviour is None:
            raise ValueError("at least one MFront modelling hypothesis must be provided")
        groups = (
            self.material_properties,
            self.external_state_variables,
            self.internal_state_variables,
        )
        for variables in groups:
            canonical = [item.canonical_name for item in variables]
            entries = [item.entry_name for item in variables]
            if len(canonical) != len(set(canonical)) or len(entries) != len(set(entries)):
                raise ValueError("MFront variable bindings must be unique inside each group")

    def behaviour_name(self, strategy: PlaneStressStrategy) -> str:
        value = (
            self.native_plane_stress_behaviour
            if strategy == "native"
            else self.tridimensional_behaviour
        )
        if value is None:
            raise ValueError(
                f"MFront behaviour {self.identifier!r} does not support {strategy!r}"
            )
        return value

    def external_entry_name(self, canonical_name: str) -> str:
        for variable in self.external_state_variables:
            if variable.canonical_name == canonical_name:
                return variable.entry_name
        raise KeyError(
            f"MFront behaviour {self.identifier!r} has no external field "
            f"{canonical_name!r}"
        )


class MFrontBehaviourRegistry:
    """Catalogue used for discovery and fail-fast compatibility checks."""

    def __init__(self) -> None:
        self._specifications: dict[str, MFrontBehaviourSpec] = {}

    def register(self, specification: MFrontBehaviourSpec, *, replace: bool = False) -> None:
        identifier = specification.identifier
        if identifier in self._specifications and not replace:
            raise ValueError(f"MFront behaviour {identifier!r} is already registered")
        self._specifications[identifier] = specification

    def get(self, identifier: str) -> MFrontBehaviourSpec:
        try:
            return self._specifications[identifier]
        except KeyError as error:
            available = ", ".join(self.identifiers()) or "none"
            raise KeyError(
                f"unknown MFront behaviour {identifier!r}; available: {available}"
            ) from error

    def identifiers(self) -> tuple[str, ...]:
        return tuple(sorted(self._specifications))


MFRONT_BEHAVIOURS = MFrontBehaviourRegistry()

_J2_PROPERTIES = (
    MFrontVariableSpec("initial_yield_stress_mpa", "InitialYieldStress"),
    MFrontVariableSpec("hardening_coefficient_mpa", "HardeningCoefficient"),
    MFrontVariableSpec("hardening_exponent", "HardeningExponent"),
)
_J2_INTERNAL = (
    MFrontVariableSpec("elastic_strain", "ElasticStrain", "symmetric_tensor"),
    MFrontVariableSpec("equivalent_plastic_strain", "EquivalentPlasticStrain"),
    MFrontVariableSpec("yield_surface_radius_mpa", "YieldSurfaceRadius"),
)

MFRONT_BEHAVIOURS.register(
    MFrontBehaviourSpec(
        identifier="ludwik_j2",
        native_plane_stress_behaviour="PixelLudwikJ2Plasticity",
        tridimensional_behaviour="PixelLudwikJ2Plasticity3D",
        material_properties=_J2_PROPERTIES,
        internal_state_variables=_J2_INTERNAL,
        linear_system_matrix_type="symmetric_positive_definite",
        bridge_profile="ludwik_j2_v1",
    )
)
MFRONT_BEHAVIOURS.register(
    MFrontBehaviourSpec(
        identifier="micromorphic_ludwik_j2",
        native_plane_stress_behaviour="PixelMicromorphicLudwikJ2Plasticity",
        tridimensional_behaviour="PixelMicromorphicLudwikJ2Plasticity3D",
        material_properties=(
            *_J2_PROPERTIES,
            MFrontVariableSpec("coupling_modulus_mpa", "MicromorphicCouplingModulus"),
        ),
        external_state_variables=(
            MFrontVariableSpec(
                "nonlocal_equivalent_plastic_strain",
                "NonlocalEquivalentPlasticStrain",
            ),
        ),
        internal_state_variables=_J2_INTERNAL,
        linear_system_matrix_type="symmetric_positive_definite",
        bridge_profile="ludwik_j2_v1",
    )
)
