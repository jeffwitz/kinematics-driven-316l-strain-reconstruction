"""Declarative catalogue of MFront behaviours and their solver contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fem_inhouse.core.linear_solver import LinearSystemMatrixType

PlaneStressStrategy = Literal["native", "condensed_3d", "structural_plane_stress"]
VariableKind = Literal["scalar", "vector", "symmetric_tensor", "tensor"]


@dataclass(frozen=True, slots=True)
class MFrontVariableSpec:
    """One canonical field bound to an MFront entry name."""

    canonical_name: str
    entry_name: str
    kind: VariableKind = "scalar"
    required: bool = True
    #: Members of the family, in the order MFront lays them out.
    #:
    #: One for a scalar, six for a symmetric tensor, and twelve for a
    #: per-slip-system quantity of an FCC crystal. The bridge reads consecutive
    #: slots from the declared offset rather than parsing entry names like
    #: "PlasticSlip[7]", which would break on any behaviour that names its
    #: members differently.
    component_count: int = 1

    def __post_init__(self) -> None:
        if not self.canonical_name or not self.entry_name:
            raise ValueError("MFront variable names must not be empty")
        if self.component_count < 1:
            raise ValueError("component_count must be at least one")


@dataclass(frozen=True, slots=True)
class MFrontBehaviourSpec:
    """Capabilities and canonical field bindings of one constitutive law."""

    identifier: str
    native_plane_stress_behaviour: str | None
    tridimensional_behaviour: str | None
    material_properties: tuple[MFrontVariableSpec, ...]
    structural_plane_stress_behaviour: str | None = None
    external_state_variables: tuple[MFrontVariableSpec, ...] = ()
    internal_state_variables: tuple[MFrontVariableSpec, ...] = ()
    linear_system_matrix_type: LinearSystemMatrixType = "nonsymmetric"
    requires_rotation_matrix: bool = False
    bridge_profile: str = "custom"
    #: Registry of selectable parameter sets, or `None` when the law exposes no
    #: configurable set. Two crystal laws share the FCC bridge but not their
    #: flow parameters -- SRIX has `R`, Meric-Cailletaud has `(K, n)` -- so the
    #: bridge cannot assume that one law's parameter names exist on the other.
    parameter_registry: str | None = None
    paired_material_family: str | None = None
    crystal_flow_rule: Literal["meric_cailletaud", "forest_rubin_srix"] | None = None

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
        if strategy == "native":
            value = self.native_plane_stress_behaviour
        elif strategy == "condensed_3d":
            value = self.tridimensional_behaviour
        else:
            value = self.structural_plane_stress_behaviour
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

# FCC single crystals. Both laws share their slip systems, interaction matrix
# and hardening; they differ only in the flow rule, so their bindings have the
# same shape and differ only in the entry names MFront generated.
#
# Neither exposes EquivalentPlasticStrain or YieldSurfaceRadius: a crystal has
# twelve slips and twelve critical resolved shear stresses, not one scalar pair.
# That is why the 3D bridge reads its internal-state bindings from this
# catalogue instead of assuming the J2 triple.
#
# The consistent tangent of a single crystal is NOT symmetric, unlike the J2
# radial return: measured at 4.0e-05 median relative asymmetry against 1.4e-16,
# so these behaviours must drive the linear solver in its nonsymmetric mode.


#: Octahedral slip systems of an FCC crystal.
FCC_SLIP_SYSTEM_COUNT = 12


def _fcc_internal(slip_entry: str, equivalent_entry: str) -> tuple[MFrontVariableSpec, ...]:
    per_system = FCC_SLIP_SYSTEM_COUNT
    return (
        MFrontVariableSpec(
            "elastic_strain", "ElasticStrain", "symmetric_tensor", component_count=6
        ),
        MFrontVariableSpec("plastic_slip", slip_entry, "scalar", component_count=per_system),
        MFrontVariableSpec(
            "equivalent_plastic_slip", equivalent_entry, "scalar", component_count=per_system
        ),
        MFrontVariableSpec("back_strain", "BackStrain", "scalar", component_count=per_system),
    )


MFRONT_BEHAVIOURS.register(
    MFrontBehaviourSpec(
        identifier="fcc_meric_cailletaud",
        native_plane_stress_behaviour=None,
        tridimensional_behaviour="Fcc316LMericCailletaud",
        material_properties=(),
        structural_plane_stress_behaviour="Fcc316LMericCailletaudStructuralPlaneStress",
        internal_state_variables=_fcc_internal(
            "ViscoplasticSlip", "EquivalentViscoplasticSlip"
        ),
        linear_system_matrix_type="nonsymmetric",
        requires_rotation_matrix=True,
        bridge_profile="fcc_single_crystal_v1",
        paired_material_family="fcc_316l_guilhem_nasri_v1",
        crystal_flow_rule="meric_cailletaud",
    )
)
MFRONT_BEHAVIOURS.register(
    MFrontBehaviourSpec(
        identifier="fcc_forest_rubin_srix",
        native_plane_stress_behaviour=None,
        tridimensional_behaviour="Fcc316LForestRubinSrix",
        material_properties=(
            MFrontVariableSpec("coupling_modulus_mpa", "MicromorphicCouplingModulus"),
        ),
        external_state_variables=(
            MFrontVariableSpec(
                "nonlocal_equivalent_plastic_strain",
                "NonlocalEquivalentPlasticStrain",
            ),
        ),
        structural_plane_stress_behaviour="Fcc316LForestRubinSrixStructuralPlaneStress",
        internal_state_variables=_fcc_internal("PlasticSlip", "EquivalentPlasticSlip"),
        linear_system_matrix_type="nonsymmetric",
        requires_rotation_matrix=True,
        bridge_profile="fcc_single_crystal_v1",
        paired_material_family="fcc_316l_guilhem_nasri_v1",
        crystal_flow_rule="forest_rubin_srix",
        parameter_registry="srix",
    )
)
# Generalised plane stress inside the local Newton (UMAT closure): the law
# receives the nine rotation components as per-point material properties and
# carries the closure on the global transverse stresses itself. Qualified by
# validation/srix_umat_gps_closure_preregistration.md; not a production
# backend until that qualification passes.
MFRONT_BEHAVIOURS.register(
    MFrontBehaviourSpec(
        identifier="fcc_forest_rubin_srix_gps",
        native_plane_stress_behaviour=None,
        tridimensional_behaviour="Fcc316LForestRubinSrixGps",
        material_properties=(),
        internal_state_variables=_fcc_internal("PlasticSlip", "EquivalentPlasticSlip"),
        linear_system_matrix_type="nonsymmetric",
        requires_rotation_matrix=True,
        bridge_profile="fcc_single_crystal_v1",
        paired_material_family="fcc_316l_guilhem_nasri_v1",
        crystal_flow_rule="forest_rubin_srix",
        parameter_registry="srix",
    )
)
MFRONT_BEHAVIOURS.register(
    MFrontBehaviourSpec(
        identifier="fcc_forest_rubin_srix_structural_plane_stress",
        native_plane_stress_behaviour=None,
        tridimensional_behaviour="Fcc316LForestRubinSrixStructuralPlaneStress",
        material_properties=(),
        structural_plane_stress_behaviour="Fcc316LForestRubinSrixStructuralPlaneStress",
        internal_state_variables=_fcc_internal("PlasticSlip", "EquivalentPlasticSlip"),
        linear_system_matrix_type="nonsymmetric",
        requires_rotation_matrix=True,
        bridge_profile="fcc_single_crystal_v1",
        paired_material_family="fcc_316l_guilhem_nasri_v1",
        crystal_flow_rule="forest_rubin_srix",
        parameter_registry="srix",
    )
)
MFRONT_BEHAVIOURS.register(
    MFrontBehaviourSpec(
        identifier="fcc_meric_cailletaud_structural_plane_stress",
        native_plane_stress_behaviour=None,
        tridimensional_behaviour="Fcc316LMericCailletaudStructuralPlaneStress",
        material_properties=(),
        structural_plane_stress_behaviour="Fcc316LMericCailletaudStructuralPlaneStress",
        internal_state_variables=_fcc_internal(
            "ViscoplasticSlip", "EquivalentViscoplasticSlip"
        ),
        linear_system_matrix_type="nonsymmetric",
        requires_rotation_matrix=True,
        bridge_profile="fcc_single_crystal_v1",
        paired_material_family="fcc_316l_guilhem_nasri_v1",
        crystal_flow_rule="meric_cailletaud",
    )
)
