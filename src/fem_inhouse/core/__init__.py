"""Finite-element kernel for the supported structured CPS4 case study."""

from fem_inhouse.core.assembly import (
    assemble_stiffness,
    assembly_indices,
    element_tangent_stiffness,
    internal_force,
)
from fem_inhouse.core.constitutive import (
    PLANE_STRESS_VON_MISES_METRIC,
    HardeningFunction,
    HardeningMode,
    consistent_tangent,
    make_hardening,
    return_mapping,
    von_mises,
)
from fem_inhouse.core.element import (
    ElementOperators,
    plane_stress_elasticity,
    precompute_element,
    shape_function_derivatives,
    strain_displacement_matrix,
)
from fem_inhouse.core.mesh import StructuredMesh
from fem_inhouse.core.mfront import (
    MFront3DCondensedPlaneStressBatch,
    MFrontIntegrationError,
    MFrontIntegrationResult,
    MFrontMaterialPointBatch,
    MFrontNativePlaneStressBatch,
    MFrontUnavailableError,
    engineering_strain_to_kelvin,
    kelvin_strain_to_engineering,
    kelvin_stress_to_engineering,
    kelvin_tangent_to_engineering,
)
from fem_inhouse.core.plane_stress_material import (
    ConstitutiveIntegrationError,
    ConstitutiveTrial,
    LocalPlaneStressConvergenceError,
    PlaneStressBatchStatistics,
    PlaneStressMaterialBatch,
    PythonJ2PlaneStressBatch,
)

__all__ = [
    "PLANE_STRESS_VON_MISES_METRIC",
    "ConstitutiveIntegrationError",
    "ConstitutiveTrial",
    "ElementOperators",
    "HardeningFunction",
    "HardeningMode",
    "LocalPlaneStressConvergenceError",
    "MFront3DCondensedPlaneStressBatch",
    "MFrontIntegrationError",
    "MFrontIntegrationResult",
    "MFrontMaterialPointBatch",
    "MFrontNativePlaneStressBatch",
    "MFrontUnavailableError",
    "PlaneStressBatchStatistics",
    "PlaneStressMaterialBatch",
    "PythonJ2PlaneStressBatch",
    "StructuredMesh",
    "assemble_stiffness",
    "assembly_indices",
    "consistent_tangent",
    "element_tangent_stiffness",
    "engineering_strain_to_kelvin",
    "internal_force",
    "kelvin_strain_to_engineering",
    "kelvin_stress_to_engineering",
    "kelvin_tangent_to_engineering",
    "make_hardening",
    "plane_stress_elasticity",
    "precompute_element",
    "return_mapping",
    "shape_function_derivatives",
    "strain_displacement_matrix",
    "von_mises",
]
