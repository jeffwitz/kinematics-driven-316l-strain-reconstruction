"""Object-based comparison tools for observed EVM fields.

These modules answer "which candidate reproduces the observed bands", which a
global error cannot. They operate on fields only: nothing here runs mechanics,
selects a material parameter, or touches an archived result.

Geometry is defined from the DIC field alone and then frozen, so every
candidate is measured against the same objects.
"""

from fem_inhouse.validation.band_geometry import (
    BandObject,
    band_corridor,
    label_band_objects,
    order_centreline,
    prune_skeleton_spurs,
    quantile_thresholds,
    resample_polyline,
    smooth_centreline,
    tangents_and_normals,
    zhang_suen_thinning,
)
from fem_inhouse.validation.band_profiles import (
    BackgroundEstimate,
    NormalProfile,
    WidthMeasurement,
    WidthStatus,
    compare_profiles,
    continuity_metrics,
    estimate_background,
    excess_profile,
    measure_amplitude,
    measure_position,
    measure_width,
    sample_normal_profile,
    summarise,
)
from fem_inhouse.validation.falsification_cases import (
    PerturbedField,
    add_spurious_band,
    change_band_width,
    interrupt_region,
    remove_region,
    scale_amplitude,
    standard_cases,
    translate_field,
)
from fem_inhouse.validation.fractions_skill_score import (
    SkillCurve,
    active_fraction_field,
    fractions_skill_score,
    minimum_skilful_scale,
    skill_curve,
    skill_table,
)
from fem_inhouse.validation.residual_structure import (
    EnergyPartition,
    classify_residual,
    directional_variogram,
    energy_partition,
    radial_power_spectrum,
    residual_associations,
    signed_residual,
)

__all__ = [
    "BackgroundEstimate",
    "BandObject",
    "EnergyPartition",
    "NormalProfile",
    "PerturbedField",
    "SkillCurve",
    "WidthMeasurement",
    "WidthStatus",
    "active_fraction_field",
    "add_spurious_band",
    "band_corridor",
    "change_band_width",
    "classify_residual",
    "compare_profiles",
    "continuity_metrics",
    "directional_variogram",
    "energy_partition",
    "estimate_background",
    "excess_profile",
    "fractions_skill_score",
    "interrupt_region",
    "label_band_objects",
    "measure_amplitude",
    "measure_position",
    "measure_width",
    "minimum_skilful_scale",
    "order_centreline",
    "prune_skeleton_spurs",
    "quantile_thresholds",
    "radial_power_spectrum",
    "remove_region",
    "resample_polyline",
    "residual_associations",
    "sample_normal_profile",
    "scale_amplitude",
    "signed_residual",
    "skill_curve",
    "skill_table",
    "smooth_centreline",
    "standard_cases",
    "summarise",
    "tangents_and_normals",
    "translate_field",
    "zhang_suen_thinning",
]
