"""Post-processing functions with explicit scientific conventions."""

from fem_inhouse.postprocessing.kinematics import (
    StrainComponents,
    cell_average,
    plane_stress_equivalent_strain,
    strain_from_displacement,
)
from fem_inhouse.postprocessing.metrics import (
    FieldAcceptanceThresholds,
    FieldComparisonReport,
    FieldErrorMetrics,
    LocalizationOverlapMetrics,
    evaluate_field_comparison,
    field_error_metrics,
    interface_gradient_ratio,
    localization_overlap_metrics,
    signed_difference_field,
)
from fem_inhouse.postprocessing.stress_curves import (
    direct_fe_equivalent_stress_curve,
    reconstructed_equivalent_stress,
    reconstructed_stress_curve_from_strain,
    von_mises_stress,
)

__all__ = [
    "FieldAcceptanceThresholds",
    "FieldComparisonReport",
    "FieldErrorMetrics",
    "LocalizationOverlapMetrics",
    "StrainComponents",
    "cell_average",
    "direct_fe_equivalent_stress_curve",
    "evaluate_field_comparison",
    "field_error_metrics",
    "interface_gradient_ratio",
    "localization_overlap_metrics",
    "plane_stress_equivalent_strain",
    "reconstructed_equivalent_stress",
    "reconstructed_stress_curve_from_strain",
    "signed_difference_field",
    "strain_from_displacement",
    "von_mises_stress",
]
