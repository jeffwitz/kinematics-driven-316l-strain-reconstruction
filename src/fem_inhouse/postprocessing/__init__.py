"""Post-processing functions with explicit scientific conventions."""

from fem_inhouse.postprocessing.helmholtz import (
    HelmholtzFilterResult,
    helmholtz_filter_element_field,
)
from fem_inhouse.postprocessing.kinematics import (
    StrainComponents,
    cell_average,
    plane_stress_equivalent_strain,
    strain_from_displacement,
)
from fem_inhouse.postprocessing.metrics import (
    AbsoluteThresholdOverlapMetrics,
    FieldAcceptanceThresholds,
    FieldComparisonReport,
    FieldDiffusivityMetrics,
    FieldErrorMetrics,
    LocalizationOverlapMetrics,
    absolute_threshold_overlap_metrics,
    evaluate_field_comparison,
    field_diffusivity_metrics,
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
from fem_inhouse.postprocessing.tensor_measures import (
    instantaneous_equivalent_plastic_strain,
    reconstructed_equivalent_strain,
    von_mises_from_stress_tensor,
)

__all__ = [
    "AbsoluteThresholdOverlapMetrics",
    "FieldAcceptanceThresholds",
    "FieldComparisonReport",
    "FieldDiffusivityMetrics",
    "FieldErrorMetrics",
    "HelmholtzFilterResult",
    "LocalizationOverlapMetrics",
    "StrainComponents",
    "absolute_threshold_overlap_metrics",
    "cell_average",
    "direct_fe_equivalent_stress_curve",
    "evaluate_field_comparison",
    "field_diffusivity_metrics",
    "field_error_metrics",
    "helmholtz_filter_element_field",
    "instantaneous_equivalent_plastic_strain",
    "interface_gradient_ratio",
    "localization_overlap_metrics",
    "plane_stress_equivalent_strain",
    "reconstructed_equivalent_strain",
    "reconstructed_equivalent_stress",
    "reconstructed_stress_curve_from_strain",
    "signed_difference_field",
    "strain_from_displacement",
    "von_mises_from_stress_tensor",
    "von_mises_stress",
]
