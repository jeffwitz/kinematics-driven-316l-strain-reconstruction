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
from fem_inhouse.postprocessing.section_equilibrium import (
    SectionEquilibriumResult,
    integrated_section_equilibrium,
)
from fem_inhouse.postprocessing.spatial_correlation import (
    CorrelationProfile,
    DecayFit,
    StructuralCorrelationResult,
    correlation_profiles,
    fit_exponential_decay,
    mask_corrected_autocorrelation,
    rms_positive_correlation_radius,
    structural_correlation,
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
    "CorrelationProfile",
    "DecayFit",
    "FieldAcceptanceThresholds",
    "FieldComparisonReport",
    "FieldDiffusivityMetrics",
    "FieldErrorMetrics",
    "HelmholtzFilterResult",
    "LocalizationOverlapMetrics",
    "SectionEquilibriumResult",
    "StrainComponents",
    "StructuralCorrelationResult",
    "absolute_threshold_overlap_metrics",
    "cell_average",
    "correlation_profiles",
    "direct_fe_equivalent_stress_curve",
    "evaluate_field_comparison",
    "field_diffusivity_metrics",
    "field_error_metrics",
    "fit_exponential_decay",
    "helmholtz_filter_element_field",
    "instantaneous_equivalent_plastic_strain",
    "integrated_section_equilibrium",
    "interface_gradient_ratio",
    "localization_overlap_metrics",
    "mask_corrected_autocorrelation",
    "plane_stress_equivalent_strain",
    "reconstructed_equivalent_strain",
    "reconstructed_equivalent_stress",
    "reconstructed_stress_curve_from_strain",
    "rms_positive_correlation_radius",
    "signed_difference_field",
    "strain_from_displacement",
    "structural_correlation",
    "von_mises_from_stress_tensor",
    "von_mises_stress",
]
