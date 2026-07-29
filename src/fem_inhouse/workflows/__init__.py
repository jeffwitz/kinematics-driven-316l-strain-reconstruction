"""Reproducible workflows for the supported case study."""

from fem_inhouse.workflows.coupled_alpha_visualization import (
    CoupledAlphaVisualizationData,
    common_color_limits,
    plot_coupled_alpha_fields,
    prepare_coupled_alpha_fields,
    symmetric_color_limit,
)
from fem_inhouse.workflows.coupled_nonlocal_validation import (
    CoupledValidationThresholds,
    validate_coupled_nonlocal_campaign,
)
from fem_inhouse.workflows.dic_measurement_chain import characterise_dic_measurement_chain
from fem_inhouse.workflows.dic_observation_replay import replay_dic_observation
from fem_inhouse.workflows.dic_partition_selection import (
    scan_dic_partition_heterogeneity,
    write_dic_partition_heterogeneity_report,
)
from fem_inhouse.workflows.dic_photometric_quality import (
    diagnose_dic_photometric_quality,
)
from fem_inhouse.workflows.dic_uncertainty_propagation import (
    contiguous_residual_on_support,
    periodic_residual_on_support,
    propagate_dic_uncertainty,
)
from fem_inhouse.workflows.ebsd_structural_length import measure_ebsd_structural_length
from fem_inhouse.workflows.joint_nonlocal_identification import (
    analyze_joint_identifiability,
    collect_identification_results,
    generate_high_fidelity_manifest,
    generate_joint_identification_report,
    inspect_joint_identification,
    load_joint_identification_config,
    prepare_transfer_validation,
    profile_coupling_modulus,
    run_low_fidelity,
    screen_frozen_field,
    select_identification_candidates,
    validate_low_fidelity_ranking,
)
from fem_inhouse.workflows.material_map_control_validation import (
    validate_material_map_controls,
)
from fem_inhouse.workflows.material_map_controls import prepare_material_map_control
from fem_inhouse.workflows.nonlocal_coupling_campaign import (
    ReferenceHardeningReport,
    compute_reference_hardening_modulus,
    estimate_reference_hardening_from_campaign,
)
from fem_inhouse.workflows.nonlocality_diagnostic import (
    DecisionThresholds,
    LengthScale,
    NonlocalitySweep,
    load_decision_thresholds,
    normalize_length_scales,
    reconstruct_historical_evm,
    run_field_sweep,
    run_nonlocality_diagnostic,
)
from fem_inhouse.workflows.partitioned import PartitionWorkflow, fingerprint_array
from fem_inhouse.workflows.section_equilibrium import diagnose_section_equilibrium_campaigns

__all__ = [
    "CoupledAlphaVisualizationData",
    "CoupledValidationThresholds",
    "DecisionThresholds",
    "LengthScale",
    "NonlocalitySweep",
    "PartitionWorkflow",
    "ReferenceHardeningReport",
    "analyze_joint_identifiability",
    "characterise_dic_measurement_chain",
    "collect_identification_results",
    "common_color_limits",
    "compute_reference_hardening_modulus",
    "contiguous_residual_on_support",
    "diagnose_dic_photometric_quality",
    "diagnose_section_equilibrium_campaigns",
    "estimate_reference_hardening_from_campaign",
    "fingerprint_array",
    "generate_high_fidelity_manifest",
    "generate_joint_identification_report",
    "inspect_joint_identification",
    "load_decision_thresholds",
    "load_joint_identification_config",
    "measure_ebsd_structural_length",
    "normalize_length_scales",
    "periodic_residual_on_support",
    "plot_coupled_alpha_fields",
    "prepare_coupled_alpha_fields",
    "prepare_material_map_control",
    "prepare_transfer_validation",
    "profile_coupling_modulus",
    "propagate_dic_uncertainty",
    "reconstruct_historical_evm",
    "replay_dic_observation",
    "run_field_sweep",
    "run_low_fidelity",
    "run_nonlocality_diagnostic",
    "scan_dic_partition_heterogeneity",
    "screen_frozen_field",
    "select_identification_candidates",
    "symmetric_color_limit",
    "validate_coupled_nonlocal_campaign",
    "validate_low_fidelity_ranking",
    "validate_material_map_controls",
    "write_dic_partition_heterogeneity_report",
]
