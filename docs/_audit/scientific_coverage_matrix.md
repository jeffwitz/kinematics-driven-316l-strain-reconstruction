# Scientific coverage matrix

This matrix is backed by [`scientific_coverage.yml`](scientific_coverage.yml).
`complete` means that the subject has a reviewed page in every applicable
quadrant and a route from the corresponding portal. Scientific qualification
status remains governed by the evidence registry and validation artefacts.

| Subject | Tutorial | How-to | Reference | Explanation | Evidence | Status |
|---|---|---|---|---|---|---|
| DIC and observation | — | `how-to/data/prepare_dic_case` | `reference/observation_operator` | `explanation/measurement/dic_observation_limits` | `reference/evidence_registry` | routed |
| EBSD and registration | — | `how-to/data/inspect_ebsd_registration` | `reference/ebsd_orientation_contract` | `explanation/measurement/dic_observation_limits` | `reference/evidence_registry` | routed |
| J2/Ludwik baseline | — | `how-to/mechanics/run_local_reconstruction` | `reference/constitutive_models` | `explanation/constitutive/ludwik_j2` | `explanation/reconstruction/local_baseline` | routed |
| SRIX | — | `how-to/crystal-plasticity/run_316l_crystal_plasticity` | `reference/numerics/native_srix_backend` | `explanation/constitutive/forest_rubin_srix` | `reference/evidence_registry` | routed |
| Méric–Cailletaud | — | `how-to/reproduce/reproduce_srix_meric_comparison` | `reference/constitutive_models` | `explanation/constitutive/meric_cailletaud` | `explanation/constitutive/srix_vs_meric` | routed |
| Structural plane stress | — | `how-to/crystal-plasticity/qualify_native_srix_backend` | `reference/numerics/plane_stress` | `explanation/constitutive/structural_plane_stress` | `reference/evidence_registry` | routed |
| Spectral solver / FFTW | `tutorials/first_full_dirichlet_spectral_reconstruction` | `how-to/mechanics/run_full_dirichlet_spectral` | `reference/numerics/newton_gmres_contract` | `explanation/spectral_mechanics/index` | `reference/evidence_registry` | routed |
| TET2 / EBI-TET | — | `how-to/reproduce/reproduce_ebi_falsification` | `reference/numerics/ebi_tet_contract` | `explanation/spectral_mechanics/ebi_srix_falsification` | `explanation/evidence/negative_results` | complete |
| Native SRIX / Numba | — | `how-to/crystal-plasticity/qualify_native_srix_backend` | `reference/numerics/native_srix_backend` | `explanation/native-srix/optimization_strategy` | `reference/evidence_registry` | routed |
| FEMU and SVD | — | `how-to/identification/run_identification` | `reference/selection_indicators` | `explanation/identification/femu_identification` | `explanation/identification/identifiability` | routed |
| REGM | — | `how-to/identification/run_regm_screening` | `reference/claims_matrix` | `explanation/identification/regm_screening` | `explanation/evidence/negative_results` | routed |
| Reduced integration | — | `how-to/mechanics/use_reduced_integration` | `reference/constitutive_models` | `explanation/constitutive/reduced_integration_hourglass` | `explanation/evidence/negative_results` | routed |
| Nonlocal / micromorphic | — | `how-to/identification/run_micromorphic_identification` | `reference/scientific/nonlocal_parameters` | `explanation/reconstruction/nonlocal_micromorphic` | `explanation/evidence/current_scientific_status` | complete |
