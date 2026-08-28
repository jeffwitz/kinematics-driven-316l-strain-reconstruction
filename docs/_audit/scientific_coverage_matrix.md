# Scientific coverage matrix

This matrix is backed by [`scientific_coverage.yml`](scientific_coverage.yml).
`routing_status` reports menu coverage. `content_status` reports semantic
review state, and `scientific_status` reports what the evidence allows us to
claim. These dimensions are intentionally independent.

The scientific column is a summary only. Claim-level decisions are recorded in
each subject's `claim_statuses` mapping in the YAML: for example, TET2's
two-history operator is supported while EBI-TET state sharing is negative for
the registered case; J2 table equivalence is verified while experimental
morphology adequacy is negative.

| Subject | Routing | Content | Scientific status | Tutorial | How-to | Reference | Explanation | Evidence | Blocker |
|---|---|---|---|---|---|---|---|---|---|
| DIC and observation | complete | partial | supported | — | `how-to/data/prepare_dic_case` | `reference/scientific/observation_operator` | `explanation/measurement/dic_observation_limits` | `reference/evidence/evidence_registry` | content review required |
| EBSD and registration | complete | partial | supported | — | `how-to/data/inspect_ebsd_registration` | `reference/scientific/ebsd_orientation_contract` | `explanation/measurement/ebsd_registration_and_orientation` | `reference/evidence/evidence_registry` | content review required; registration evidence provenance-limited |
| J2/Ludwik baseline | complete | partial | verified | — | `how-to/mechanics/run_local_reconstruction` | `reference/scientific/constitutive_models` | `explanation/constitutive/ludwik_j2` | `explanation/reconstruction/local_baseline` | content review required; baseline only |
| SRIX | complete | partial | supported | — | `how-to/crystal-plasticity/run_316l_crystal_plasticity` | `reference/numerics/native_srix_backend` | `explanation/constitutive/forest_rubin_srix` | `reference/evidence/srix_qualification` | evidence review |
| Méric–Cailletaud | complete | partial | supported | — | `how-to/reproduce/reproduce_srix_meric_comparison` | `reference/scientific/meric_cailletaud` | `explanation/constitutive/meric_cailletaud` | `explanation/constitutive/srix_vs_meric` | distinguish rate and solver effects |
| Structural plane stress | complete | partial | verified | — | `how-to/crystal-plasticity/qualify_native_srix_backend` | `reference/numerics/plane_stress` | `explanation/constitutive/structural_plane_stress` | `reference/evidence/evidence_registry` | keep standard 2-D MFront distinct |
| Spectral solver / FFTW | complete | partial | supported | `tutorials/first_full_dirichlet_spectral_reconstruction` | `how-to/mechanics/run_full_dirichlet_spectral` | `reference/numerics/spectral_solver` | `explanation/spectral_mechanics/index` | `reference/evidence/evidence_registry` | FFTW performance pending |
| TET2 discretisation | complete | partial | supported | — | `how-to/reproduce/reproduce_tet2_qualification` | `reference/numerics/tet2_operators` | `explanation/spectral_mechanics/tet2_newton_gmres` | `reference/evidence/evidence_registry` | two-history qualification review |
| EBI-TET state sharing | complete | reviewed | negative | — | `how-to/reproduce/reproduce_ebi_falsification` | `reference/numerics/ebi_tet_contract` | `explanation/spectral_mechanics/ebi_srix_falsification` | `explanation/evidence/negative_results` | registered-case falsification |
| Native SRIX / Numba | complete | partial | supported | — | `how-to/crystal-plasticity/qualify_native_srix_backend` | `reference/numerics/native_srix_backend` | `explanation/native-srix/optimization_strategy` | `reference/evidence/native_srix_qualification` | expand stepwise IDs |
| FEMU and SVD | incomplete | blocked | open | — | not applicable: no boundary-only driver | `reference/numerics/femu_sensitivity_and_svd` | `explanation/identification/femu_identification` | `reference/evidence/evidence_registry` | smoke is a negative control |
| REGM | complete | partial | negative | — | `how-to/identification/run_regm_screening` | `reference/numerics/regm` | `explanation/identification/regm_screening` | `reference/evidence/regm_qualification` | observed-DIC transfer no-go |
| Reduced integration | complete | partial | negative | — | `how-to/mechanics/use_reduced_integration` | `reference/numerics/cps4r_hourglass` | `explanation/constitutive/reduced_integration_hourglass` | `explanation/evidence/negative_results` | content review required; CPS4R not production-qualified |
| Nonlocal / micromorphic | complete | partial | historical | — | `how-to/identification/run_micromorphic_identification` | `reference/scientific/nonlocal_parameters` | `explanation/reconstruction/nonlocal_micromorphic` | `explanation/evidence/current_scientific_status` | historical branch |
