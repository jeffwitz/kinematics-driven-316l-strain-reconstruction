<!-- Generated from validation/documentation_evidence_registry.json. -->

**E-FE-001 — Does the finite-element kernel satisfy its numerical contract?**
: Fidelity: unit and integration. Parameters: local baseline.
: Status: **verified**.
: Conclusion: Patch, material-point, tensor and nonlinear regression tests cover the supported Q4 plane-stress scope.
: Sources: `tests (test suite)`.

**E-TABLE-001 — What part of the historical Abaqus-oriented definition is reproduced?**
: Fidelity: constitutive definition. Parameters: 1000-knot table, p <= 0.2, explicit clamp.
: Status: **verified**.
: Conclusion: InHouse/Table implements the available historical tabulated hardening definition. No complete Abaqus solve parity is inferred.
: Sources: `src/fem_inhouse/core/constitutive.py (implementation); docs/archive/explanation/material_law.md (historical contract)`.

**E-LOCAL-001 — Does analytical MFront reproduce InHouse/Table at material points?**
: Fidelity: material point. Parameters: three loading paths, 200 steps, below the historical cap.
: Status: **verified**.
: Conclusion: Stress and PEEQ differences remain within the pre-declared comparison thresholds.
: Sources: `validation/reference_data/mfront_material_point_v1/report.json (primary)`.

**E-LOCAL-002 — Do InHouse/Table and MFront produce coherent finite-element fields?**
: Fidelity: small DIC-driven FEM. Parameters: 10 by 10 elements, 20 increments.
: Status: **verified**.
: Conclusion: All historical fields pass the declared relative error thresholds.
: Sources: `validation/reference_data/mfront_newton_dic_10x10_v1/report.json (primary)`.

**E-3D-001 — Can a three-dimensional law be condensed without changing the 2D solver?**
: Fidelity: small DIC-driven FEM. Parameters: native plane stress versus 3D condensation.
: Status: **verified**.
: Conclusion: Native and condensed MFront paths agree within numerical tolerances while the global finite-element problem remains two-dimensional.
: Sources: `validation/reference_data/mfront_3d_condensed_dic_10x10_v1/report.json (primary)`.

**E-DIAG-001 — Does output-only spatial broadening improve DIC agreement?**
: Fidelity: saved full-resolution fields. Parameters: selection ROI and held-out ROI.
: Status: **supported**.
: Conclusion: A positive Helmholtz length improves amplitude and localization metrics on selection and held-out regions without modifying mechanics.
: Sources: `validation/reference_data/nonlocality_helmholtz_article_p0042_confirmatory_v1/confirmation-report.json (primary)`.

**E-NL-002 — Does constitutive micromorphic coupling redistribute plastic localization?**
: Fidelity: full-resolution band-containing ROI. Parameters: fixed ell, alpha 0, 1, 2 and 4.
: Status: **verified**.
: Conclusion: Coupling strongly lowers concentrated PEEQ peaks while preserving most mean PEEQ; the raw mechanical EVM improves without post-filtering.
: Sources: `validation/figures/p0043-alpha-comparison/plot_metadata.json (metrics and figure provenance); validation/nonlocal_p0043_validation_results.md (campaign report)`.

**E-ID-001 — Can coupling strength and spatial length be separated?**
: Fidelity: F0 and F1. Parameters: 23-point homogeneous Newton-25 design.
: Status: **provisional**.
: Conclusion: Versioned constant-Achi and fixed-alpha comparisons support an independent length effect, but two short-length points fail and no F2 confirmation or transfer has been performed.
: Sources: `validation/joint_nonlocal_identifiability_p0043_newton25_execution.json (execution attestation); docs/_static/joint_identification/joint_identification_constant_a_fields.svg (constant-Achi field comparison); docs/archive/explanation/joint_nonlocal_identification.md (versioned interpretation)`.
