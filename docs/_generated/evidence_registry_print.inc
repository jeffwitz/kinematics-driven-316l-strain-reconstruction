<!-- Generated from validation/documentation_evidence_registry.json. -->

**E-FE-001 — Does the finite-element kernel satisfy its numerical contract?**
: Fidelity: unit and integration. Parameters: local baseline.
: Status: **verified**.
: Conclusion: Patch, material-point, tensor and nonlinear regression tests cover the supported Q4 plane-stress scope.
: Source: `tests`.

**E-LOCAL-001 — Does the analytical MFront law reproduce the historical tabulated baseline?**
: Fidelity: material point. Parameters: three loading paths, 200 steps.
: Status: **verified**.
: Conclusion: Stress and PEEQ differences remain within the declared comparison thresholds below the historical table cap.
: Source: `validation/reference_data/mfront_material_point_v1/report.json`.

**E-LOCAL-002 — Do the historical table and MFront produce coherent finite-element fields?**
: Fidelity: small DIC-driven FEM. Parameters: 10 by 10 elements, 20 increments.
: Status: **verified**.
: Conclusion: All historical fields pass the declared relative error thresholds.
: Source: `validation/reference_data/mfront_newton_dic_10x10_v1/report.json`.

**E-3D-001 — Can a three-dimensional law be condensed without changing the 2D solver?**
: Fidelity: small DIC-driven FEM. Parameters: native plane stress versus 3D condensation.
: Status: **verified**.
: Conclusion: Native and condensed MFront paths agree within numerical tolerances while the global finite-element problem remains two-dimensional.
: Source: `validation/reference_data/mfront_3d_condensed_dic_10x10_v1/report.json`.

**E-DIAG-001 — Does output-only spatial broadening improve DIC agreement?**
: Fidelity: saved full-resolution fields. Parameters: selection ROI and held-out ROI.
: Status: **supported**.
: Conclusion: A positive Helmholtz length improves amplitude and localization metrics on selection and held-out regions without modifying mechanics.
: Source: `validation/reference_data/nonlocality_helmholtz_article_p0042_confirmatory_v1/confirmation-report.json`.

**E-NL-002 — Does constitutive micromorphic coupling redistribute plastic localization?**
: Fidelity: full-resolution band-containing ROI. Parameters: fixed ell, alpha 0, 1, 2 and 4.
: Status: **verified**.
: Conclusion: Coupling lowers concentrated PEEQ peaks and broadens the active support; stronger coupling can improve global error while erasing relevant bands.
: Source: `validation/figures/p0043-alpha-comparison/plot_metadata.json`.

**E-ID-001 — Can coupling strength and spatial length be separated?**
: Fidelity: F0 and F1. Parameters: 23-point homogeneous Newton-25 design.
: Status: **provisional**.
: Conclusion: Constant-Achi responses remain distinct, supporting an independent length effect, but two short-length points fail and no F2 confirmation or transfer has been performed.
: Source: `validation/joint_nonlocal_identifiability_p0043_newton25_execution.json`.
