<!-- Generated from validation/documentation_evidence_registry.json. -->

**E-TABLE-001 — Is the available Abaqus-oriented table definition generated exactly as documented?**
: Fidelity: definition. Parameters: 1000 rows over PEEQ 0 to 0.2.
: Status: **verified at definition level**.
: Conclusion: The versioned generator reproduces the documented table grid and Ludwik values exactly; this is not an Abaqus FE parity test.
: Sources: primary: `validation/reference_data/historical_table_definition_v1/report.json`.

**E-LOCAL-001 — Does the analytical MFront law reproduce the in-house historical table path?**
: Fidelity: material point. Parameters: three loading paths, 200 steps.
: Status: **verified**.
: Conclusion: Stress and PEEQ differences remain below all declared comparison thresholds below the historical table cap.
: Sources: primary: `validation/reference_data/mfront_material_point_v1/report.json`.

**E-LOCAL-002 — Do InHouse/Table and InHouse/MFront produce coherent finite-element fields?**
: Fidelity: small DIC-driven FEM. Parameters: 10 by 10 elements, 20 increments.
: Status: **verified**.
: Conclusion: All historical fields pass their declared relative error thresholds.
: Sources: primary: `validation/reference_data/mfront_newton_dic_10x10_v1/report.json`.

**E-DIAG-001 — Does output-only spatial broadening improve DIC agreement?**
: Fidelity: saved full-resolution fields. Parameters: selection ROI and held-out ROI.
: Status: **supported**.
: Conclusion: A positive Helmholtz length improves pre-registered metrics without modifying mechanics.
: Sources: primary: `validation/reference_data/nonlocality_helmholtz_article_p0042_confirmatory_v1/confirmation-report.json`.

**E-NL-002 — Does constitutive micromorphic coupling redistribute plastic localization?**
: Fidelity: full-resolution band-containing ROI. Parameters: fixed ell, alpha 0, 1, 2 and 4.
: Status: **verified**.
: Conclusion: The converged numerical fields show lower PEEQ peaks and gradients as coupling increases; the figure metadata is retained only as secondary provenance.
: Sources: primary: `validation/reference_data/joint_nonlocal_identification_p0043_v1/results.json`<br>secondary figure provenance: `validation/figures/p0043-alpha-comparison/plot_metadata.json`.

**E-ID-001 — Can coupling strength and spatial length be separated?**
: Fidelity: F0 and F1. Parameters: 23-point homogeneous Newton-25 design.
: Status: **provisional**.
: Conclusion: Interior amplitude optima exist at 40 and 60 um, but two short-length points fail; F2 confirmation and transfer remain absent.
: Sources: primary: `validation/joint_nonlocal_identifiability_p0043_newton25_execution.json`.
