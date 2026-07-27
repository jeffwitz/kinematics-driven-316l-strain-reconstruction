<!-- Generated from validation/documentation_evidence_registry.json. -->

**E-MAP-001 — What do boundary kinematics and the spatial placement of the material maps contribute?**
: Fidelity: full-resolution local FEM controls. Parameters: P43 mapped, homogeneous 124/380 MPa, joint map shift 600/500 pixels.
: Status: **supported**.
: Conclusion: The homogeneous control minimizes global error but erases q90-active bands; translating the maps strongly degrades spatial correlation and overlap, showing that their original placement contains localisation information.
: Sources: primary: `validation/reference_data/material_map_controls_p0043_v1/report.json`.

**E-EQ-001 — What is the cross-section-integrated equilibrium baseline of the archived P43 fields?**
: Fidelity: saved full-resolution element-centred stresses. Parameters: local and alpha 1, 2, 4; padded domain and retained core.
: Status: **reported baseline**.
: Conclusion: Lateral shear flux closes about 69 to 71 percent of the naive adjacent-section imbalance; the remaining RMS residual is about 1.5e-4 to 2.3e-4 of mean section force.
: Sources: primary: `validation/reference_data/section_equilibrium_p0043_v1/report.json`.

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

**E-3D-001 — Can a three-dimensional law be condensed without changing the 2D solver?**
: Fidelity: small DIC-driven FEM. Parameters: native plane stress versus 3D condensation.
: Status: **verified**.
: Conclusion: Native and condensed MFront paths agree within declared tolerances while the global problem remains two-dimensional.
: Sources: primary: `validation/reference_data/mfront_3d_condensed_dic_10x10_v1/report.json`.

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

**E-EBSD-001 — What structural scale is carried by the archived grain-mean Schmid field?**
: Fidelity: independent EBSD/Schmid spatial statistic. Parameters: frozen exponential-tail fit, directional profiles, 4 by 4 block bootstrap.
: Status: **verified**.
: Conclusion: The radial decay is 179.38 um, spatial-block median 108.57 um, and directional ratio 1.60; this is an independent structural scale, not a direct measurement of ell.
: Sources: primary: `validation/reference_data/ebsd_structural_length_v1/report.json`.
