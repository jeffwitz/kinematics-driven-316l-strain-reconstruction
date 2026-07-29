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

**E-DIC-001 — What noise upper bound and spatial transfer does the declared DISFlow reproduction produce?**
: Fidelity: full-crop candidate null pair and synthetic 1024 by 1024 warps. Parameters: OpenCV 4.14 DIS medium; native finest scale 0; reported alpha, delta, gamma, epsilon; 30 iterations.
: Status: **verified for the reproduction implementation**.
: Conclusion: At native finest scale 0, the candidate-repeat EVM RMS is 4.52 percent of the final DIC RMS but remains spatially coherent, so it is an upper bound rather than a certified noise floor. Synthetic sinusoidal and band tests show strongly scale-dependent transfer.
: Sources: primary: `validation/reference_data/dic_measurement_chain_v4/manifest.json`<br>primary: `validation/reference_data/dic_measurement_chain_v4/null_test_report.json`<br>primary: `validation/reference_data/dic_measurement_chain_v4/transfer_report.json`.

**E-DIC-002 — Can Charbonnier epsilon remove the waviness of a recovered 32 px strain band without biasing its width and amplitude?**
: Fidelity: synthetic 1024 by 1024 real-speckle warp at native scale 0. Parameters: 32 px Gaussian-FWHM EVM band; epsilon from 0.0002 to 2; all other DISFlow settings fixed.
: Status: **exploratory**.
: Conclusion: Increasing epsilon strongly suppresses along-band waviness, but the strong reduction is inseparable from peak attenuation and excessive band broadening. Intermediate epsilon 0.01 reduces waviness while preserving peak amplitude, but narrows the band to 26 px.
: Sources: primary: `validation/reference_data/dic_epsilon_band32_v2/manifest.json`.

**E-DIC-003 — How do the declared V4 and supplied legacy-source DISFlow profiles differ after correcting the synthetic warp?**
: Fidelity: candidate null pair and synthetic 1024 by 1024 real-speckle warps. Parameters: native scale 0; declared medium profile versus no-argument legacy-source factory; legacy and iterative inverse warps.
: Status: **verified for OpenCV 4.14**.
: Conclusion: The corrected inverse materially changes localized-band metrology while barely moving sinusoidal MTF-50. The legacy-source profile has a different null response and transfer from the declared V4 profile; it is primary by provenance, not by FEM/DIC score.
: Sources: V4 non-regression: `validation/reference_data/dic_legacy_profile_comparison_v1/case_a_declared_legacy_warp/manifest.json`<br>corrected-warp control: `validation/reference_data/dic_legacy_profile_comparison_v1/case_b_declared_corrected_warp/manifest.json`<br>legacy-source provenance profile: `validation/reference_data/dic_legacy_profile_comparison_v1/case_c_legacy_corrected_warp/manifest.json`.

**E-DIC-004 — What changes when archived P43 FEM displacement is observed through the same image-level DISFlow chain as DIC?**
: Fidelity: eight full-resolution archived-displacement replays. Parameters: alpha 0, 1, 2 and 4; legacy-source profile primary; declared V4 sensitivity; no mechanical rerun.
: Status: **verified**.
: Conclusion: Symmetric observation substantially reduces apparent L2 error and raises correlation, but it separates the objectives: alpha 4 is best for amplitude and correlation while alpha 1 is best for absolute-q90 overlap. The old identification objective must not be reused.
: Sources: primary consolidated metrics: `validation/reference_data/dic_symmetric_observation_p0043_v1/summary.json`.

**E-DIC-005 — Does direct local brightness mismatch explain the remaining observed-FEM/DIC EVM error?**
: Fidelity: full-resolution P43 direct-image diagnostic on four archived V3 replays. Parameters: alpha 0, 1, 2 and 4; cell-averaged direct brightness residual; fixed q90 sensitivity; no mechanical rerun.
: Status: **negative result**.
: Conclusion: Residual/error Pearson and Spearman associations are negligible, and excluding the worst photometric decile leaves the candidate ordering and field metrics essentially unchanged. The primary V3 comparison therefore remains unmasked.
: Sources: primary: `validation/reference_data/dic_photometric_quality_p0043_v1/report.json`.

**E-DIC-006 — Does the measured repeated-frame DIC residual change the ranking of the archived P43 candidates?**
: Fidelity: 256 translated measured-residual surrogate samples on four immutable V3 replays. Parameters: alpha 0, 1, 2 and 4; legacy-source profile; contiguous full-support windows; seed 20260729; no mechanical rerun.
: Status: **verified sensitivity**.
: Conclusion: The measured repeat-frame residual changes metric values but not their objective-dependent ranking: alpha 4 is always best for L2, RMSE, correlation and active-area error, alpha 1 for absolute-q90 IoU, and the local model for relative top-10 IoU. These are surrogate sensitivity intervals, not confidence intervals.
: Sources: primary: `validation/reference_data/dic_uncertainty_propagation_p0043_v1/report.json`.

**E-EBSD-001 — What structural scale is carried by the archived grain-mean Schmid field?**
: Fidelity: independent EBSD/Schmid spatial statistic. Parameters: frozen exponential-tail fit, directional profiles, 4 by 4 block bootstrap.
: Status: **verified**.
: Conclusion: The radial decay is 179.38 um, spatial-block median 108.57 um, and directional ratio 1.60; this is an independent structural scale, not a direct measurement of ell.
: Sources: primary: `validation/reference_data/ebsd_structural_length_v1/report.json`.

**E-DIC-007 — Can the local P43 mechanics replay the measured 40-state DIC boundary history?**
: Fidelity: full 660 by 610 element local MFront solve with measured boundary states. Parameters: 40 direct-reference states; endpoint anchored; documented states 31 and 32 linearly repaired; state 3 and state 4 measurement-only bridge diagnostics; transactional internal states; early-state spatial and photometric outlier audit; unchanged constitutive settings.
: Status: **blocked negative result**.
: Conclusion: The documented late optical-flow corruption is removed. Early states 3 and 4 contain no spatial or photometric spike at the rejected elements, while Newton creates strains at least 3.64e5 times larger than measured. Transparent state bridges do not move the limit. Measured-history prediction remains blocked by nonlinear globalisation.
: Sources: primary measurement-history repair: `validation/reference_data/dic_multistep_history_p0043_repaired_v1/report.json`<br>primary mechanical failure: `validation/reference_data/dic_multistep_mechanics_p0043_measured_repaired_v1/failure_report.json`<br>measurement-only state bridge: `validation/reference_data/dic_multistep_history_p0043_state3_bridge_v1/report.json`<br>state-bridge mechanical failure: `validation/reference_data/dic_multistep_mechanics_p0043_measured_state3_bridge_secant_v1/failure_report.json`<br>mechanically selected target-state bridge: `validation/reference_data/dic_multistep_history_p0043_blocked_state4_v1/report.json`<br>blocked-target bridge mechanical failure: `validation/reference_data/dic_multistep_mechanics_p0043_measured_blocked_state4_v1/failure_report.json`<br>primary early-state boundary and image outlier audit: `validation/reference_data/dic_multistep_p0043_boundary_outlier_v1/report.json`.
