<!-- Generated from validation/documentation_evidence_registry.json. -->

**E-SPECTRAL-001 — Does the one-point spectral stencil develop a high-frequency near-null mode?**
: Fidelity: registered spectral witness. Parameters: one-point stencil, recorded non-affine load.
: Status: **verified**.
: Conclusion: The witness stalled after 7928 iterations and retained a high-frequency displacement fraction of about 1.43e-4.
: Sources: primary: `validation/reference_data/spectral_mechanics_evidence_v1/report.json`.

**E-SPECTRAL-002 — Does TET2 approach CPS4 under refinement?**
: Fidelity: two-state TET2 versus CPS4. Parameters: registered SRIX case, 24x24.
: Status: **supported**.
: Conclusion: At 24x24, TET2 is within 0.72% in accumulated slip and 0.19% in stress of CPS4.
: Sources: primary: `validation/reference_data/spectral_mechanics_evidence_v1/report.json`.

**E-SPECTRAL-003 — Does Newton-GMRES reach and verify the spectral equilibrium solution?**
: Fidelity: transactional matrix-free Newton-GMRES. Parameters: 12x12 and 24x24, tolerance 1e-8.
: Status: **verified**.
: Conclusion: The TET2 verification residuals are below 1e-11 and 1e-13 at 12x12 and 24x24 respectively.
: Sources: primary: `validation/_generated/ebi_tet/state_sharing_m12.json`<br>primary: `validation/_generated/ebi_tet/state_sharing_m24.json`.

**E-EBI-001 — Does the SRIX Hookean prerequisite and directional EBI tangent pass?**
: Fidelity: plane-stress SRIX Hookean reconstruction. Parameters: three orientations, plastified states, one state per pixel.
: Status: **verified**.
: Conclusion: The Hookean prerequisite passes and the directional tangent reaches 8.58e-7 minimum relative error.
: Sources: primary: `validation/_generated/ebi_tet/srix_hookean.json`<br>primary: `validation/_generated/ebi_tet/srix_tangent.json`.

**E-EBI-002 — Is the dominant registered EBI error caused by state sharing?**
: Fidelity: same TET2 kinematics, TET2 versus EBI. Parameters: registered SRIX case, 24x24.
: Status: **supported**.
: Conclusion: EBI versus TET2 accumulated-slip error is 5.39%, while TET2 versus CPS4 is 0.72%.
: Sources: primary: `validation/_generated/ebi_tet/state_sharing_m24.json`.

**E-DTT-001 — Do the DST-I transform and discrete inner product contracts hold?**
: Fidelity: full-Dirichlet DST-I unit tests. Parameters: rectangular and square unit-test grids.
: Status: **verified**.
: Conclusion: The documented round-trip and inner-product contracts pass at the declared numerical precision.
: Sources: primary: `validation/reference_data/spectral_mechanics_evidence_v1/report.json`.

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
: Fidelity: full 660 by 610 element local MFront solve with measured boundary states. Parameters: 40 direct-reference states; endpoint anchored; documented states 31 and 32 linearly repaired; unchanged constitutive settings; per-iteration Newton trace.
: Status: **resolved**.
: Conclusion: Yes. The replay was blocked by a software defect, not by mechanics: the history-branch elastic predictor was solved against a CSR buffer already overwritten by the elastoplastic tangent. With the predictor given its own operator the full 40-state history completes, with 65 converged increments and 3 cutbacks. Only measured-history runs were affected, so no archived result changed.
: Sources: primary measurement-history repair: `validation/reference_data/dic_multistep_history_p0043_repaired_v1/report.json`<br>primary completed measured-history mechanics: `validation/reference_data/dic_multistep_predictor_fix_p0043_v1/report.json`.

**E-DIC-008 — Does the loading path change the reconstructed plasticity at a shared endpoint?**
: Fidelity: two full local MFront solves on P43, core 360 by 310, padding excluded. Parameters: identical endpoint boundary displacement; measured 40-state history against a proportional ramp at 40 increments; archived 20-increment ramp as discretisation control.
: Status: **supported**.
: Conclusion: Yes. Core PEEQ differs by 15.82 % in relative L2 while the discretisation control differs by 0.20 %, so the effect is 78 times the control. The excess is concentrated in the localisation bands, band structure ratio 13.11, so it is not the diffuse DIC noise ratchet. The measured path accumulates more plasticity.
: Sources: primary path-dependence comparison: `validation/reference_data/dic_multistep_path_dependence_p0043_v1/report.json`.

**E-DIC-009 — Does the measured loading path agree better with DIC total strain than a proportional ramp?**
: Fidelity: symmetric image-level observation of both final fields through DISFlow, core only. Parameters: legacy_script_2021 primary profile and declared_medium_v4 sensitivity profile; significance margins taken from the archived DIC-noise sensitivity intervals.
: Status: **indistinguishable**.
: Conclusion: No. Under symmetric observation no registered metric moves by more than its margin, on either profile. A 15.8 % change of an unobservable internal variable produces less than DIC noise sensitivity on the observable, because EVM is dominated by the imposed kinematics and DISFlow smooths at about 49 px. This observable can neither validate nor falsify the measured history.
: Sources: primary measured-path observation: `validation/reference_data/dic_multistep_observed_path_p0043_v1/measured_legacy_script_2021/report.json`<br>primary proportional-path observation: `validation/reference_data/dic_multistep_observed_path_p0043_v1/proportional40_legacy_script_2021/report.json`.

**E-DIC-010 — Can the imposed boundary be restricted to the modes the DIC chain resolves?**
: Fidelity: rank-3 truncation of the boundary deviation from the endpoint ramp, then a full local MFront solve. Parameters: origin, endpoint and interior bit-identical; exact Dirichlet enforcement retained; rank fixed before the run.
: Status: **supported**.
: Conclusion: Yes. The filter removes 0.00972 px RMS against a measured noise floor of 0.0511 px, so 5.3 times below it, and the stage-0 roughness rule independently selects the same three modes. The filtered history converges in 245 Newton iterations with no cutback, against 469 iterations and 3 cutbacks unfiltered, while perturbing core PEEQ by only 1.63 % and leaving DIC agreement indistinguishable. The gain is numerical, not evidential.
: Sources: primary boundary filter: `validation/reference_data/dic_multistep_history_p0043_modal3_v1/report.json`<br>primary filtered-history mechanics: `validation/reference_data/dic_multistep_modal3_p0043_v1/report.json`.
