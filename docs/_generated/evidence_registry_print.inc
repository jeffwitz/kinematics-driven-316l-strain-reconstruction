<!-- Generated from validation/documentation_evidence_registry.json. -->

**E-FFT-001 — Do the SciPy and FFTW DST-I implementations agree?**
: Fidelity: deterministic float64 transform contract. Parameters: full-Dirichlet grids 4x4, 8x8, 12x12, 24x24 and 48x48; one thread; estimate planning.
: Status: **verified**.
: Conclusion: Forward and inverse backend differences remain at machine precision.
: Sources: primary: `validation/reference_data/fftw_backend_contract_v1/contract.json`.

**E-FFT-002 — Do the FFTW round-trip and inner-product contracts hold?**
: Fidelity: deterministic float64 transform contract. Parameters: same registered grids and random seeds.
: Status: **verified**.
: Conclusion: Both implementations preserve the orthonormal DST-I to machine precision.
: Sources: primary: `validation/reference_data/fftw_backend_contract_v1/contract.json`.

**E-FFT-003 — Is FFTW wisdom machine-local and non-blocking when incompatible?**
: Fidelity: optional plan persistence contract. Parameters: metadata-keyed wisdom, compatible and corrupt files.
: Status: **verified by unit contract**.
: Conclusion: Compatible wisdom is loaded and corrupt or incompatible files are ignored.
: Sources: .

**E-FFT-004 — Does the FFTW B0 preconditioner agree with the SciPy reference?**
: Fidelity: deterministic float64 preconditioner contract. Parameters: B0, one thread, full-Dirichlet grids through 48x48.
: Status: **verified**.
: Conclusion: The complete forward-Green-inverse action agrees to machine precision.
: Sources: primary: `validation/reference_data/fftw_backend_contract_v1/contract.json`.

**E-FFT-005 — Does the optional FFTW backend preserve complete elastic solver results?**
: Fidelity: deterministic Hookean EBI and two-state TRI2 unit cases. Parameters: same boundary data, tolerance and constitutive tangent.
: Status: **verified by unit contract**.
: Conclusion: EBI and two-state solver tests compare SciPy and FFTW results without changing the mechanics.
: Sources: .

**E-FFT-006 — Has FFTW performance been qualified for the complete spectral solver?**
: Fidelity: machine-specific benchmark. Parameters: planned meshes 12 through 192, planning and steady-state statistics.
: Status: **pending benchmark**.
: Conclusion: The benchmark script is available; no FFTW speedup is claimed yet.
: Sources: .

**E-SRIX-P43-001 — Is SRIX qualified as the production law for the registered quasi-static P43 spectral reconstruction?**
: Fidelity: two-state TRI2, 3D-condensed plane stress, nonsymmetric LGMRES Newton solve. Parameters: P43 100x100, eight proportional increments, tolerance 1e-8, MFront threads 4, FFTW threads 1.
: Status: **qualified for the registered case**.
: Conclusion: SRIX converges reproducibly with 46 Newton iterations, 637 Jacobian matvecs and a median wall time of 78.12 s (MAD 0.74 s); all three repeated field hashes are identical and the final residual is 6.13e-9.
: Sources: primary: `validation/_generated/performance/crystal_tet2_srix_p43_m100_repeated.json`<br>provenance: `validation/_generated/performance/crystal_tet2_srix_p43_m100_repeated_manifest.json`.

**E-MERIC-P43-001 — Is Méric-Cailletaud qualified for the registered P43 eight-increment production reconstruction?**
: Fidelity: two-state TRI2, 3D-condensed plane stress, same P43 crop and constitutive bridge. Parameters: 100x100 P43, total pseudo-time one, proportional path, eight-increment failure and retained 16-increment diagnostic.
: Status: **not qualified**.
: Conclusion: The eight-increment run fails local plane-stress Newton at 15 iterations with a 5.19e-5 MPa transverse-stress residual. With 16 increments the same monotone proportional path converges to 8.59e-10, but costs 304.5 s and does not establish temporal convergence of the fields.
: Sources: primary: `validation/_generated/performance/crystal_tet2_meric_p43_m100_failure.json`<br>primary: `validation/_generated/performance/crystal_tet2_meric_p43_time_refinement_16.json`.

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
: Sources: primary: `validation/reference_data/spectral_mechanics_evidence_v1/dtt_contract.json`.

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

**E-SRIX-REGM-001 — Does SRIX-REGM recover and rank SRIX laws on an exact digital twin?**
: Fidelity: exact M8 SRIX twin plus 20 complete adaptive forward simulations. Parameters: tau0, R, Q and b varied in fixed log coordinates; identity observation; EBSD-cubic K0.
: Status: **verified in exact kinematic space**.
: Conclusion: The truth is recovered in the identifiable subspace, and REGM/FEMU ranking passes with Spearman 0.866, log-Pearson 0.878 and top-five overlap 3/5.
: Sources: exact-twin recovery: `validation/reference_data/srix_regm_twin_v1/report.json`<br>exact-space ranking: `validation/reference_data/srix_regm_femu_ranking_v1/report.json`.

**E-SRIX-REGM-002 — Does SRIX-REGM remain centred and correctly ranked after the qualified DIC observation chain?**
: Fidelity: M8 transfer/noise twin and the same 20 complete forward candidates in observed space. Parameters: qualified affine-preserving transfer; deterministic measured-noise realization; qualified spectral whitener.
: Status: **negative result**.
: Conclusion: No. Transfer alone moves the objective below the truth and reduces REGM/FEMU ranking to Spearman 0.326, log-Pearson 0.276 and top-five overlap 2/5. P43 identification is blocked.
: Sources: truth-centering test: `validation/reference_data/srix_regm_transfer_noise_v1/report.json`<br>observed ranking gate: `validation/reference_data/srix_regm_femu_observed_ranking_v1/report.json`.

**E-SRIX-REGM-003 — Does the observed SRIX-REGM ranking fail because observation is applied before constitutive replay?**
: Fidelity: M8 twin; frozen 20-candidate ranking; raw versus affine-preserving replay histories; periodic and affine-preserving score transfers. Parameters: same SRIX theta4 population, EBSD-cubic K0 and observed FEMU target as E-SRIX-REGM-002.
: Status: **verified diagnosis; observed-input formulation remains NO-GO**.
: Conclusion: Raw replay retains the observed FEMU ranking (Spearman 0.940-0.950), while transferred replay fails before scoring (Spearman 0.338). The truth pseudo-displacement bias is 11.64 times the candidate spread.
: Sources: A/B/C/D ablation and truth-bias diagnostic: `validation/reference_data/srix_regm_observation_placement_v1/report.json`.

**E-SRIX-REGM-004 — Can a low-dimensional latent-history correction restore observed-space REGM ranking on an exact twin?**
: Fidelity: M8 exact twin; snapshot-POD reconstruction of u*-O(u*); no new FEMU solves. Parameters: ranks 0,1,2,3,4,5 and full rank 162; affine-preserving score transfer; same 20 candidates.
: Status: **supported as a twin upper bound; not a real-data method**.
: Conclusion: Rank five recovers 99.9897% of missing-history energy and reaches Spearman 0.859, log-Pearson 0.888 and top-five overlap 4/5. The basis uses the exact twin truth and therefore does not authorize P43.
: Sources: latent-mode rank curve: `validation/reference_data/srix_regm_latent_modes_v1/report.json`.

**E-SRIX-REGM-005 — Can a damped first-order mechanical projection recover the observed-space SRIX ranking?**
: Fidelity: M8 transferred twin; one or two causal projection passes; same 20 candidates and observed FEMU target. Parameters: preset SRIX reference law; EBSD-cubic K0; damping 0.25, 0.50 and 1.00.
: Status: **negative result; simple projection rejected**.
: Conclusion: The projection lowers the reference residual but does not restore the candidate ordering: Spearman remains 0.326-0.341 and top-five overlap remains 2/5.
: Sources: projection ranking and truth-distance test: `validation/reference_data/srix_regm_mechanical_projection_v1/report.json`.

**E-SRIX-REGM-006 — Does exact-space SRIX-REGM preserve the local information geometry of observed FEMU?**
: Fidelity: M8 twin; exact and transferred REGM finite differences versus eight complete observed FEMU perturbations. Parameters: log tau0, R, Q, b; central step 3e-3; eight macro endpoints; affine-preserving transfer.
: Status: **negative surrogate-geometry result**.
: Conclusion: Observed FEMU has normalized spectrum (1, 0.542, 0.407, 0.0679), whereas exact REGM has (1, 0.422, 0.0324, 4.65e-5). REGM is therefore not information-preserving for four-parameter identification.
: Sources: Fisher/Gauss-Newton geometry comparison: `validation/reference_data/srix_regm_information_geometry_v1/report.json`.

**E-SRIX-REGM-007 — Does a statewise SRIX algorithmic tangent restore the missing REGM/FEMU sensitivity geometry?**
: Fidelity: exact M8 twin; statewise consistent tangent reconditioning; no global Newton solve. Parameters: log tau0, R, Q, b; central step 3e-3; EBSD-cubic tangent assembled at each scored state.
: Status: **negative reconditioner result**.
: Conclusion: No. The algorithmic tangent changes the REGM spectrum only modestly and does not recover the FEMU weak directions: normalized spectrum (1, 0.37594, 0.03469, 8.62e-5), condition number 1.16e4, and leading angle 73.9 degrees.
: Sources: statewise algorithmic-tangent geometry: `validation/reference_data/srix_regm_algorithmic_tangent_v1/report.json`.

**E-SRIX-REGM-008 — Does one causal tangent correction per increment restore REGM/FEMU sensitivity geometry?**
: Fidelity: exact M8 twin; sequential one-correction replay with constitutive re-evaluation and commit. Parameters: log tau0, R, Q, b; central step 3e-3; affine-preserving score transfer.
: Status: **negative sequential-surrogate result**.
: Conclusion: No. The sequential replay improves conditioning but leaves the rank-two angle at 67.9 degrees and the normalized weak directions at 0.0576 and 2.30e-4, far below observed FEMU values 0.4067 and 0.0679.
: Sources: sequential causal replay geometry: `validation/reference_data/srix_regm_sequential_one_newton_v2/report.json`.

**E-SRIX-REGM-009 — Does scoring the cumulative accepted endpoint displacement repair the sequential REGM/FEMU geometry?**
: Fidelity: exact M8 twin; sequential one-correction replay; incremental and cumulative endpoint observables. Parameters: log tau0, R, Q, b; central step 3e-3; affine-preserving score transfer.
: Status: **negative corrected-observable result**.
: Conclusion: No. The cumulative endpoint observable raises the third direction to 0.0938 but leaves the fourth at 2.17e-4; the leading angle to FEMU is 74.7 degrees and P43 remains blocked.
: Sources: corrected cumulative endpoint geometry: `validation/reference_data/srix_regm_sequential_one_newton_v3/report.json`.

**E-SRIX-FEMU-COMMON-PATH-001R — Does the corrected Dirichlet contract yield one strict common path for the FEMU base and all eight parameter perturbations?**
: Fidelity: M8 SRIX twin; corrected free-DOF initial predictor; unqualified v9 proposal with synchronized local bisection. Parameters: log tau0, R, Q, b; central step 3e-3; strict oracle replay.
: Status: **demonstrated on corrected discrete path**.
: Conclusion: The corrected common path converged for all nine trajectories after 25 local insertions (94 steps). Direct-vs-FD relative column errors are 3.95e-4, 9.71e-4, 7.94e-5 and 7.96e-5, with cosines above 0.9999995. This qualifies direct FEMU differentiation on the 94-step path, but not path-refinement convergence or parameter identification.
: Sources: corrected common-path rebaseline: `validation/reference_data/srix_femu_common_path_gate_v17/report.json`.

**E-SRIX-FEMU-PATH-002R — Does the corrected FEMU sensitivity geometry converge under nested loading-path refinement?**
: Fidelity: M8 SRIX twin; corrected Dirichlet contract 2; L0/L1/L2 nested paths with strict local repairs. Parameters: log tau0, R, Q, b; direct sensitivity; same physical L0 scoring fractions.
: Status: **negative primary gate**.
: Conclusion: The observed forward is stable (2.18e-4 then 9.51e-5 relative changes), but L1-to-L2 sensitivity changes remain 3.67% and 4.20% for log(tau0) and log(R), with a 2.606 degree rank-3 angle. The pre-registered path-convergence gate is false; no identification or P43 is authorized.
: Sources: nested corrected path convergence: `validation/reference_data/srix_femu_path_convergence_v3/report.json`.

**E-SRIX-FEMU-PATH-002S — Does one final L3 refinement stabilize the direct FEMU sensitivity geometry?**
: Fidelity: M8 SRIX twin; corrected Dirichlet contract 2; L3 nested path with strict local repairs. Parameters: log tau0, R, Q, b; direct sensitivity; L2 artefact reused.
: Status: **blocked at direct sensitivity evaluation**.
: Conclusion: The L3 forward converged with 809 steps and 25 repairs, but a constitutive shadow replay failed with MFront status -1 while constructing the direct Jacobian. No L2-to-L3 sensitivity metric is claimed; no L4, identification or P43 is authorized.
: Sources: final path extension diagnostic: `validation/reference_data/srix_femu_path_convergence_v4/report.json`.

**E-SRIX-FEMU-SHADOW-003 — Where does the L3 direct FEMU shadow replay fail, and does reducing h remove the failure?**
: Fidelity: M8 SRIX twin; corrected Dirichlet contract 2; exact 809-step L3 path. Parameters: log tau0, R, Q, b; h in {0.003, 0.0015, 0.001}.
: Status: **L3 shadow failure localized; L2 h diagnostic passes**.
: Conclusion: The L3 forward converges, but fixed-current-strain shadow tau0-minus fails at accepted increment 271, fraction 0.232177734375 to 0.2322998046875, with MFront status -1. On L2 all three preregistered h values complete. No h is adopted and no L2-to-L3 sensitivity convergence is claimed.
: Sources: machine-readable shadow diagnostic: `validation/reference_data/srix_femu_shadow_diagnostic_v1/report.json`.

**E-SRIX-FEMU-SHADOW-003B — Does reducing h allow the failing tau0-minus shadow to pass the L3 failure increment, and are L2 Jacobians stable?**
: Fidelity: M8 SRIX twin; L3 exact-path prefix through increment 271; L2 392-step Jacobian comparison. Parameters: h = 0.003, 0.0015, 0.001.
: Status: **diagnostic positive at the local prefix; full L3 adoption pending**.
: Conclusion: On the L3 prefix, h=0.003 fails at tau0-minus while h=0.0015 and h=0.001 pass increment 271. On L2, maximum column changes are 0.204% and 0.246%, with minimum cosines 0.9999981 and 0.9999973; the normalized spectrum remains stable. The smaller h values are candidates, not yet adopted.
: Sources: prefix and L2 h comparison: `validation/reference_data/srix_femu_shadow_diagnostic_v1/report.json`<br>machine-readable Jacobian comparison: `validation/reference_data/srix_femu_shadow_diagnostic_v1/l2_jacobian_h_sweep.json`.

**E-SRIX-FEMU-SHADOW-003C — Do the two prospective shadow steps produce stable complete L3 Jacobians, and does PATH-002S pass?**
: Fidelity: M8 SRIX twin; exact 809-step L3 path; full direct shadow replay. Parameters: h=0.0015 primary and h=0.001 control.
: Status: **L3 h stability passed; L2-to-L3 path gate remains negative**.
: Conclusion: Both L3 Jacobians complete. Their maximum column difference is 0.102%, minimum cosine 0.99999957 and rank-3 angle 0.136°, so h=0.0015 is adopted as the primary shadow-FD step and h=0.001 remains a control. PATH-002S remains negative only because L2-to-L3 rank-3 angles are 2.290° and 2.195°, above the preregistered 2° threshold. Identification and P43 remain blocked.
: Sources: full L3 h qualification: `validation/reference_data/srix_femu_shadow_h_l3_v1/report.json`.

**E-SRIX-P43-SYNTH-001 — La chaîne de sensibilité FEMU directe retrouve-t-elle une vérité SRIX connue sur un crop P43 synthétique ?**
: Fidelity: P43 M20 réel avec EBSD réel, histoire DIC réparée, 32 incréments, observation identité, sans bruit. Parameters: theta4 log-paramètres; shadow FD h=0.0015; départ proche; six évaluations préenregistrées.
: Status: **smoke positif; qualification multi-départs non terminée**.
: Conclusion: Le résidu RMS passe de 8.8502e-8 à 1.2693e-13 et les paramètres finaux sont tau0=40.000004 MPa, R=18.781912 MPa, Q=10.006888 MPa, b=2.997907. La SVD vaut (1, 0.135725, 0.036116, 1.031e-4), avec rho(Q,b)=0.999999997; Q et b ne sont donc pas revendiqués séparément. L'optimiseur a atteint la limite de six évaluations, et aucun P43 expérimental n'est autorisé.
: Sources: machine-readable synthetic smoke result: `validation/reference_data/p0043_synthetic_identification_v1/report.json`.
