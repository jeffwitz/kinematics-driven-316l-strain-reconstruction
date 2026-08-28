# SRIX qualification evidence

**Mode:** reference  
**Domain:** evidence

| Claim | Evidence ID / source | Artifact path | Observed value/status | Boundary |
|---|---|---|---|---|
| Constitutive equations and conventions | source/tests | `mfront/Fcc316LForestRubinSrix.mfront`; `tests/unit/core/test_forest_rubin_srix.py`; `validation/srix_canonical_qualification_report.md` | closed-form, dissipation, symmetry and tangent checks qualified at the recorded tolerances | model/integration qualification; no 316L calibration claim |
| P43 robustness | `E-SRIX-P43-001` | `validation/_generated/performance/crystal_tet2_srix_p43_m100_repeated.json` | 46 Newton, 637 Jacobian matvecs, median 78.12 s (MAD 0.74 s), final residual `6.13e-9` | two-state TRI2, 3-D condensed plane stress, P43 100x100 |
| native/MFront equivalence | registered qualification reports | `validation/p0043_srix_m100_coupled_optimization_results.md`, `validation/p0043_direct_coupled_tangent_results.md` | same fields and tangents within the report tolerances; coupled/direct paths keep the registered trajectory where compared | only recorded NumPy/MFront configurations; performance is a separate claim |
| native optimization performance | registered optimization reports | `validation/p0043_coupled_fused_block_results.md`, `validation/p0043_coupled_fused_tangent_results.md`, `validation/p0043_coupled_state_crossover_results.md` | M100 block 109.43→51.94 s; M20 direct tangent 3.327→2.310 s; large-batch state crossover is machine-dependent | wall time and scaling only; no new scientific equivalence claim |
| parameters identified for 316L | none | — | **not claimed** | registered `R` is an analytical transposition |

Stress, slips, tangents, residuals and fields are compared with stated
tolerances. Wall time and Newton/GMRES counts are performance measurements,
never evidence of scientific equivalence. The registry remains authoritative;
the IDs above are stable keys into
`validation/documentation_evidence_registry.json`.
