# SRIX qualification evidence

**Mode:** reference  
**Domain:** evidence

| Claim | Evidence ID | Artifact path | Observed value/status | Boundary |
|---|---|---|---|---|
| SRIX equations and conventions | `E-SRIX-P43-001` | `validation/_generated/performance/crystal_tet2_srix_p43_m100_repeated.json` | qualified registered case; final residual `6.13e-9` | two-state TRI2, 3-D condensed plane stress |
| P43 robustness | `E-SRIX-P43-001` | `validation/_generated/performance/crystal_tet2_srix_p43_m100_repeated_manifest.json` | 46 Newton, 637 Jacobian matvecs, median 78.12 s (MAD 0.74 s) | P43 100x100, eight proportional increments |
| native/MFront equivalence | native qualification reports | `validation/p0043_srix_m100_coupled_optimization_results.md` and related reports | fields/tangents compared at declared tolerances | only recorded NumPy/MFront configurations |
| parameters identified for 316L | none | — | **not claimed** | registered `R` is an analytical transposition |

Stress, slips, tangents, residuals and fields are compared with stated
tolerances. Wall time and Newton/GMRES counts are performance measurements,
never evidence of scientific equivalence. The registry remains authoritative;
the IDs above are stable keys into
`validation/documentation_evidence_registry.json`.
