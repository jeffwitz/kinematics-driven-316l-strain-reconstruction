# E-SRIX-FEMU-COMMON-PATH-001R — results

Status: **blocked_oracle** (2026-08-24)

This gate is the re-baseline after the initial-Dirichlet contract correction.
The old `srix_femu_common_path_gate_v9` and `srix_femu_path_convergence_v2`
artifacts remain historical and are superseded for scientific interpretation.

The machine-readable result is
`validation/reference_data/srix_femu_common_path_gate_v16/` (`dirty=false`,
commit `73a08be`). The historical v9 proposal was used only as an unqualified
starting partition; no cache was overwritten.

The search inserted 12 local nodes and reached a 69-step candidate. The strict
oracle then blocked on `R_plus` at increment 23 (fraction approximately
`0.21875`), after earlier strict failures had already required local
subdivisions. Base plus all eight perturbations were therefore **not**
converged on one partition.

The search configuration was fail-fast only for proposing local bisections; it
did not relax the final oracle tolerance. Because the strict common-path gate
is blocked, no direct-vs-FD Jacobian, new spectrum, PATH-002 refinement,
identification, or P43 claim is authorized.
