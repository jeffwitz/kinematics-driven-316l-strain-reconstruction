# E-SRIX-FEMU-COMMON-PATH-001R — results

Status: **in progress**

This gate is the re-baseline after the initial-Dirichlet contract correction.
The old `srix_femu_common_path_gate_v9` and `srix_femu_path_convergence_v2`
artifacts remain historical and are superseded for scientific interpretation.

The definitive machine-readable result will be written under
`validation/reference_data/srix_femu_common_path_gate_v15/` (or the next
versioned run directory) only after the strict common-path replay completes.
Until then, no new spectrum, identification result, or P43 claim is authorized.

The search configuration is fail-fast only for proposing local bisections; it
does not relax the final oracle tolerance. The final report must state the
number of path steps, the status of all nine variants, the direct-vs-FD column
errors and cosines, and the initialization contract version.
