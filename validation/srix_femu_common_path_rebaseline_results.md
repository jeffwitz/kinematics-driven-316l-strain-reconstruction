# E-SRIX-FEMU-COMMON-PATH-001R — results

Status: **converged** (2026-08-24)

This gate is the re-baseline after the initial-Dirichlet contract correction.
The old `srix_femu_common_path_gate_v9` and `srix_femu_path_convergence_v2`
artifacts remain historical and are superseded for scientific interpretation.

The corrected continuation is
`validation/reference_data/srix_femu_common_path_gate_v17/` (`dirty=false`,
commit `387af84`). The v16 path was used only as an unqualified starting
partition; no cache was overwritten.

The synchronized search inserted 25 local nodes and reached a 94-step common
path. Base plus all eight perturbations converged under the strict oracle. The
global bisection budget was not reached; the recorded stop reason is null
because the oracle passed.

The direct FEMU sensitivity versus the common-path central FD passes for all
four log-parameters:

| direction | relative L2 error | cosine |
|---|---:|---:|
| `log(tau0)` | `3.95e-4` | `0.99999992` |
| `log(R)` | `9.71e-4` | `0.99999959` |
| `log(Q)` | `7.94e-5` | `1.00000000` |
| `log(b)` | `7.96e-5` | `1.00000000` |

The common-path FD normalized spectrum is
`(1, 0.18020, 0.04029, 6.32e-5)`, with condition number `1.58e4`.

The search configuration was fail-fast only for proposing local bisections; it
did not relax the final oracle tolerance. This qualifies the direct sensitivity
implementation against the corrected 94-step discrete FEMU path. It does not
yet establish convergence of the spectrum with respect to further path
refinement, and it does not authorize identification or P43.
