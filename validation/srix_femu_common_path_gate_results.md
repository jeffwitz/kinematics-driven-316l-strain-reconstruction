# Synchronized common-path gate — adaptive trajectory blocker

The synchronized gate is implemented in
`scripts/qualify_srix_femu_common_path_gate.py`. It first runs the nine
adaptive trajectories, takes the union of their accepted fractions, and then
replays all directions on that common partition with local synchronized
bisection when necessary.

The first M8 execution did not reach the common replay. The directions through
`b_plus` returned, but `b_minus` remained in the adaptive solver for more than
40 minutes without returning an accepted path. The run was interrupted rather
than allowed to remain unbounded. The result is recorded in
`validation/reference_data/srix_femu_common_path_gate_v1/report.json`.

This is a diagnostic of adaptive-path cost/branching, not a negative result for
the direct sensitivity method. No common-path FD, SVD, or parameter claim is
authorized from this run.

## Three-policy rerun and strict common-path result

The driver now separates the exploratory seed, fail-fast path search, and
strict oracle policies. Seed fractions are cached under
`validation/reference_data/srix_femu_common_path_cache/`; the known-expensive
`b_minus` adaptive seed is skipped by default. The production controller still
uses its historical line-search threshold of `1.0`; only the exploratory seed
uses `0.25`.

The strict M8 replay was then resumed from the qualified 57-step partition in
`validation/reference_data/srix_femu_common_path_gate_v9/common_path.npz`.
The direct sensitivity and the fixed-path central FD use the same seven unique
common-path endpoints (the historical index list contains one duplicate after
normalization). The raw column comparison is:

| parameter direction | relative L2 error | cosine |
| --- | ---: | ---: |
| `log(tau0)` | 1.656e-3 | 0.99999896 |
| `log(R)` | 6.964e-4 | 0.99999983 |
| `log(Q)` | 6.105e-4 | 0.99999982 |
| `log(b)` | 6.166e-4 | 0.99999982 |

Thus the strict common-path gate passes its declared numerical criterion
(relative column error below 2 %, cosine above 0.999). This validates the
direct FEMU sensitivity implementation **against the same discrete path**;
it does not authorize P43 or an identification campaign.

The normalized singular spectrum of the common-path FD is
`(1, 0.1871, 0.04053, 5.35e-5)`, with condition number `1.87e4`. It is not the
older adaptive-path spectrum `(1, 0.542, 0.407, 0.0679)`: the two are different
discrete load-path experiments. The gate therefore establishes tangent/FD
consistency, not recovery of the historical adaptive FEMU information
geometry. The complete machine-readable evidence is
`validation/reference_data/srix_femu_common_path_gate_v9/report.json` and
`jacobians.npz`; the report has `dirty=false` and `p43_authorized=false`.
