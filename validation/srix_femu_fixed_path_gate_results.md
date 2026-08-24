# Fixed-path FEMU sensitivity gate — blocked oracle

This report records the common-path finite-difference attempts for the M8 twin.
It does not authorize P43 and does not qualify the direct sensitivity method.

The archived adaptive finite difference cannot be used as the primary oracle:
the base accepted trajectory has 338 increments, whereas the `Q+` and `Q-`
trajectories have 326 and 328. The direct method is therefore compared against
a finite difference on the base `LoadPathStep` sequence.

| log step `h` | fixed-path result | first failure |
|---:|---|---:|
| `3e-3` | blocked | increment 18 |
| `1e-3` | blocked | increment 5 |
| `1e-4` | blocked | increment 12 |

A uniformly refined common path (two substeps per accepted base increment) was
also tested at `h=1e-3`. It reached 676 prescribed increments but blocked at
increment 407. Its machine-readable status is in
`validation/reference_data/srix_femu_fixed_path_gate_ref2_h1e3_v1/report.json`.

The perturbed runs used the same boundary path, plane-stress backend and
Newton formulation as the base solver, with a larger iteration allowance
(80 Newton iterations, 20 line-search reductions) and the first base
displacement as an initial predictor. The failures are therefore reported as a
numerical robustness problem of the fixed-path FD oracle, not converted into a
claim that the derivative is wrong.

The already archived direct-versus-adaptive comparison is explicitly secondary:
its raw column relative errors are `(0.942, 0.967, 0.997, 0.998)` and its
cosines are `(0.406, 0.307, 0.168, 0.258)`. Those numbers are not a direct
sensitivity gate because the reference perturbations change the accepted path.

Next step: build a common refined path (or diagnose the branch at the failing
increments), then rerun the raw-column comparison before any SVD or parameter
identification.
