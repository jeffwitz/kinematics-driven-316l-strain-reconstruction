# P43 — direct coupled plane-stress tangent

The native NumPy coupled SRIX closure now differentiates the converged local
system `[R_gamma, sigma_transverse] = 0` directly.  The existing 3-D
`tangent_from_trial()` plus plane-stress condensation remains the oracle.

## Qualification

* `tests/unit/core/test_srix_numpy.py`: 11/11 tests pass.
* Local and two-step plastic tests compare the direct tangent with the 3-D
  oracle at relative error below `2e-16` for the checked states.
* M20 fields reproduce the previous coupled run: displacement max difference
  `2.8e-17 mm`, EVM max difference `6.3e-15`.

## M100 forward

| quantity | previous coupled | direct tangent |
|---|---:|---:|
| wall time (s) | 341.456 | 324.269 |
| global Newton iterations | 140 | 140 |
| GMRES iterations | 3926 | 3926 |
| RAW RMS (mm) | 2.767030391669e-05 | 2.767030391669e-05 |
| verification residual | — | 5.13e-07 |

The direct tangent reduces wall time by approximately 5.0% on this run.  The
M100 fields remain numerically identical to the previous coupled result:
displacement max difference `2.8e-17 mm`, in-plane stress max difference
`4.5e-09 MPa`, and EVM max difference `9.3e-15`.

The run artifact is:
`validation/reference_data/p0043_m100_numpy_srix_direct_tangent_v1/report.json`.

The direct path is qualified as numerically equivalent to the existing coupled
oracle.  The nested closure remains available and unchanged as the reference
algorithm.
