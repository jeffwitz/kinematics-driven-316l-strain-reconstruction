# P43 nested/coupled scaling

The nested and coupled native NumPy SRIX closures were run with the same
identified parameters, order-F EBSD mapping, tangent predictor, `numba-lu12`,
100 local iterations, and 32 load increments.  The M200 trial was stopped
before completion on request and is not used below.

| crop | material points | nested (s) | coupled (s) | speedup coupled | nested GMRES | coupled GMRES |
|---|---:|---:|---:|---:|---:|---:|
| M20 | 800 | 15.793 | 14.325 | 9.3% | 2303 | 2984 |
| M100 | 20,000 | 477.091 | 379.395 | 20.5% | 3390 | 3926 |

The final RAW RMS is identical on M100 (`2.767030392e-5 mm`, absolute
difference `3.4e-15 mm`).  Coupled uses more global Newton iterations (140
versus 124) and GMRES iterations (3926 versus 3390), but its local timing is
lower: 144.8 s for block construction/Schur, 39.7 s for state evaluation,
31.5 s for line search, and 91.3 s for the final tangent.  The nested material
closure reports 446.2 s in plane-stress work.  Thus the local saving dominates
the additional global work at M100.

The M100 verification residuals are `5.13e-7` (coupled) and `5.13e-7`
(nested-scale run); both runs completed all 32 increments with the same final
fields.  These are performance measurements, not a change to the accepted
mechanical tolerance.

The result supports retaining both methods: `nested` remains the portable
reference/default (including MFront), while `coupled` is the native NumPy
high-performance option.  The M100 result is already sufficient to show that
the coupled local formulation becomes more advantageous as the material batch
grows.  A complete M200 comparison should be scheduled separately when the
machine can be left unattended.

Artifacts:

* `reference_data/p0043_scaling_m100_nested_v1/`
* `reference_data/p0043_scaling_m100_coupled_v1/`
