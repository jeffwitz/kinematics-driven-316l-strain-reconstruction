# P43 M100 coupled profile

This profile uses the existing M100 coupled forward artifact; no simulation was
rerun.

| component | seconds | fraction of wall time |
|---|---:|---:|
| coupled state evaluations | 39.675 | 10.5% |
| block construction + Schur | 144.790 | 38.2% |
| local line search | 31.494 | 8.3% |
| final constitutive tangent | 91.339 | 24.1% |
| **material subtotal** | **307.298** | **81.0%** |
| GMRES calls | 26.131 | 6.9% |
| global Jacobian actions | 14.648 | 3.9% |
| preconditioner actions | 6.170 | 1.6% |
| Krylov overhead | 5.314 | 1.4% |
| remaining wall time | 45.837 | 12.1% |
| **wall time** | **379.395** | **100%** |

The M100 coupled path used 140 global Newton iterations and 3926 GMRES
iterations (108 recorded linear solves; the remaining global iterations are
converged/verification evaluations).  The material path is therefore still the
dominant cost at this size, with block construction/Schur and the final tangent
the two largest targets.  FFT/Krylov work is not yet the limiting component.

The next performance work should remain inside the native SRIX material path,
but should target the 12x3 block/Schur and tangent construction rather than
changing global stopping criteria or the constitutive equations.  `nested`
remains the reference; `coupled` is the validated high-performance native path.
