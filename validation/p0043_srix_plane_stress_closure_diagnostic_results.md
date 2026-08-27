# P43 nested/coupled closure diagnostic

This is a diagnostic-only comparison on the same P43 M20 setup, with order-F
EBSD mapping, identical parameters, tangent predictor, and `numba-lu12`.
No constitutive or solver algorithm was changed for this comparison.

| quantity | nested | coupled |
|---|---:|---:|
| wall time (s) | 15.793 | 14.325 |
| global Newton iterations | 121 | 146 |
| GMRES iterations | 2303 | 2984 |
| GMRES / global Newton | 25.88 | 26.18 |
| RAW RMS (mm) | 3.952271743e-6 | 3.952271743e-6 |
| verification residual | 6.998e-9 | 3.070e-12 |

The extra GMRES work is therefore explained by more global Newton iterations
(146 versus 121), not by a large change in GMRES iterations per global step.
The coupled local path made 207 calls, accumulated 630161 point-iterations,
and performed 466961 line-search candidate evaluations; 464558 accepted on the
full step. Its measured local timing was 0.779 s in state evaluation, 5.292 s
in block construction/Schur solves, 0.933 s in line search, and 3.030 s in the
final tangent.

At identical local strain paths on an independent eight-state check, the
relative Frobenius difference of the condensed tangents was at most
`5.6e-10`; the active plastic-slip masks agreed for all `1536` tested
point-system entries. The coupled and nested displacement fields in the full
forward remain equal to `1.94e-12 mm` maximum absolute difference.

Conclusion: the coupled closure is numerically equivalent to nested, but this
run does not demonstrate a robust speedup. Keep `nested` as the default and
use the diagnostic artifacts to guide any later performance work.
