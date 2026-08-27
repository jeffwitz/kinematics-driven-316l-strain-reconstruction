# P43 M100 coupled-kernel optimization

The cached invariant blocks, single-factorization multi-RHS solve, and active
line-search compression were benchmarked against the archived M100 coupled
run.  Inputs, parameters, F-order EBSD mapping, 32 increments, and solver
controls are unchanged.

| quantity | baseline coupled | optimized coupled |
|---|---:|---:|
| wall time (s) | 379.395 | 341.456 |
| global Newton iterations | 140 | 140 |
| GMRES iterations | 3926 | 3926 |
| RAW RMS (mm) | 2.767030391669e-5 | 2.767030391669e-5 |
| block + Schur (s) | 144.790 | 107.560 |
| state evaluation (s) | 39.675 | 37.107 |
| line search (s) | 31.494 | 31.852 |
| final tangent (s) | 91.339 | 92.708 |

The optimized run is `10.0%` faster while producing the same global trajectory
and the same displacement solution.  The block/Schur stage is `25.7%` faster,
which accounts for essentially all of the wall-time gain.  The line-search
compression does not yet reduce its measured time on this case, despite
removing about 2.4% of candidate evaluations; it should therefore not be
optimized further without a new profile.

The coupled path remains the validated high-performance native option, with
`nested` retained as the reference/default.  No M200 run is included.
