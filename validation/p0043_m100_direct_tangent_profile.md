# P43 M100 direct-tangent profile

This is a read-only comparison of the two already archived M100 coupled
runs; no new forward or tolerance change was made.

## Strict verification comparison

| quantity | coupled optimized (oracle tangent) | direct tangent |
|---|---:|---:|
| wall time (s) | 341.456334 | 324.269487 |
| verification residual | 5.12889812e-07 | 5.12889764e-07 |
| absolute difference | — | 4.77e-14 |
| global Newton | 140 | 140 |
| GMRES | 3926 | 3926 |
| RAW RMS (mm) | 2.767030391669e-05 | 2.767030391669e-05 |

The verification residual is therefore unchanged to reporting precision.  The
field comparison is also numerical noise: displacement max difference
`2.8e-17 mm`, stress max difference `4.5e-9 MPa`, and EVM max difference
`9.3e-15`.

## Direct-run counters

The direct run reports:

| counter | seconds |
|---|---:|
| coupled state | 51.1740 |
| A/B + Schur blocks | 109.4287 |
| line search | 39.0696 |
| direct tangent | 51.3022 |

These counters are not mutually exclusive: state evaluations are included in
line-search activity.  They must not be summed as a wall-time decomposition.
The residual wall-time remainder includes global FFT/GMRES and other solver
work; the per-solve GMRES counters sum to 27.51 s (with 15.39 s recorded for
global Jacobian assembly).

Conclusion: the `5.13e-7` verification residual is not a regression caused by
the direct tangent.  The direct path preserves the same final state and global
trajectory while reducing wall time by 17.19 s (5.03%).
