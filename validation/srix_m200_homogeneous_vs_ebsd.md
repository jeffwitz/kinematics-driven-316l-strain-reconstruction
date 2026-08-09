# P43 M200 SRIX: homogeneous orientation versus EBSD

This report compares the same P43 M200 calculation with the same DIC-derived
Dirichlet boundary history, material parameters, eight increments, solver
configuration, and `mfront-structural-plane-stress` backend. The only physical
change is the orientation field: one Bunge orientation `[35, 20, 15]` degrees
versus the co-registered EBSD map.

The run used four MFront threads, one Krylov BLAS thread, one FFTW thread,
LGMRES with Eisenstat--Walker forcing, and the qualified composite tangent
option. The raw run reports and fields are kept alongside this report.

## Solver and timing comparison

| Quantity | Homogeneous | EBSD |
|---|---:|---:|
| elapsed time (s) | 90.06 | 305.02 |
| Newton iterations | 42 | 58 |
| iterations per increment | `[6,5,5,5,5,5,5,6]` | `[6,6,7,7,7,7,8,10]` |
| final residual | 1.16e-11 | 8.62e-9 |
| material time (s) | 61.61 | 256.53 |
| material integration (s) | 46.29 | 233.54 |
| Krylov (s) | 17.92 | 40.58 |
| Krylov overhead (s) | 1.54 | 5.16 |
| Jacobian (s) | 12.11 | 25.80 |
| preconditioner (s) | 4.28 | 9.62 |
| substepped point occurrences | 0 | 978 |
| composite-FD points | 0 | 978 |
| composite-FD trajectories | 0 | 5868 |
| composite-FD time (s) | 0 | 7.33 |
| FD partition changes | 0 | 22 |

The EBSD calculation is therefore about 3.39 times slower in this single
controlled run. Most of the difference is constitutive integration and the
additional global Newton work; the composite FD itself is only 7.33 s of the
305.02 s total. Its six trajectories per selected point are explicit in the
telemetry. The 22 partition changes are recorded as an algorithmic branch
diagnostic; they do not invalidate the converged field comparison.

## Field changes caused by EBSD

The following values compare EBSD fields against the homogeneous fields on the
same M200 grid.

| Observable | Relative L2 / statistic |
|---|---:|
| displacement | 1.09e-3 |
| in-plane stress | 4.40e-1 |
| reaction forces | 5.08e-1 |
| accumulated slip | 4.28e-1 |
| signed plastic slip | 1.38 |
| equivalent plastic slip | 1.01 |
| accumulated-slip spatial correlation | 0.617 |
| maximum-system-slip spatial correlation | 0.507 |
| dominant-system agreement | 8.34% |
| top-5% accumulated-slip Jaccard | 0.176 |
| top-5% maximum-slip Jaccard | 0.176 |

Accumulated-slip quantiles are:

| Quantile | Homogeneous | EBSD |
|---|---:|---:|
| median | 1.192e-2 | 7.275e-3 |
| 90% | 1.439e-2 | 1.278e-2 |
| 95% | 1.677e-2 | 1.484e-2 |
| 99% | 2.140e-2 | 2.092e-2 |
| maximum | 3.328e-2 | 5.351e-2 |

The EBSD field lowers the median accumulated slip but raises its maximum,
which is consistent with stronger localisation. The low dominant-system
agreement and low top-5% overlap show that the change is not a simple scalar
rescaling of the homogeneous response.

## Reconstruction of the measured displacement field

The full DIC displacement field is used only as an observation for this
analysis; the solver imposes its boundary values as Dirichlet data. The
interior displacement errors are:

| Case | Interior RMSE (mm) | Interior relative L2 |
|---|---:|---:|
| Homogeneous | 4.81e-5 | 9.81e-4 |
| EBSD | 6.81e-5 | 1.39e-3 |

Thus the EBSD calculation changes the reconstructed interior kinematics by a
measurable amount, while the boundary history remains identical. This is an
observation about this constitutive calibration and discretisation; it is not
yet a claim that EBSD is statistically superior without uncertainty or
registration analysis.

## Reproducibility

- Raw homogeneous run: `srix_p43_m200_homogeneous_structural_fd.json`
- Raw EBSD run: `srix_p43_m200_ebsd_structural_fd.json`
- Field arrays: matching `.fields.npz` files
- Machine-readable comparison: `srix_p43_m200_homogeneous_vs_ebsd.json`
- Analysis script: `scripts/analyse_srix_m200_homogeneous_vs_ebsd.py`

The EBSD source and crop provenance are recorded in the raw JSON. The
comparison is a single campaign, not a repeated timing sweep; the elapsed
times should therefore be treated as campaign measurements rather than
universal performance guarantees.
