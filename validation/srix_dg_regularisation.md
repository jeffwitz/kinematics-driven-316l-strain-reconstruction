# Compact regularisation of SRIX `abs(dg)` branches

This is an experimental numerical study. The Macaulay regularisation is
disabled (`SrixSmoothingEpsilon = 0`); only `abs(dg)` and its derivative are
regularised inside a compact C2 band of width `SrixSlipSmoothingDelta`.

## Local failure replay

The diagnostic archive from `3d90d70` contains 380 isolated failures at 50
points. Among 4560 recorded slip increments, 2846 are exactly zero. The
non-zero quantiles are `8.43e-6` (1%), `2.25e-5` (5%), `2.83e-3` (median),
and `1.09e-2` (90%).

Replaying exactly the 380 archived trials gives:

| `delta` | failed | rescued | new failures |
|---:|---:|---:|---:|
| 0 | 380 | 0 | 0 |
| 1e-5 | 0 | 380 | 0 |
| 3e-5 | 0 | 380 | 0 |
| 1e-4 | 0 | 380 | 0 |
| 3e-4 | 0 | 380 | 0 |

This is a strong local signal, but it does not prove that the regularised
solution is physically equivalent to the historical one.

As a separate semismooth control, keeping `delta=0` and changing only the
subgradient at exactly `dg=0` from the historical `-1` to `0` also rescues
`380/380` archived trials. This control is recorded in
`srix_dg_regularisation_zero_derivative_replay.json`.

## M20 screening

On the P43 M20 EBSD case, `delta=1e-5` removed the 23 sub-stepped points in
the current replay and required 46 global Newton iterations, versus 45 for
the unregularised run. Relative to that run, the maximum relative field
changes were `1.11e-8` for displacement, `4.16e-5` for stress, and about
`1.0e-4` for signed slip. The larger candidates changed the fields more.

## M200 result

The single qualifying run used P43 M200 EBSD, crop `[1520:1720] x [985:1185]`,
eight increments, the structural-plane-stress backend, four MFront threads,
BLAS/FFTW/OpenMP pinned to one thread, and `delta=1e-5`.

| quantity | historical `delta=0` | `delta=1e-5` |
|---|---:|---:|
| global Newton | 58 | 56 |
| Newton per increment | `[6,6,7,7,7,7,8,10]` | `[6,6,7,7,7,7,8,8]` |
| sub-stepped points | 978 | 0 |
| composite-FD points | 978 | 0 |
| composite-FD trajectories | 5868 | 0 |
| final residual | `8.62e-9` | `9.89e-12` |
| elapsed time, single run | `305.02 s` | `116.30 s` |

The regularised run therefore removes the expensive substepping path and is
substantially faster in this single measurement. This is not a repeated
performance benchmark, and the time must not yet be treated as a qualified
speedup.

Compared with the historical fields, the regularised run gives relative L2
differences of `5.46e-8` in displacement, `3.92e-5` in in-plane stress,
`2.02e-5` in reactions, `1.83e-4` in signed slip, `2.48e-4` in equivalent
slip, and `1.66e-4` in accumulated slip. The maximum absolute stress
difference is `0.332 MPa`.

These differences are small in engineering terms but are not numerical noise.
The experiment therefore demonstrates a robust local solver effect, not yet a
validated replacement law. The candidate must remain experimental until the
regularised tangent is checked by direct finite differences in the compact band
and the `sign(0)=0` semismooth control is separately quantified.

The `sign(0)=0` M200 control was also run. It produced 56 Newton iterations,
`[6,6,7,7,7,7,8,8]`, no substepped points, no composite-FD trajectories, a
final residual of `1.85e-10`, and `130.50 s` wall time. Its field differences
versus the historical run were `5.37e-8` displacement, `3.86e-5` stress,
`1.64e-5` reactions, `1.82e-4` signed slip, and `1.21e-4` accumulated slip.
Thus the semismooth control and the compact `delta=1e-5` run lead to the same
qualitative M200 conclusion; neither is yet a validated replacement law.

## Reproducibility

The replay and screening artefacts are:

- `validation/_generated/performance/srix_dg_regularisation_failure_replay.json`
- `validation/_generated/performance/srix_dg_regularisation_zero_derivative_replay.json`
- `validation/_generated/performance/srix_dg_regularisation_sweep.json`
- `validation/_generated/performance/srix_dg_regularisation_m20_delta_*.json`
- `validation/_generated/performance/srix_dg_regularisation_m200.json`

The implementation is in both SRIX MFront behaviours and the option is
forwarded by `plane_stress_material.py`. `delta=0` preserves the historical
`abs/sign` constitutive branch, including its current inactive-system
derivative convention.
