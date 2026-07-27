# EBSD-derived structural length preregistration

Date: 2026-07-27

## Scientific role

This analysis estimates a microstructural correlation scale independently of
the FEM/DIC agreement. It does not fit `ell`, and it does not claim that the
measured scale is automatically equal to the micromorphic parameter.

The result will be called an **EBSD/Schmid structural correlation length**.
Only a later unchanged-parameter mechanical test may determine whether it is
a useful closure for the reduced two-dimensional model.

## Frozen field and grid

Primary scalar field:

```text
/schmid/max_schmid_factor
```

from `/home/jeff/CNRS/Theses/Adil/essais/CP_dataset.h5`.

The field is declared co-registered with the DIC crop on a `3600 x 3100`
grid. The physical spacing used for reporting is the DIC spacing,
`1.84 µm/pixel`. Because the native EBSD step and registration procedure are
not archived, this spacing is treated as declared metadata rather than an
independently verified EBSD calibration.

## Frozen validity mask

Accept finite Schmid factors in the physical interval:

```text
0 < max_schmid_factor <= 0.5
```

Pixels outside that interval, and pixels whose three Euler angles are all
zero, are excluded. No interpolation or nearest-neighbour filling is allowed
for the primary calculation.

Annealing twins are retained as orientation discontinuities. They are not
merged, because the target is correlation of a mechanically relevant
orientation descriptor rather than metallurgical equivalent grain diameter.

## Frozen autocorrelation estimator

Let `m` be the validity mask and `z=(s-mean_valid(s))*m`. Compute the
mask-corrected two-point covariance by FFT:

```text
C_raw = ifft2(|fft2(z)|^2)
W     = ifft2(|fft2(m)|^2)
C     = (C_raw / W) / C(0)
```

Only lags with positive pair count are retained. The radial statistic uses
integer-pixel annuli and the pair count as weight. Directional statistics use
the positive x and y axes before radial averaging.

The primary exponential length is obtained from the first contiguous
decreasing branch for which:

```text
0.15 <= C(r) <= 0.60
```

by weighted least squares of `log(C)` against distance. The reported decay
length is `ell_decay=-1/slope`. The fit is invalid if fewer than five lag
values are available or if the fitted slope is non-negative.

## Frozen control statistic

Before the first zero crossing, compute the positive-correlation RMS radius
using the radial area weight:

```text
R_rms = sqrt(sum(r^2 * C(r) * pair_weight * r) /
             sum(C(r) * pair_weight * r))
ell_rms_control = R_rms / 2
```

The factor two follows the second moment of the two-dimensional Helmholtz
Green function. Disagreement between the decay and RMS definitions is
reported, not resolved by selecting whichever is closer to 58.88 µm.

## Anisotropy and uncertainty

- Fit directional decay lengths `ell_x` and `ell_y` with the same correlation
  interval.
- Report `max(ell_x,ell_y)/min(ell_x,ell_y)`.
- Divide the valid field into a fixed `4 x 4` set of non-overlapping spatial
  blocks.
- Compute the radial decay length independently in every valid block.
- Bootstrap the median of the valid block estimates with 10,000 resamples and
  RNG seed 20260727.
- Report the 2.5%, 50% and 97.5% percentiles.

This interval measures spatial heterogeneity between subregions. It is not a
complete EBSD registration or measurement uncertainty.

## Decision boundary

No micromorphic calculation is launched automatically. After this report:

1. the measured scale and its uncertainty are documented;
2. a separate preregistration decides how it may be imposed in mechanics;
3. `ell`, `ell/2` and `2*ell` remain sensitivity cases, not refitted values;
4. failure at the imposed value is reported without moving the EBSD estimate.
