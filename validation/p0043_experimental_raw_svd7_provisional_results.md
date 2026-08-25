# P43 experimental RAW M20 — provisional SVD-rank-7 exploration

## Status

This run is exploratory only. It uses the two-point smoke aggregate SVD basis,
not a globally qualified Sobol basis. No measurement-noise model, covariance,
whitening, regularisation, M100 scale-up, or identification claim is allowed.

## Result

The raw displacement RMS decreases from

```text
4.724716887e-6 mm  ->  4.671270569e-6 mm
```

which is a relative decrease of approximately **1.13%**. The optimizer stopped
after two evaluations because the projected gradient tolerance was met.

The final provisional parameters are:

```text
C11 = 193428.23 MPa    C12 = 116986.62 MPa    C44 = 124613.59 MPa
tau0 = 38.6449 MPa      R = 18.6021 MPa         Q = 9.5200 MPa
b = 2.8576              C = 42351.43 MPa        d = 1359.56
```

All remain inside the conservative transformed Sobol domain. The seven
reduced coordinates are approximately

```text
(+0.05997, -0.06411, +0.05859, -0.04041,
 -0.06148, -0.06560, -0.06466)
```

## Interpretation

The extra elastic, threshold, isotropic-hardening, and kinematic directions do
not produce a large raw-displacement improvement from the nominal prior in
this first exploratory run. This is not yet evidence that the constitutive
family is inadequate: the basis is provisional, the optimizer stopped at the
projected-gradient criterion after only two evaluations, and the global Sobol
qualification is still pending.

```text
exploratory_run_completed = true
global_svd_qualified = false
experimental_parameters_identified = false
experimental_m100_authorized = false
```
