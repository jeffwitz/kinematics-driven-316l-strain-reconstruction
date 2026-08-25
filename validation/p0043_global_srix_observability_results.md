# P43 global SRIX observability — smoke result

## Status

This is a two-point smoke test (the prior plus one Sobol point), not the
pre-registered eight-point campaign and not an identification result.

The raw displacement output is in millimetres and no DIC noise model or
covariance is used.  The nine coordinates are admissible logarithmic
coordinates:

```text
log(C11-C12), log(C11+2 C12), log(C44), log(tau0), log(R),
log(Q), log(b), log(C), log(d)
```

## Corrected local SVD

The original smoke calculation archived the Jacobians correctly but had a
reporting bug: the SVD left factor `U` was stored under `singular_values`.
The probe now uses the correct `(U, S, Vh)` assignment, and the report was
rebuilt from the archived Jacobians (`report_schema_version=2`).

Normalized local spectra (`S / S[0]`) are:

```text
prior:  1, 0.52172, 0.39568, 0.12307, 0.08011, 0.07525,
        0.03130, 4.00e-4, 8.998e-5
Sobol:  1, 0.33944, 0.28179, 0.11727, 0.07378, 0.04880,
        0.01927, 4.27e-4, 3.052e-5
```

The local rank-3-to-global principal angles (degrees) are:

```text
prior:  6.62, 1.28, 0.45
Sobol:  4.95, 2.15, 0.48
```

The local observable subspace therefore moves over this first two-point
sample; this is a reason to complete the prescribed sample campaign before
freezing a nine-parameter reduced basis.

## Aggregate smoke Hessian

Normalized eigenvalues of `sum_k J_k.T @ J_k` are:

```text
1, 1.776e-1, 1.121e-1, 2.060e-2, 6.885e-3,
3.824e-3, 6.590e-4, 1.825e-7, 5.894e-9
```

The weakest aggregate direction is numerically aligned with the
`log(Q)-log(b)` contrast (absolute alignment about 1.000 in the smoke
aggregate).  This is a diagnostic observation only: it does not authorize a
fixed rank or a parameter identification.

## Decision

```text
global_observability_smoke_valid = true
global_observability_campaign_complete = false
parameter_identification_authorized = false
```

Next step: run the pre-registered eight-point Sobol campaign with the corrected
reporting code, then assess stability of the aggregate and local observable
subspaces before any nine-parameter inverse fit.
