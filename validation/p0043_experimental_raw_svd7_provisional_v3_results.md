# P43 experimental RAW rank-7 exploration — v3

This artifact is exploratory only. It uses the provisional rank-7 SVD basis, the exact physical eta-polytope constraints, raw displacement residuals in mm, and no DIC noise model or covariance.

## Result

The prior RMS displacement is `4.724716887e-6 mm`. The constrained run reaches `4.159220033e-6 mm`, a relative reduction of `11.9689%`.

The optimizer performed 35 objective evaluations and stopped on the configured iteration limit (`status=9`, `success=false`), after approximately 2980.5 s. The unconstrained linear Gauss–Newton diagnostic predicted a 29.1711% reduction; the constrained nonlinear run achieved 11.9689%.

The final point is on several physical bounds (elastic coordinates at their upper bounds, `C44` at its lower bound, `C` at its upper bound, and `d` at its lower bound). Therefore this is not an identified parameter set and must not be used to authorize M100 or experimental claims.

Final physical values were approximately:

```text
tau0 = 25.23996 MPa
R    = 10.41726 MPa
Q    = 12.69528 MPa
b    = 3.79838
C    = 80000.0019 MPa
d    = 499.99999
C11  = 226549.99 MPa
C12  = 143749.99 MPa
C44  = 103699.996 MPa
```

## Status

```text
exploratory_only = true
experimental_parameters_identified = false
global_svd_qualified = false
experimental_m100_authorized = false
```

The prior v1/v2 rectangular-coordinate runs remain historical/superseded; this v3 is the first run using the exact eta-polytope constraints.
