# P43 global SRIX observability — preregistration

This gate studies the complete local SRIX vector without changing the FCC
interaction matrix or slip systems:

```text
(C11, C12, C44, tau0, R, Q, b, C, d)
```

The optimization coordinates are the admissible logarithmic coordinates

```text
log(C11-C12), log(C11+2*C12), log(C44),
log(tau0), log(R), log(Q), log(b), log(C), log(d)
```

so cubic stability is preserved by construction. Runtime overrides are passed
through the existing SRIX parameter resolver; presets are not modified.

## Scope and ordering

1. Implement and unit-test the coordinate transform.
2. Run a small synthetic M20 feasibility check before any experimental run.
3. At each sampled point, compute the raw displacement-output sensitivity in
   physical millimetres. No DIC noise weighting or covariance is used.
4. Aggregate `H = sum_k J_k.T @ J_k` over a preregistered parameter domain and
   diagonalize it.
5. Report local spectra, global eigenvalues, principal-angle stability and
   parameter combinations before attempting a reduced identification.

The first implementation uses central finite differences of the complete
forward output for the new elasticity and `(C,d)` directions; the qualified
four-parameter direct sensitivity remains the reference for the existing
`tau0,R,Q,b` directions. This is an audit/prototype, not an authorization for
a large experimental campaign.

## Exploratory domain

The initial domain is a numerical observability study around the registered
transposed SRIX prior. Bounds are recorded before sampling:

```text
C11-C12, C11+2*C12, C44: multiplicative factor [0.85, 1.15]
tau0: [25, 60] MPa
R:    [8, 35] MPa
Q:    [4, 20] MPa
b:    [1, 6]
C:    [10000, 80000] MPa
d:    [500, 3000]
```

These are exploratory numerical bounds and carry no material-identification
claim. A Sobol/LHS sample count of 8 is the first smoke gate; 20–50 points are
allowed only after the smoke gate and timing are reviewed.

## Claims prohibited before the gate

No parameter is called identified. No P43 experimental optimization is
authorized by this preregistration. The global spectrum may only be used to
define an observable subspace if the synthetic consistency and subspace-angle
checks pass.
