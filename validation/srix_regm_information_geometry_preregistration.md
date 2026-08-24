# Preregistration — information geometry of SRIX-REGM and FEMU

## Question

On the exact M8 SRIX twin, compare the local information carried by:

1. REGM replay of the exact mechanical history;
2. REGM replay after the qualified DIC transfer;
3. the observed displacement residual of a complete forward FEMU solve.

The goal is to distinguish intrinsic parameter non-identifiability from loss
caused by the REGM surrogate or by injecting the observation operator before
constitutive replay.

## Fixed setup

- parameters: `eta = log(tau0, R, Q, b)`;
- base point: the registered SRIX twin preset;
- central finite-difference step: `3e-3` in log coordinates;
- observation: the affine-preserving qualified transfer;
- whitening: identity for the transfer-only comparison;
- states: the eight registered macro-endpoint states;
- no P43 data and no parameter optimization.

The FEMU Jacobian is computed from eight complete forward solves at
`eta +/- 3e-3 e_i`. Its residual is the observed displacement difference at
the same macro endpoints. It is not replaced by a REGM approximation.

## Quantities to report

For each Jacobian `S`, report `S.T @ S`, singular values, normalized singular
values, right singular vectors, numerical rank, pseudo-inverse covariance and
parameter correlations in the identifiable subspace. Report principal angles
between the right-singular subspaces and cumulative spectra as loading states
are added.

The covariance is a Gauss–Newton geometry diagnostic, not a calibrated
uncertainty estimate: no experimental noise scale is inferred from this twin.

## Decision rule

Do not launch P43. The result is diagnostic even if the three spectra disagree.
The only positive conclusion allowed is that a direction is visible in a given
synthetic objective. A small singular value is reported as non-identifiable,
not fixed by a prior and called measured.
