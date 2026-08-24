# Preregistration — direct FEMU sensitivities for SRIX

This gate is frozen before implementation. It targets the exact M8 twin only;
no P43 calculation, optimization, or change to REGM is allowed.

## Primary question

Can one converged nonlinear FEMU trajectory plus four constitutive shadow
histories and four linear sensitivity right-hand sides reproduce the archived
observed-FEMU Jacobian from `b7ff50e`? The reference is
`validation/reference_data/srix_regm_information_geometry_v1/jacobians.npz`,
entry `FEMU_observed`, with log parameters
`(tau0, R, Q, b)`, central step `h=3e-3`, eight registered macro endpoints,
and affine-preserving observation.

## Frozen gates

Before SVD, each direct column must have cosine greater than `0.999` and
relative L2 error below `2 %` against the archived central-FD column; the
global relative Frobenius error must remain below `2 %`. The normalized spectrum
must reproduce approximately `(1, 0.542, 0.407, 0.0679)`. Principal angles of
rank 1, 2 and 3 subspaces are reported; the target is below `2 degrees` for
each. A failure is reported as a negative result and does not authorize P43.

The finite-difference constitutive-shadow step study is frozen at
`h in {1e-2, 3e-3, 1e-3}`. Accepted forward subincrements, cutbacks, plane
stress closure and boundary sensitivities must match the reference trajectory.

## Method boundary

The first provider is central constitutive shadows. For each parameter and
sign, a persistent MFront batch is evaluated at the unperturbed current strain
to obtain the history-aware stress forcing, then reverted. After the global
sensitivity solve, it is advanced at the perturbed current strain and
committed. Shadows are never reinitialized between increments.

The global driver must use the same tangent action and boundary formulation as
the reference FEMU. The archived M8 forward solver is matrix-free spectral
TRI2 with GMRES and an EBI Green preconditioner; a sparse `B^T C B` solve is the
REGM operator, not automatically the reference FEMU tangent. This distinction
is an audit requirement: using the sparse operator is allowed only as a
separately labelled diagnostic and cannot be claimed to reproduce FEMU.

Analytical SRIX/MFront directional derivatives are explicitly out of scope
until the shadow provider passes these gates.
