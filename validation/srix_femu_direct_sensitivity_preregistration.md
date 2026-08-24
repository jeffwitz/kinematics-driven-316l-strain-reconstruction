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
`h in {1e-2, 3e-3, 1e-3}`. For the primary fixed-path oracle, the accepted base
subincrements, plane-stress closure and boundary sensitivities are held fixed;
the adaptive-path FD is retained only as provenance for the secondary
comparison.

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

For the M8 gate, the direct implementation must therefore reuse the exact
`TraditionalTwoStateTriangleBatch.tangent_action_into`,
`TwoSubcellDiagnostic2D.divergence_from_sample_stress_into`, Dirichlet
extension, packing and matrix-free Krylov conventions used by
`solve_two_state_dirichlet_plane_stress`. It must not call
`TensorPlasticObservabilityOperator`, `weak_equilibrium_residual`,
`_assemble_sparse_stiffness`, or any other REGM mechanical surrogate for the
global sensitivity operator. A dense or sparse `K_II` may be introduced only
after an explicit equivalence test against the reference matrix-free action.

The archived adaptive FEMU FD is not the primary equality oracle: the audit in
`validation/srix_femu_fd_adaptive_path_audit.md` shows that the `+h/-h`
trajectories take different accepted subincrements. The primary qualification
is therefore raw-column equality to a second central-FD oracle built by freezing
the accepted base `LoadPathStep` sequence for both signs. The archived adaptive
FD remains a secondary diagnostic and must be reported separately. The four
column relative L2 errors and cosines are the first gate; the singular spectrum
is a consequence, not a replacement for this test.

Analytical SRIX/MFront directional derivatives are explicitly out of scope
until the shadow provider passes these gates.

The preceding REGM diagnostics are now closed. In particular, the corrected
cumulative endpoint audit is recorded as `E-SRIX-REGM-009` and does not
authorize another REGM variant or any P43 identification. The next gate is
`E-SRIX-FEMU-DIRECT-001`: direct differentiated FEMU on the exact M8 solver.
