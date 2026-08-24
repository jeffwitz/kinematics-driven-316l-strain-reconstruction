# Audit — exact global operator for direct FEMU sensitivities

Date: 2026-08-24  
Scope: M8 SRIX twin only; no new mechanical run and no P43.

## Finding

The M8 reference forward is not the sparse REGM operator. It is the matrix-free
solver `solve_two_state_dirichlet_plane_stress` in
`src/fem_inhouse/spectral2d/newton_two_state.py`.

At every accepted Newton state it uses:

1. `TraditionalTwoStateTriangleBatch.evaluate_samples` to obtain the
   plane-stress stress and consistent in-plane tangent;
2. `TwoSubcellDiagnostic2D.divergence_from_sample_stress_into` for the global
   internal-force residual;
3. `TraditionalTwoStateTriangleBatch.tangent_action_into` for the exact
   matrix-free tangent action;
4. `pack_interior`/`unpack_interior` with the harmonic full-Dirichlet
   extension;
5. nonsymmetric GMRES/LGMRES/GCROT with the EBI Green preconditioner.

The forward residual is therefore the packed interior divergence of the
current TRI2 stress. Its tangent action is the same divergence applied to the
constitutive consistent tangent times the kinematic strain increment. No
dense or sparse global `K_II` is assembled in this path.

## Consequence for the direct sensitivity gate

The direct sensitivity implementation must call or factor the same matrix-free
action. It must not use:

```text
TensorPlasticObservabilityOperator
weak_equilibrium_residual
_assemble_sparse_stiffness
```

Those are the mechanical REGM surrogate used by the closed diagnostics
`E-SRIX-REGM-008` and `E-SRIX-REGM-009`. Reusing them would reproduce the wrong
global Jacobian even if the MFront tangent and constitutive history were exact.

The first implementation should consequently solve one matrix-free linear
system per sensitivity column (or introduce a multi-RHS wrapper around the
same action only after a numerical equivalence test). The four columns must be
compared directly with the archived `FEMU_observed` FD Jacobian before any SVD
or parameter interpretation.

## Boundary and history facts

The M8 twin uses exact prescribed boundary values through the harmonic
Dirichlet extension. Boundary sensitivities are zero because the twin boundary
history is independent of the SRIX parameters. Constitutive states are
committed only after an increment reaches the forward convergence criterion;
adaptive subincrements must be replayed in the same accepted order by any
shadow provider.

This audit does not implement the sensitivity driver and does not authorize
P43. The implementation contract is frozen in
`validation/srix_femu_direct_sensitivity_preregistration.md`.
