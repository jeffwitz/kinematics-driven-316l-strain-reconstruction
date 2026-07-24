# ADR 0006 — Condense 3D material laws behind a plane-stress protocol

Date: 2026-07-25  
Status: accepted

## Context

The native MFront `PlaneStress` J2 behaviour is correct and efficient for the
current isotropic case. Its four-component interface does not represent all
couplings of a generally oriented three-dimensional crystal-plasticity law.
Putting those details into the global finite-element Newton loop would couple
the solver to one material model.

## Decision

The global solver depends on a transactional `PlaneStressMaterialBatch`
protocol. A second MFront backend integrates a `Tridimensional` behaviour,
locally solves for `[epsilon33,gamma13,gamma23]`, and condenses the 6×6
algorithmic tangent by a Schur complement.

The finite-element mesh, degrees of freedom, CPS4 operators, global residual,
and global Newton algorithm remain two-dimensional and unchanged. The native
MFront plane-stress path remains the production default for J2.

Analytical out-of-plane completion is an explicit
`j2_isotropic_analytical` capability. It is not a generic fallback for an
unknown MFront behaviour.

## Consequences

- a small-strain 3D material law can be substituted without modifying global
  Newton;
- local and global trial states remain transactionally separate;
- all three transverse residual components are persisted and checked at Gauss
  points;
- the condensed tangent must pass finite-difference verification;
- the current validation establishes the adapter with J2, not a future
  crystal-plasticity law;
- finite-strain multiplicative kinematics remain outside this interface.
