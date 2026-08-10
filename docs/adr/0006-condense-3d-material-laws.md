# ADR 0006 — Condense 3D material laws behind a plane-stress protocol

Date: 2026-07-25  
Status: superseded by the current MFront backend contract

## Context

This decision records the original protocol design. The current implementation
also provides the registered `mfront-structural-plane-stress` backend and the
external 3D condensation reference. The stable formulation and validity domain
are defined in
{doc}`../reference/numerics/mfront_structural_plane_stress`.

## Decision

The global solver depends on a transactional `PlaneStressMaterialBatch`
protocol. A three-dimensional MFront backend integrates a `Tridimensional`
behaviour, locally solves for the three transverse kinematic components, and
returns the constrained tangent. External 3D condensation remains available
as an independent reference.

The finite-element mesh, degrees of freedom, CPS4 operators, global residual,
and global Newton algorithm remain two-dimensional and unchanged. The native
MFront plane-stress path remains available for native 2D laws.

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
- the current structural backend contract is documented separately and is
  qualified for the registered crystal behaviours;
- finite-strain multiplicative kinematics remain outside this interface.
