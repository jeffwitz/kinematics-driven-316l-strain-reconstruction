# Monolithic structural plane stress for SRIX

The registered SRIX structural behaviour applies a three-component structural
plane-stress closure inside the MFront implicit constitutive solve. The host
passes the imposed gradient in the structural/global frame and supplies the
orientation as per-point material properties. It does not perform a second
rotation or a separate transverse closure.

## Local problem

The local SRIX system retains six elastic-strain components and twelve signed
slip increments. The in-plane elastic rows retain their kinematic meaning. The
three transverse rows impose

$$
\sigma_{zz}=\sigma_{xz}=\sigma_{yz}=0
$$

in the structural frame. The relaxed transverse strains are local constitutive
outputs; they are not additional FEM unknowns.

The generic formulation is described in
{doc}`../../reference/numerics/mfront_structural_plane_stress`. The SRIX
behaviour is one registered instance of that contract; its constitutive flow
rule remains entirely separate from the closure.

## Tangent and host integration

The MFront behaviour returns the tangent of the one-step constrained map. If a
material point is integrated through several host substeps, the host may
replace that one-step tangent with the tangent of the composed algorithmic map.
This distinction is essential for a global Newton solve and is independent of
the SRIX constitutive equations.

The external three-dimensional condensation backend is retained as an
independent reference. Agreement is expected when both routes start from the
same committed state, use the same orientation and branch, and integrate the
same increment.

## Validity domain

The registered route is qualified for small-strain, `Implicit`,
`Tridimensional`, standard-elasticity-compatible crystal behaviours with the
repository's structural orientation contract. It is not a general finite-
strain plane-stress formulation and does not resolve through-thickness
heterogeneity or warping.
