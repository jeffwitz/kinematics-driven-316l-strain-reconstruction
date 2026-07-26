# Model contract

**Category: Reference.**

## Supported problem

| Item | Contract |
|---|---|
| geometry | structured rectangular two-dimensional grid |
| element | bilinear Q4, four Gauss points |
| kinematics | infinitesimal strain |
| mechanical hypothesis | plane stress |
| mechanical unknowns | in-plane nodal displacements |
| loading | DIC displacement prescribed on the solved boundary |
| local law | isotropic associative J2 with regularized Ludwik hardening |
| nominal backend | analytical MFront native plane stress |
| sparse solver | PyPardiso/MKL required for production |

The solver does not contain an out-of-plane degree of freedom, a 3D element,
contact, dynamics, finite transformations or thickness variation.

## Partition contract

A partition contains a **core**, which uniquely owns its output, and optional
padding, which provides a solved neighbourhood. The full padded region is
solved. Only the core contributes to stitched fields and scientific metrics.
Partition bounds and padding are read from the immutable campaign manifest.

For nonlocal calculations, zero micromorphic flux is imposed at the padded
boundary. The required padding-to-length ratio belongs to the campaign
configuration and must be recorded.

## Constitutive replacement boundary

The global solver consumes an in-plane stress and tangent from a
plane-stress-material protocol. Native plane-stress MFront and a locally
condensed three-dimensional MFront behaviour implement that protocol. A new
3D law may replace J2 inside the adapter without adding mechanical degrees of
freedom.

## Claim boundary

This contract is intentionally case-study-specific. It is not a
general-purpose Abaqus replacement.
