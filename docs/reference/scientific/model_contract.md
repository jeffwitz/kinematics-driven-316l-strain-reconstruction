# Mechanical model contract

**Mode:** reference  
**Domain:** scientific

The maintained case-study solver is a two-dimensional, infinitesimal-strain
Q4 formulation with prescribed DIC displacement on the solved boundary.  The
global unknown is the in-plane displacement; the material interface returns an
in-plane stress and tangent under structural plane stress.

The constitutive slot may contain isotropic J2/Ludwik, MFront crystal laws or
the native SRIX backend.  The global solver does not add an out-of-plane degree
of freedom, contact, dynamics, finite transformations or thickness variation.
Partition ownership, padding and provenance are defined by the campaign
manifest.  This is a case-study contract, not a general-purpose Abaqus
replacement.

See {doc}`../../reference/numerics/plane_stress` for the local reduction and
{doc}`constitutive_models` for the supported constitutive families.
