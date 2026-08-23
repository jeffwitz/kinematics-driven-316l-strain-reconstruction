# Current scientific evidence

This page summarises the evidence currently supporting the 316L
kinematics-driven reconstruction workflow. It is organised by claim, not by
development sequence.

## Measurement and boundary data

The DIC pipeline supplies the measured boundary kinematics and the associated
quality and convention metadata. The EBSD provider supplies co-registered
crystal orientations at material points. Their coordinate, indexing, and
provenance contracts are documented in the corresponding reference pages.

The interior displacement field remains a model prediction unless a case
explicitly imposes additional observations. Boundary agreement therefore does
not by itself establish interior agreement.

## Constitutive evidence

The qualified production crystal route is the MFront SRIX behaviour with the
structural three-traction plane-stress closure. The independent reference is
external three-dimensional condensation. The same host adapter supports the
registered structural SRIX and Méric–Cailletaud behaviours.

At material-point level, the structural closure has been checked against the
three-dimensional Schur complement for rotated elasticity, J2, SRIX, and
Méric–Cailletaud. The assumptions and validity domain are stated in
{doc}`../reference/numerics/mfront_structural_plane_stress`.

## Solver evidence

The spectral solver uses Newton iterations with a Krylov linear solve and
explicitly controlled thread pools. Constitutive trial states are transactional:
failed evaluations are reverted and accepted states are committed only after
the global step succeeds.

For points that require constitutive substeps, the host may use the tangent of
the complete substepped map. The one-step constitutive tangent and the
host-side composite tangent are distinct mathematical objects.

## Present qualification level

The following claims are supported by the current tests and qualification
artefacts:

- DIC boundary data and EBSD orientations are consumed through explicit,
  co-registered contracts;
- raw 3D condensation and structural plane stress preserve the complete local
  crystal state;
- the structural closure enforces all three transverse traction components;
- SRIX and Méric use the same structural host implementation;
- SRIX uses the canonical semismooth Jacobian convention at zero slip;
- snapshots, trial evaluation, revert, and commit obey the transaction
  invariants;
- field comparisons are made with the same load path, mesh, orientation data,
  and parameter provenance when a backend comparison is claimed.

These claims qualify the implemented numerical workflow. They do not imply
that the constitutive parameters are identified uniquely from DIC, that a
two-dimensional model resolves through-thickness physics, or that every
experimental observable is reproduced without model discrepancy.

## Fast SRIX identification gate

The reconditioned weak-equilibrium gap recovers the known SRIX parameter
valley on an exact digital twin and ranks 20 nearby laws similarly to complete
FEMU in exact kinematic space. The corresponding Spearman and logarithmic
Pearson correlations are `0.866` and `0.878`.

That qualification does not transfer through the present DIC observation
operator. With the measured transfer and no noise, the ranking correlations
fall to `0.326` and `0.276`, below the frozen `0.80` and `0.70` thresholds.
Consequently no P43 parameter optimization is authorized with the current
objective. This negative gate is explained in {doc}`srix_regm_identification`;
it is an observation-model limitation, not a failure of the SRIX transaction
or weak-equilibrium assembly.

## Reading the results

Use the evidence registry and the case-specific result artefact for numerical
values. Interpret slip maps together with orientation, boundary data, stress,
and displacement fields. For the scientific interpretation of the temporal
loading path and observation operator, see
{doc}`temporal_loading_path` and {doc}`../reference/observation_operator`.
