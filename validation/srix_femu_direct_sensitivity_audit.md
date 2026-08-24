# Phase 0 audit — direct SRIX FEMU sensitivities

Audit performed on the working branch before changing mechanics. The
machine-readable source facts are the archived M8 reports and the MFront
introspection command recorded below.

## Environment and MFront contract

| item | observed value |
|---|---|
| MFront executable | `/home/jeff/.local/bin/mfront` |
| TFEL version | `5.1.0` |
| Python | `3.12.3` |
| MGIS module | `/home/jeff/.local/lib/python3.12/site-packages/mgis` |
| production library | `build/mfront/src/libBehaviour.so` |
| production behaviour | `Fcc316LForestRubinSrix`, 3-D hypothesis |

The loaded production behaviour exposes `@Parameter` entries `tau0`, `Q`,
`b` and `SrixOverstressModulus` (the entry name of `R`). It exposes one
`Strain` gradient and one `Stress` force. Its internal variables are
`ElasticStrain`, `PlasticSlip[0..11]`, `EquivalentPlasticSlip[0..11]` and
`BackStrain[0..11]`; its external variables are `Temperature` and
`NonlocalEquivalentPlasticStrain`; its material property is
`MicromorphicCouplingModulus`. The standard MGIS tangent block list is only
`Stress / Strain`. No parameter-derivative stress blocks are exported, as
expected for the current `@DSL Implicit` behaviour.

The structural plane-stress symbol is not loadable from the current shared
library under the direct `mgis.load(..., PlaneStress)` probe because the
library does not export the expected
`Fcc316LForestRubinSrixStructuralPlaneStress_PlaneStress_requiresStiffnessTensor`
symbol. The maintained condensed plane-stress batch therefore remains the
qualified production bridge and must not be replaced by a raw MGIS probe.

The existing generic validation probe was run successfully. It generates a
validation-only `ImplicitGenericBehaviour` and reports the four named blocks:
`Stress/Strain`, `Stress/NonlocalEquivalentPlasticStrain`,
`AccumulatedSlipOutput/Strain` and
`AccumulatedSlipOutput/NonlocalEquivalentPlasticStrain`. This does not alter
the production SRIX behaviour and does not prove that SRIX analytic parameter
blocks exist.

## Exact M8 reference

The reference twin is `validation/reference_data/srix_regm_twin_v1`:

| item | recorded value |
|---|---|
| mesh | `8 x 8` pixels, two TRI2 material states per pixel |
| pixel size | `0.00184 mm` |
| replay origin | state `0` |
| accepted history | `338` increments |
| scored endpoints | `4, 36, 125, 275, 312, 316, 326, 338` |
| parameter preset | `316l_srix_transposed_from_nasri2018_rate_1e-3` |
| reference FEMU FD step | `3e-3` in log coordinates |
| FEMU FD cost | `658.68 s` for eight perturbed forward solves |

The reference forward driver is
`scripts/qualify_srix_regm_twin.py::_generate_twin`. It calls
`solve_two_state_dirichlet_plane_stress` with exact Dirichlet boundary data,
adaptive substepping, maximum 25 Newton iterations, and verification of the
accepted state. Its global Newton Jacobian is matrix-free: the converged
constitutive tangent is applied through `TwoState...tangent_action`, and the
linear correction is obtained with nonsymmetric GMRES plus the EBI Green
preconditioner. The archived `k0_type` describing a sparse factorization belongs
to REGM, not to this forward FEMU reference.

Consequently, the first direct-sensitivity implementation must either reuse
the spectral tangent-action/GMRES path or explicitly label a sparse assembly as
a separate approximation. It must not silently compare a sparse `K_II` solve
with the FEMU FD Jacobian and call the result equivalent.

## Decision after audit

The MFront parameter and tangent audit is compatible with a Phase-1 shadow
provider. The main implementation risk is not parameter access; it is retaining
the accepted adaptive forward trajectory and applying the same matrix-free
global tangent operator to four sensitivity right-hand sides. No analytical
MFront parameter derivative should be attempted before this shadow oracle is
implemented and compared to the archived Jacobian.
