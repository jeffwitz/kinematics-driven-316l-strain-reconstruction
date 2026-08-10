# MFront can return the four coupled tangent blocks itself

Feasibility probe, executed under `34b4dfe`. Artefact:
`validation/_generated/performance/micromorphic_generic_tangent_blocks.json`.
Reproduce with `scripts/probe_micromorphic_generic_blocks.sh`.

## What this settles

`validation/generic_implicit_sensitivity_contract.md` concluded that the
coupled Newton could not obtain its off-diagonal blocks from an arbitrary
MFront behaviour "without an additional MFront/TFEL export hook", because MGIS
exposes neither the local residual nor its Jacobian.

That conclusion is correct about MGIS and wrong about the consequence. **The
host does not need the local Jacobian.** It needs the four derivatives, and
MFront already computes them internally and hands them over through the
standard tangent-operator interface. Nothing has to be exported, and nothing
has to be asked of TFEL upstream.

## The mechanism, all four parts verified on this machine

| claim | verified |
|---|---|
| `ImplicitGenericBehaviour` DSL exists | `mfront --list-dsl` |
| a behaviour may declare several gradient/force pairs | `PoroPlasticity.mfront`, TFEL 5.1.0 reference test |
| `@TangentOperatorBlocks` requests named cross-derivatives | same |
| `getIntegrationVariablesDerivatives_*` reuses the converged implicit Jacobian | same |
| MGIS exposes `tangent_operator_blocks` | `mgis.behaviour.Behaviour` |

`validation/mfront/MicromorphicJ2GenericBlocksProbe.mfront` applies the
`PoroPlasticity` pattern to the micromorphic J2/Ludwik law: the mechanical pair
`(eto, sig)` and a second pair `(chi, pobs)` where `pobs` is the equivalent
plastic strain. That pairing is an **interface device, not a physical
conjugacy** — `pobs` is not the energetic dual of `chi`, it is the observable
the coupled solver differentiates. MFront does not require the pairing to be
energetic, and MGIS reports exactly the four requested blocks:

```text
gradients: [('Strain', 6), ('NonlocalEquivalentPlasticStrain', 1)]
forces   : [('Stress', 6), ('EquivalentPlasticStrainOutput', 1)]
blocks   : [('Stress','Strain'), ('Stress','NonlocalEquivalentPlasticStrain'),
            ('EquivalentPlasticStrainOutput','Strain'),
            ('EquivalentPlasticStrainOutput','NonlocalEquivalentPlasticStrain')]
```

## Accuracy

Twelve plastic steps, each block against central finite differences of the same
behaviour restarted from the identical committed state. Worst relative error
over the plastic steps:

| block | `h = 1e-6` | `h = 1e-7` | `h = 1e-8` |
|---|---:|---:|---:|
| `dsig_ddeto` | `4.74e-08` | `4.74e-10` | `1.36e-10` |
| `dsig_ddchi` | `1.39e-10` | `6.66e-10` | `9.24e-09` |
| `dpobs_ddeto` | `1.17e-07` | `1.17e-09` | `2.18e-10` |
| `dpobs_ddchi` | `6.22e-11` | `2.56e-10` | `1.01e-08` |

The two `_ddeto` blocks fall by exactly `100×` from `1e-6` to `1e-7`, the
`O(h^2)` of a central difference, then flatten on the subtraction noise floor.
The two `_ddchi` blocks are already at that floor at `h = 1e-6`, so the
comparison bounds them at `~1e-10` rather than resolving them. Nothing here is
a fitted agreement: the finite differences are the approximation and the blocks
are the exact derivative.

## Cost

10 000 material points, one plastic increment, single thread:

| | time |
|---|---:|
| one integration, no tangent | `159.59 ms` |
| one integration **plus the four blocks** | `177.82 ms` |
| host finite-difference route, estimated as nine integrations | `1436.34 ms` |

The complete coupled linearisation costs **11 % over a bare integration**. The
finite-difference route costs `9×`. The estimate for the FD route is a floor,
not a measurement of the current host code: it counts only the integrations,
ignoring the snapshot, restore and assembly around them.

## What remains before this reaches production

This probe answers one question and deliberately no others.

**It is tridimensional.** The production micromorphic law is
`@ModellingHypothesis PlaneStress`. Combining the block mechanism with the
plane-stress closure is the next step, and keeping them apart here is what
makes this result interpretable.

**It abandons the brick.** The production law is built on
`@Brick StandardElastoViscoPlasticity` with a `UserDefined` isotropic
hardening. `ImplicitGenericBehaviour` does not offer that brick, so the return
mapping is hand-written here. Any production conversion inherits that cost and
must be qualified against the current behaviour, stress and state, before it
can replace it.

**It is J2.** The contract's crystal-plasticity requirement is untouched by
this probe. The mechanism is generic — `getIntegrationVariablesDerivatives_*`
does not care how many unknowns the local system has — but genericity claimed
is not genericity measured, and SRIX has not been tried.

## One trap, recorded because it cost the first three attempts

Under `Implicit` with `StandardElasticity`, the brick sets the elastic guess
`deel = deto` before the local Newton. Under `ImplicitGenericBehaviour`
**nothing does**, and the default `deel = 0` leaves `sig` at the committed
stress. On the first plastic increment of a virgin point that stress is zero,
so the flow direction `3 dev(sig) / (2 seq)` divides by zero and the local
Newton dies before its first iteration.

The symptom is deceptive: every elastic step succeeds, and every step above
yield fails at any amplitude, which reads like a broken residual rather than a
missing initial guess. It was isolated by bisection — a numerical Jacobian
failed identically, ruling out the analytic one, and a single-gradient variant
failed identically, ruling out the two-gradient mechanism. `@Predictor` must
set `deel = deto` explicitly.
