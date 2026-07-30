# Does the loading path matter?

**Category: Explanation.** Every reconstruction so far pushed the boundary along
a straight ramp to the final measured displacement. The experiment did not do
that. This chapter asks what changes when the model is driven by the real
sequence of measured states instead.

## The question, and why it is well posed

The DIC series gives 40 states between the reference image and the final one.
Two calculations can be built from it:

- a **proportional ramp**, which interpolates linearly from zero to the final
  measured displacement;
- the **measured history**, which imposes each of the 40 states in turn.

Both end on exactly the same prescribed boundary displacement. The workflow
overwrites the last history state with the prepared final field, so the two
runs are bit-identical at the boundary at the end.

That is what makes the comparison clean. Plasticity is history dependent: the
equivalent plastic strain is a path integral, so two paths sharing an endpoint
need not share it. Any interior difference at the final state is therefore
**path dependence**, not a difference in what was imposed.

## The blockage was a bug, not mechanics

For a long time the measured history could not be run at all. It failed on the
transition from state 3 to state 4, and resisted frame removal, a state bridge,
a long line-search campaign and a boundary-noise hypothesis.

The instrumented Newton trace settled it. The solver assembles its stiffness
into one reusable CSR buffer, and the elastic predictor of the history branch
was being solved against that buffer *after* the elastoplastic tangent had
overwritten it. The predictor was therefore not elastic, and its error
compounded at every cutback, which is why smaller steps made the failure worse
instead of better.

The proportional path never met the defect, because its predictor is computed
once before the loop and afterwards only scaled. That is the whole reason one
path converged and the other did not — a software asymmetry, with no physical
content.

Giving the predictor its own copy of the elastic operator fixed it. The
measured history now runs to completion. No archived result changed, because
every archived campaign uses the proportional path.

:::{note}
This is worth remembering as a method lesson. A failure that gets *worse* when
the step size shrinks is not a globalisation problem. Cutbacks help when the
problem is nonlinearity; they cannot help when the linear system itself is
wrong.
:::

## What the path changes: plasticity

On the P43 core, excluding the padding, final PEEQ differs between the two
paths by **15.8 %** in relative $L_2$. The measured path accumulates *more*
plasticity, and the excess grows with the level: `+4.9 %` on the mean,
`+9.7 %` at the 99th percentile, `+14.8 %` at the maximum.

Two controls make this trustworthy.

**It is not a discretisation artefact.** Running the proportional ramp with 20
and with 40 increments changes core PEEQ by `0.20 %`, seventy-eight times less
than the path effect.

**It is not accumulated measurement noise.** PEEQ only grows, so DIC noise
produces a one-sided bias, estimated at about `3.6 %`. But a noise ratchet has
no reason to prefer any particular place, whereas the observed excess is
**thirteen times stronger inside the localisation bands than outside**. The
difference map is flat and near zero over almost the whole core, with positive
filaments tracing the bands.

The path effect also survives the boundary filter described below, which
removes content below the noise floor. A noise artefact would not.

## What the path does not change: the observable

The natural next question is whether the measured path therefore agrees better
with the DIC total strain. It does not.

Under symmetric image-level observation — both FEM fields warped and
re-observed through DISFlow, exactly as the DIC was — **no metric moves by more
than its significance margin**, on either profile. The margins are not chosen
here: they are the DIC-noise sensitivity intervals already measured in the
uncertainty campaign.

:::{warning}
A raw comparison, FEM against image-observed DIC, would report the measured
path as *worse*. That is the observation asymmetry described in
{doc}`dic_synthetic_measurement_tests`, and it is the reason the symmetric
replay is the only admissible comparison here.
:::

Two effects explain why a 15.8 % change in plasticity leaves the observable
untouched:

- **EVM is not PEEQ.** Total strain is dominated by the imposed kinematics and
  the elastic part, and both runs share the same endpoint boundary data. The
  plastic redistribution enters only as a correction.
- **The measurement chain smooths it away.** The MTF-50 of the chain is near
  49 px, and the path difference lives in narrow filaments along the bands.

The consequence is an **identifiability statement**, not a defect: the measured
history changes an unobservable internal variable substantially while changing
the observable by less than the noise. This observable can therefore neither
validate nor falsify it. Discriminating the two paths would require an
observable sensitive to accumulated plasticity, which the present measurement
chain does not provide.

## Imposing only what the instrument resolves

Imposing the measured field node by node at 1.84 µm asserts far more spatial
information than DISFlow provides, since the chain resolves about 49 px. The
boundary motion is in fact almost one-dimensional: a single mode carries
99.91 % of its energy.

Truncating the boundary to three modes removes content whose RMS amplitude is
`0.00972 px`, against a measured per-state noise floor of `0.0511 px` — five
times below the noise. The mode count is not a free choice: the temporal
roughness criterion fixed in the measurement campaign selects exactly the same
three modes.

The effect is striking and almost entirely numerical:

| | unfiltered | 3-mode filter | proportional ramp |
|---|---:|---:|---:|
| increments | 65 / 68 | 40 / 40 | 40 / 40 |
| cutbacks | 3 | **0** | 0 |
| Newton iterations | 469 | **245** | 225 |

The filtered measured history converges like a synthetic ramp. All of the
numerical difficulty was carried by content below the measurement noise floor.
Meanwhile core PEEQ moves by only `1.63 %` and DIC agreement does not move
measurably.

The filter keeps exact Dirichlet enforcement — only the data being imposed
changes — so conditioning, reactions and every archived result are untouched.
It should be described as a production trade, not as a gain in fidelity.

## What to take away

- The measured temporal history is now runnable, and its earlier blockage had
  no physical meaning.
- The loading path genuinely changes the reconstructed plasticity, by about
  16 % on the core, concentrated where the material localises.
- That difference is invisible to the DIC total strain, so it cannot currently
  be checked against experiment.
- Every archived micromorphic campaign uses the proportional path, so this is a
  known systematic to carry, not a correction to apply. The archived rankings
  are separated by much larger margins.

Details of what is and is not claimed are in {doc}`current_evidence` and
{doc}`scope_and_prediction`.
