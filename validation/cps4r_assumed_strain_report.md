# CPS4R-AS — interim qualification report

Date: 2026-08-04
Preregistration: `validation/cps4r_assumed_strain_preregistration.md`
Derivation: `docs/reference/quas4_assumed_strain_derivation.md`

Generator: `scripts/compare_reduced_integration_formulations.py`
Archive: `validation/_generated/cps4r_as/formulation_comparison_srix.json`

```bash
MFRONT_BEHAVIOUR_LIBRARY="$PWD/build/mfront/src/libBehaviour.so" \
python scripts/compare_reduced_integration_formulations.py \
  --case srix --mesh 12 --increments 8 --repeats 5
```

The archive carries the five individual elapsed and constitutive times per
formulation, the Newton and cutback counts, the SHA-256 of every compared field
on both sides, and the reduction used. The first version of this report quoted
the numbers without a generator; with 0.17 points of margin that was not good
enough, and a review said so.

**Interim. No verdict of section 26 is issued yet**: the campaign of sections 13
and 14 is not complete, and the performance figures below were taken on the
cheap J2 backend, which is the wrong law to measure a constitutive speed-up on.
What follows is what has been established, and it is enough to say where the
formulation stands.

## Where it stands

The element algebra is correct and verified. The solver wiring works. On the one
heterogeneous plastic case run so far, the assumed-strain element **halves the
error of the existing `cps4r`** and still **misses the one-percent bound by a
factor of four**.

| formulation | `E_p` | `E_sigma` | `E_R` | `E_u` | points | speed-up |
|---|---:|---:|---:|---:|---:|---:|
| `CPS4` (reference) | — | — | — | — | 2304 | 1.00× |
| `CPS4R-elastic` | 8.86 % | 4.71 % | 11.19 % | 0.59 % | 576 | 1.57× |
| `CPS4R-AS-current`, ASMD | **4.20 %** | **2.57 %** | **3.21 %** | 0.26 % | 576 | 1.08× |
| `CPS4R-AS-energy`, ASMD | 4.20 % | 2.57 % | 3.21 % | 0.26 % | 576 | 1.04× |
| `CPS4R-AS-energy`, ASOI(1/2) | — | — | — | — | — | did not converge |

Pixelwise heterogeneous J2, 24×24 at the campaign spacing, 15 % yield scatter, a
25 % soft band, 20 increments, Python J2 backend.

Read against the preregistered bounds: **`E_p < 1 %` fails**, `E_sigma < 1 %`
fails, `E_R < 1 %` fails, `S_total > 1.8` fails. The direction is right — every
error is roughly halved against the elastic-difference form, and the reaction
error falls by 3.5× — but the bounds are not met.

## SRIX, the measurement the targets were set for

Homogeneous crystal, Bunge `(35, 20, 15)`, 12×12, **non-affine** boundary — a
sinusoidal perturbation at 5 % of the axial displacement, because an affine field
on a homogeneous material is an exact equilibrium solution for every formulation
and measures nothing. Eight increments, medians of five runs.

CPS4 reference: 576 material points, 6.28 s total, 5.75 s constitutive, 37 Newton
iterations, no cutback.

**The reduction, stated because the margin is 0.17 points.** CPS4 carries four
constitutive states per element and CPS4R-AS one. Every field compared here is
reduced to the element grid by the **arithmetic mean over Gauss points**, which
for CPS4R-AS is the identity and for CPS4 averages four states. Comparing an
average of four against one central value is therefore *part of* the measured
difference, not a neutral projection, and a different reduction — the central
value of the four, or a volume-weighted mean — would give a different number.
`E_Gamma` is computed on `cumulated_slip`, itself the sum over the twelve systems
of the accumulated slip `p_s`, after that reduction.

| formulation | `E_Gamma` | `E_sigma` | `E_R` | `E_u` | total | constitutive | Newton |
|---|---:|---:|---:|---:|---:|---:|---:|
| `CPS4R-elastic` | 5.89 % | 1.31 % | 5.19 % | 1.25 % | 3.58× | 4.53× | 32 |
| **`CPS4R-AS`, ASMD, energy** | **1.17 %** | **0.25 %** | **0.69 %** | 0.20 % | **2.18–2.27×** | 2.72–2.87× | 47 |
| `CPS4R-AS`, ASMD, current | did not converge | | | | | | |

Against the preregistered bounds: `E_sigma` **passes**, `E_R` **passes**,
`S_total > 1.8` **passes** at 2.18. `E_Gamma` **misses at 1.17 % against 1 %** —
by 17 %. `S_const > 3.5` **misses at 2.72**.

Four things in that table matter more than the pass/fail.

**The assumed-strain element cuts the cumulated-slip error by five** against the
elastic-difference form, 5.89 % to 1.17 %, and the reaction error by seven and a
half, 5.19 % to 0.69 %. With the same single constitutive point.

**The constitutive miss is an iteration count, not a per-call cost.** The point
count falls exactly 4× — 576 to 144 — and every call is one call. What dilutes
the ratio is that Newton takes 47 iterations against 37, 27 % more, so the
constitutive *time* ratio is roughly `4 / 1.27`. The premise of section 2 holds
exactly; the section 17 target, which is stated on total constitutive time,
does not.

**The energy projection earns its place here.** `assumed_strain_current` fails to
converge on SRIX where the projected variant does not — the opposite of the J2
case, where the two were identical to every digit because the tangent never lost
definiteness. A crystal tangent does, and section 6.2 exists for exactly that.

**No cutback is introduced**, so falsifier F2 does not fire.

The speed ranges are two independent runs of the same command on the same
machine — 2.18 and 2.27 on total, 2.72 and 2.87 on constitutive. Both sides of
each range fall on the same side of their bound, so the conclusion does not
depend on which run is quoted.

## What the J2 numbers say

**The current tangent is worth about a factor two.** That is the whole thesis of
this work and it is supported: replacing a frozen elastic stabilisation by one
built on the current tangent cuts the plastic-strain error from 8.9 % to 4.2 %
with the same single constitutive point.

**The energy projection is inactive here.** `AS-current` and `AS-energy` agree to
every digit printed, meaning the spectral floor never lifted an eigenvalue: the
J2 tangent stays positive definite throughout this case. The floor is therefore
untested by this run, and its sensitivity sweep (falsifier F4) is not yet
meaningful.

**ASOI(1/2) does not converge** where ASMD does. Consistent with the
frame-dependence measured at the element level: it is not a tensor operation, and
the resulting stabilisation is not a consistent derivative of anything.

**The J2 speed-up figure is not the one that matters, and SRIX confirms it.** The Python J2 return
mapping is cheap, so the constitutive share of the total is small and dividing it
by four barely moves the total; the per-iteration cost of rebuilding the
stabilisation from the current tangent then eats most of the gain. R3.06.10 makes
exactly this point about its own 20 % measurement — *"des gains de temps beaucoup
plus importants sont attendus pour des lois plus difficiles à intégrer"*. The
target of 1.8 was set with SRIX in mind, where a material point costs roughly
sixteen times a J2 point, and it must be measured there.

## Why Newton takes 47 iterations: measured, not hypothesised

`scripts/diagnose_assumed_strain_tangent.py`, archived in
`validation/_generated/cps4r_as/assumed_strain_tangent_consistency.json`.

At a plasticised SRIX state with a non-zero hourglass amplitude, the derivative
of the **complete** element internal force is compared against the matrix
actually assembled, **with the constitutive law re-integrated at every
perturbation** so that `C` moves as it does in a real Newton step. The existing
element tests hold `C` fixed and cannot see this.

| part of the element force | relative error of the assembled tangent |
|---|---:|
| physical, `A Bc^T sigma_c` | **1.9e-6** |
| stabilisation | **3.70**, i.e. 370 % |
| total | **0.36**, i.e. 36 % |

The physical part is consistent to the finite-difference floor. **The
stabilisation is not**, and the missing term is exactly the one suspected:
`f_stab` depends on the projected current tangent, `f_stab(u, C(u))`, while the
matrix differentiates it holding `C` fixed and drops `(df_stab/dC)(dC/du)`.

A 36 % error on the element tangent is a quasi-Newton matrix, not a wrong
solution: it converges, and it converges more slowly. That is the whole of the
27 % iteration overhead, and therefore the whole of the constitutive speed-up
shortfall — the per-element call count is exactly one throughout.

It also makes `assumed_strain_energy_lagged` a directed fix rather than a guess.
Freezing the projected tangent through Newton makes `dC/du = 0` by construction,
so the stabilisation force acquires an exactly consistent derivative.

**Two probe defects were found on the way, both of which first looked like the
element failing.** A central difference reported 16 % of error on the *physical*
part alone — not credible for `A Bc^T C Bc`, and the signal that the probe was
wrong: at a converged plastic state the algorithmic tangent is one-sided, and a
central difference averages the elastic unloading branch with the plastic loading
one. Switching to a forward difference made it worse, 95 %, because the reference
was still being taken at a **zero** strain increment, where SRIX takes its
guarded elastic branch: an elastic tangent was being compared against a plastic
response. Committing all but the last step fixed both, and the physical part then
agreed to 1.9e-6 — which is what makes the 370 % on the stabilisation
believable.

## A defect found in the wiring, and what it teaches

The first wiring **double-counted the stabilisation**. `precompute_element` folds
an elastic stabilisation into `Ke` so the elastic predictor sees a non-singular
element, and `element_tangent_stiffness` starts every element from `Ke`; adding
the current stabilisation on top rather than replacing that baseline meant both
were present.

It passed every elastic test — with an elastic tangent the two contributions are
identical and the element is merely uniformly too stiff, which an affine patch
test cannot see — and it destroyed Newton convergence the moment the tangent
softened. **All four variants failed to converge, which is what made it obvious
the fault was mine and not the formulation's**: a property of the method would
not have hit every projection and both strategies equally.

## What remains before a verdict

1. **SRIX**, where the speed-up target was set and where it can be met or missed
   honestly. Nothing below this line is decidable on a J2 law.
2. The homogeneous battery of section 13 and the remaining heterogeneous cases of
   section 14 — checkerboard, oblique interface, synthetic polycrystal,
   controlled hourglass perturbation.
3. The instrumented constitutive-call count (falsifier F1). The guarantee is
   currently **structural** — the strategy interface receives only the central
   tangent and cannot reach a material model — and the material-point count in
   the table above is consistent with it, but it has not been proved by counting
   calls.
4. Performance as medians over five repetitions after warm-up, per the protocol.
5. Increment sensitivity, to rule out accuracy bought with a different cutback
   path.

## Provisional reading

**No verdict can be issued.** Two quantitative SRIX criteria are **missed**:
the cumulated-slip error at 1.17 % against 1 %, and the constitutive speed-up at
2.72 against 3.5 — the latter because Newton needs 27 % more iterations, not
because any element calls the material more than once. Several other criteria
are **not evaluated**: the homogeneous battery, every crystal heterogeneity case,
the instrumented constitutive-call count, increment sensitivity, the spectral
floor sweep, and the localisation diagnostics.

An earlier draft of this section said "fourteen of the sixteen criteria are met
or not yet contradicted". That was too favourable: **not contradicted is not
satisfied**, and counting unevaluated criteria as near-passes is exactly the
arithmetic a preregistration exists to prevent.

That is a materially better position than the formulation it replaces, which
misses the same bounds by five to seven times. But the preregistration does not
grade on improvement, and the bounds were frozen before the numbers existed.

**The formulation is not authorised for any campaign and CPS4 remains the
reference.** The two remaining questions are whether the extra Newton iterations
can be removed — the stabilisation force is differentiated holding `C` fixed,
which is exact in the elastic range and an approximation once the tangent moves —
and whether the slip error falls below 1 % on a finer mesh, which would make the
miss a resolution matter rather than a formulation one. Neither is settled, and
neither may be settled by moving a threshold.
