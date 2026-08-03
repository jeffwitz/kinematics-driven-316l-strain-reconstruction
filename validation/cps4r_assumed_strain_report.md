# CPS4R-AS — interim qualification report

Date: 2026-08-04
Preregistration: `validation/cps4r_assumed_strain_preregistration.md`
Derivation: `docs/reference/quas4_assumed_strain_derivation.md`

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

## What the numbers say

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

**The speed-up figure here is not the one that matters.** The Python J2 return
mapping is cheap, so the constitutive share of the total is small and dividing it
by four barely moves the total; the per-iteration cost of rebuilding the
stabilisation from the current tangent then eats most of the gain. R3.06.10 makes
exactly this point about its own 20 % measurement — *"des gains de temps beaucoup
plus importants sont attendus pour des lois plus difficiles à intégrer"*. The
target of 1.8 was set with SRIX in mind, where a material point costs roughly
sixteen times a J2 point, and it must be measured there.

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

If the SRIX measurement confirms the speed-up and the accuracy stays near 4 %,
this is **case B**: the cost target met, the constitutive accuracy not, usable
for exploration and not for scientific conclusions. Nothing seen so far suggests
case A. The formulation is not authorised for any campaign, and CPS4 remains the
reference.
