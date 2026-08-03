# CPS4R reduced integration — qualification results

Date: 2026-08-03
Preregistration: `validation/cps4r_qualification_preregistration.md`
Generator: `scripts/qualify_reduced_integration.py`, figure by
`scripts/plot_reduced_integration_diagnostic.py`
Archive: `validation/_generated/cps4r_qualification/`

## Verdict

**CPS4R is not authorised for scientific elastoplastic campaigns. No value of
`beta` is recommended.** The registered accuracy criterion A1 fails at every
`beta` tested and on both cases.

**The hourglass energy ratio is refuted as a gate.** Falsifier F3 fired in every
single configuration: each one passes the documented `1 %` warning threshold by
roughly an order of magnitude while failing the accuracy bound it is supposed to
protect, by four to twenty times. The registered consequence is applied below.

Two findings were not anticipated and are the useful part of this campaign.

## C1 — pixel-wise heterogeneous J2, 32x32

Campaign spacing `0.00184 mm`, affine `2 %` tensile boundary, yield map with
`15 %` pixel scatter and a `25 %` soft band, 20 increments, `mfront` backend.
CPS4 reference: 4096 material points, 99 Newton iterations, no cutback, peak
PEEQ `0.0367`. Timings are the median of five solves.

| `beta` | `r_hg` | displacement rel. `L2` | PEEQ rel. `L2` | RMS / `sigma_DIC` | total speed-up | constitutive speed-up |
|---|---|---|---|---|---|---|
| 0.1 | `1.43e-3` | `8.85e-4` | `1.94e-2` | `0.005` | `1.91x` | `3.71x` |
| 0.25 | `1.30e-3` | `2.89e-3` | `5.68e-2` | `0.017` | `1.90x` | `4.40x` |
| 0.5 | `1.17e-3` | `4.30e-3` | `8.14e-2` | `0.025` | `2.45x` | `4.43x` |
| 1.0 | `1.03e-3` | `5.50e-3` | `1.01e-1` | `0.032` | `2.57x` | `4.42x` |

F1 satisfied: the case excites the modes, `r_hg` around `1e-3` rather than the
`1e-13` the affine tests produce. A2 passes only at `beta = 0.1`. A4 and A5 pass
throughout. **A1 fails everywhere**, by a factor of 4 at best and 20 at worst.

## C2 — SRIX single crystal, tilted orientation, 8x8

Bunge `(35, 20, 15)` degrees, homogeneous, `beta = 1`, 10 increments, non-affine
boundary perturbation at `5 %` of the axial displacement. 256 material points
against 64.

| quantity | value |
|---|---|
| displacement rel. `L2` | `1.21e-2` |
| stress rel. `L2` | `1.35e-2` |
| RMS / `sigma_DIC` | `0.0071` |
| `r_hg` | `9.37e-4` |
| cutbacks | 0 against 0 |
| total speed-up | `2.93x` |
| constitutive speed-up | `4.78x` |

A1 is evaluated on stress here: crystal laws leave PEEQ at zero, and comparing
an empty field against an empty field would have reported a perfect score. The
first version of the script did exactly that and had to be corrected before any
result was recorded.

## Finding 1 — `beta = 1` is the worst choice after yielding

Registered hypothesis H2 predicted that lowering `beta` would move the answer
*away* from CPS4. **The opposite is observed, monotonically**: `beta = 0.1` is
six times closer to CPS4 on displacement and five times closer on PEEQ than
`beta = 1`.

The mechanism is not subtle once seen. The assembled element is
`K^{1pt}(C_tangent) + beta (K^{4pt}_{elastic} - K^{1pt}_{elastic})`. After
yielding the constitutive tangent collapses while the stabilisation keeps the
**full elastic** reference. At `beta = 1` the hourglass modes therefore retain
elastic stiffness while every other mode softens, and the element is
over-stiffened exactly where CPS4 would have softened. Lowering `beta`
compensates part of that.

This inverts the recommendation the documentation carried. `beta = 1` was
presented as the natural default because it is the only value with an exact
elastic equivalence — which remains true, and remains irrelevant once the
material yields.

The direction is monotone, so F2 did not fire; but H2 as written is refuted, and
the preregistration's own tie-break rule — "if several qualify, the largest is
recommended" — is now known to point the wrong way. It must not be reused.

## Finding 2 — the energy diagnostic predicts nothing, globally or spatially

F3 fired everywhere. Beyond the global gate, the spatial reading the
documentation recommends does not hold either. On C1 at `beta = 1`, over the
1024 elements:

- correlation between hourglass energy and PEEQ: **`r = 0.066`**;
- correlation between hourglass energy and the CPS4-to-CPS4R PEEQ error:
  **`r = 0.033`**.

Both are indistinguishable from zero. The stabilisation energy does not
concentrate in the plastic band, and it does not sit where the error is. Its
largest values are in isolated boundary elements. See
`_generated/cps4r_qualification/cps4r_spatial_diagnostic.png`.

The advice to look at the field beside PEEQ is not harmful, but the premise
behind it — that the energy localises with the error — is not supported by this
case. `r_hg` measures how much the stabilisation is being exercised. It does not
measure how wrong the answer is, and this campaign found no evidence that the
two are related.

## Finding 3 — the difference is real, and unmeasurable

A3 is a reported fact, not a pass condition, and it cuts the other way from
every criterion above. The RMS displacement difference between the two
formulations is `0.005` to `0.032` times the DIC boundary noise on C1, and
`0.0071` times on C2 — **30 to 200 times below what this project's measurement
can resolve**.

The relative `L2` figures are large because PEEQ is a small, spatially
structured field and the difference falls partly where PEEQ is small. Both
statements are true at once: by the registered internal-consistency criterion
CPS4R fails, and by the external measurement standard the two formulations are
indistinguishable.

This is not a licence to use CPS4R. The registered criterion was chosen so the
element formulation would not become a comparable error term in the
reconstruction, and it is that criterion which governs. But a reader deciding
what to do next should know that the failure is one of numerical
self-consistency, not of measurable physics.

## Cost — F4 satisfied

Constitutive time falls by `3.7x` to `4.8x`, matching H4's prediction of about
four. Total wall time falls by `1.9x` to `2.6x` on J2 and `2.9x` on the crystal
case, above the registered `1.5x` bound.

The gain is larger for SRIX, as expected: a crystal material point costs far
more than a J2 point, so the constitutive share of the total is larger and
dividing it by four moves the total further. The cost case for CPS4R is sound.
It is the accuracy case that fails.

An earlier single-shot timing showed `0.91x` at `beta = 1`, i.e. CPS4R apparently
slower than CPS4 with an identical Newton count. That was machine noise, and it
is recorded here because it was nearly reported as a finding. The script now
takes the median of five solves.

## Registered consequences, applied

**F3 fired, so the thresholds are withdrawn as a gate.** The `1 % / 5 %` bands in
`docs/explanation/reduced_integration_hourglass.md` and in
`docs/scientific_contract.md` are restated as descriptive only, with this
campaign cited: they describe how hard the stabilisation is working, and this
campaign found no relationship between that and the error. A configured
`hourglass_energy_failure_ratio` remains available as a blunt guard against a
runaway solve, which is a different purpose and is documented as such.

**H2 refuted, so no `beta` is recommended and the tie-break rule is withdrawn.**

**CPS4 remains the reference formulation and the default.** Nothing here changes
that, and this campaign moves CPS4R no closer to replacing it.

## What would change the verdict

This is a negative result on two small synthetic cases, not a proof that CPS4R
cannot work. What is missing:

- a mesh-convergence study. Both cases have element-scale material contrast by
  construction; the error may fall quickly with refinement, and if it does, the
  verdict is about resolution rather than about the element;
- a stabilisation built on the **current** tangent rather than the fixed elastic
  reference, which is what Finding 1 points at directly. That is a different
  element and would need its own qualification, including whether it stays
  positive semi-definite once the tangent softens;
- an error estimator that actually predicts the CPS4-to-CPS4R difference. This
  campaign shows the stored stabilisation energy is not one.

## Reproduction

```bash
MFRONT_BEHAVIOUR_LIBRARY="$PWD/build/mfront/src/libBehaviour.so" \
python scripts/qualify_reduced_integration.py \
  --mesh 32 --crystal-mesh 8 --repeats 5 \
  --output validation/_generated/cps4r_qualification
python scripts/plot_reduced_integration_diagnostic.py
```

Runtime is a few minutes. No experimental ROI is used and no campaign is
replayed.
