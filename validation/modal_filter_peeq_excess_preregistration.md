# Modal filter mean-PEEQ excess — preregistration

Date: 2026-07-31
Closes an item recorded as unexplained in
`dic_multistep_p0043_modal_boundary_filter_results.md`.

## The anomaly

The 3-mode boundary filter removes content below the measurement noise floor,
yet the filtered run accumulates **more** mean core PEEQ than the unfiltered
one, `3.2507e-3` against `3.2301e-3`, a `+0.64 %` excess — while its **peak
falls**, `7.2394e-2` against `7.3608e-2`.

That is contrary to a simple additive noise-ratchet picture, in which removing
boundary noise should reduce accumulation everywhere.

## Registered candidate explanations

**H1 — sub-increment confound.** The two runs do not share a sub-increment
structure: 40 converged increments for the filtered run against 65 for the
unfiltered one, which was cut back three times. The results document asserted
this confound is "of the right order to matter". That assertion was made
without measurement and is tested here.

**H2 — redistribution.** The filter changes the boundary path slightly, and
PEEQ is path dependent by `15.8 %` between genuinely different paths. A mean
that rises while the peak falls is the signature of plasticity spreading rather
than growing.

**H3 — neither.** Something else, to be reported as still unexplained.

## Method, on archived data only

No new mechanics. Three measurements:

1. **Calibrate the discretisation trend** from the archived proportional pair,
   20 against 40 increments, which converged with zero cutbacks and so differ
   in increment count alone. Report the signed change in mean core PEEQ per
   increment-count ratio, and extrapolate to the 40-against-65 range.
2. **Decompose the excess by PEEQ decile** of the unfiltered field: is the
   `+0.64 %` uniform, or concentrated at particular levels?
3. **Measure the active area** at the `1e-4` threshold in both runs: does the
   filtered run plastify more elements, or the same elements harder?

## Registered decision rule

| Observation | Conclusion |
|---|---|
| discretisation extrapolation explains `>= 50 %` of the `+0.64 %` | H1: confound, as previously assumed |
| it explains `< 20 %`, and the excess is non-uniform across deciles | H2: redistribution, and the earlier "right order" claim was wrong |
| neither pattern holds | H3: report as still unexplained |

The `50 %` and `20 %` bounds are fixed here, before any of the three
measurements is computed.

## Claim boundary

This explains a numerical observation about two computed fields. It licenses no
statement about which field is more physically correct, and none about the
measurement chain.
