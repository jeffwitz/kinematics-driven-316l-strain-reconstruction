# P43 measured versus proportional path against DIC strain — results

Date: 2026-07-30
Preregistration:
`dic_multistep_p0043_observed_path_comparison_preregistration.md`.
Primary machine-readable results:
`reference_data/dic_multistep_observed_path_p0043_v1/`.

## Short answer

**Indistinguishable.** Under the symmetric image-level observation, driving the
model with the real measured incremental boundary history does **not** bring
the final total strain measurably closer to DIC than a proportional ramp to the
same endpoint. No registered metric moves by more than its registered margin,
on either profile.

This is the registered "indistinguishable" outcome, and it contradicts the
expectation recorded before computing, which was that the measured history
would be at least as close and probably closer.

## Symmetric comparison, primary profile `legacy_script_2021`

Both FEM fields warped onto the reference image, re-observed through DISFlow
with the DIC profile, EVM reconstructed with the same operator on both sides.
No Helmholtz post-filter.

| Metric | A measured | B proportional 40 | C proportional 20 | A − B | margin | verdict |
|---|---:|---:|---:|---:|---:|---|
| relative L2 | `0.50060` | `0.48515` | `0.48584` | `+0.01545` | `0.0202` | indistinguishable |
| Pearson | `0.60413` | `0.60390` | `0.60422` | `+0.00023` | `0.0185` | indistinguishable |
| top-10 % IoU | `0.29421` | `0.29866` | `0.29873` | `−0.00444` | `0.0189` | indistinguishable |
| absolute-q90 IoU | `0.32364` | `0.30876` | `0.30862` | `+0.01488` | `0.0217` | indistinguishable |
| q90 active fraction | `0.15243` | `0.16108` | `0.16133` | `−0.00866` | — | descriptive |

Sensitivity profile `declared_medium_v4` agrees on every verdict: relative L2
`+0.01629`, Pearson `+0.00092`, top-10 % IoU `−0.00359`, absolute-q90 IoU
`+0.01655`. All below margin. The two profiles do not disagree, so nothing has
to be averaged away.

## The raw comparison, reported as a known-biased control

| Metric | A measured | B proportional 40 | A − B | verdict |
|---|---:|---:|---:|---|
| relative L2 | `0.97854` | `0.95046` | `+0.02809` | B better, beyond margin |
| Pearson | `0.37614` | `0.37851` | `−0.00237` | indistinguishable |
| top-10 % IoU | `0.20845` | `0.20714` | `+0.00131` | indistinguishable |
| absolute-q90 IoU | `0.20728` | `0.20413` | `+0.00316` | indistinguishable |

Raw numbers are identical across profiles by construction, since no DISFlow
observation is applied to the FEM side. The raw view would report the measured
path as **worse** on amplitude. It is not the registered conclusion, and the
project has already established why: only the experimental field has passed
through DISFlow, so the raw view overstates model error. It is recorded here
only to keep the asymmetry visible.

## A weak, consistent, sub-margin trend

The same pattern appears on both profiles and deserves recording even though it
does not clear the margin:

- the measured path is slightly **worse** on amplitude, relative L2 `+0.015` to
  `+0.016`;
- it is slightly **better** on absolute-q90 overlap, `+0.015` to `+0.017`, and
  its q90 active fraction is closer to the DIC reference value of `0.10`,
  `0.152` against `0.161`.

That is the observable trace of the PEEQ result: the measured path concentrates
plasticity more and spreads it less. It sharpens the bands slightly while
overshooting amplitude slightly. Neither effect is large enough to claim.

## Why a 15.8 % PEEQ difference produces no measurable gain

`dic_multistep_p0043_path_dependence_results.md` measured `15.82 %` relative L2
between the two paths on core PEEQ, concentrated in the bands. That difference
does not survive to the observable, for two compounding reasons:

- **EVM is not PEEQ.** Total strain is dominated by the imposed kinematics and
  the elastic part, and both runs are driven to the same endpoint boundary
  displacement. The plastic redistribution enters as a small correction;
- **DISFlow attenuates exactly where the paths differ.** The measured MTF-50 of
  the chain is near `49 px`, and the path difference lives in narrow filaments
  along the bands. The observation operator smooths away most of the signal
  that distinguishes the two paths.

Observed EVM statistics make the dilution concrete: the two paths differ by
`1 %` on the observed maximum (`0.016318` against `0.016158`) and by `0.08 %`
on the observed mean, against DIC mean `0.003928` and maximum `0.013101`.

## Consequences

1. **The measured incremental history is not validated or falsified by this
   observable.** It changes an internal variable that cannot be observed by
   `15.8 %`, while changing the observable by less than DIC noise sensitivity.
   This is an identifiability statement, not a defect.
2. **The proportional ramp remains a defensible modelling choice** for
   comparison against DIC total strain, at `2.2` times less Newton work: 225
   iterations and no cutback against 469 and three.
3. The `16 %` PEEQ systematic recorded for the archived micromorphic campaigns
   stands, and this result shows it cannot be resolved by comparing EVM to DIC.
   Discriminating the paths would need an observable sensitive to accumulated
   plasticity, which the present measurement chain does not provide.
4. The `C` archived 20-increment case reproduces `B` to within `0.0007` on
   every observed metric, independently confirming that increment count is not
   a confound.

## Claim boundary

DIC total strain is an image-derived observable, not ground truth. States are
ordered image indices, not force-synchronised load fractions. This compares
final states only; nothing is established about intermediate mechanical states,
and no material parameter was identified or re-identified.

## Reproduction

```bash
for run in dic_multistep_predictor_fix_p0043_v1 dic_multistep_proportional40_p0043_v1; do
  fem-inhouse export-run-as-campaign \
    --run validation/reference_data/$run --partition-id 43 --output /tmp/campaign_$run
done

fem-inhouse replay-dic-observation \
  --campaign /tmp/campaign_<run> --prepared-case data/processed/case_study \
  --reference-image <DIC_images>/000294.tif \
  --partition-id 43 --profile legacy_script_2021 \
  --output validation/reference_data/dic_multistep_observed_path_p0043_v1/<name>_legacy_script_2021
```
