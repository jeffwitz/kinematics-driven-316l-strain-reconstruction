# Modal filter mean-PEEQ excess — results

Date: 2026-07-31
Preregistration: `modal_filter_peeq_excess_preregistration.md`.
Archived data only; no mechanics were run.

## Short answer

**H2: redistribution.** The `+0.64 %` mean-PEEQ excess of the filtered run is
not a sub-increment artefact and not an amplitude increase. The filter
**concentrates** plasticity: it removes scattered marginal yielding at low
levels, adds plasticity in the mid-to-high range where the bands are, and
shaves the extreme tail.

The sub-increment confound explains **5.8 %** of the excess, against a
registered bound of 20 % for refuting it. The claim in
`dic_multistep_p0043_modal_boundary_filter_results.md` that this confound was
"of the right order to matter" was made without measurement and **is wrong**;
that document is corrected.

## 1. The sub-increment confound is an order of magnitude too small

Calibrated on the archived proportional pair, which converged with zero
cutbacks and so differs in increment count alone:

| Change | Effect on mean core PEEQ |
|---|---:|
| 20 → 40 increments (measured) | `-0.0533 %` |
| 65 → 40 increments (extrapolated) | `+0.0373 %` |
| **observed filtered excess** | **`+0.6390 %`** |
| share explained | **`5.8 %`** |

The sign is right — fewer increments do give slightly more accumulated PEEQ, so
the filtered run's 40 against the unfiltered run's 65 pushes the correct way —
but the magnitude is seventeen times too small.

## 2. The excess is strongly non-uniform

Decomposed by decile of the **unfiltered** field:

| Decile | PEEQ range | mean difference | share of total excess |
|---:|---|---:|---:|
| 1–2 | `0` … `3.4e-8` | `~0` | `0.0 %` |
| 3 | `3.4e-8` … `2.3e-4` | `-2.978e-6` | **`-1.4 %`** |
| 4 | `2.3e-4` … `7.6e-4` | `-4.599e-6` | **`-2.2 %`** |
| 5 | `7.6e-4` … `1.5e-3` | `+1.841e-6` | `+0.9 %` |
| 6 | `1.5e-3` … `2.4e-3` | `+1.682e-5` | `+8.1 %` |
| 7 | `2.4e-3` … `3.4e-3` | `+3.384e-5` | `+16.4 %` |
| 8 | `3.4e-3` … `5.0e-3` | `+4.609e-5` | `+22.3 %` |
| 9 | `5.0e-3` … `8.4e-3` | `+5.247e-5` | `+25.4 %` |
| 10 | `8.4e-3` … `7.4e-2` | `+6.292e-5` | **`+30.5 %`** |

The top four deciles carry `94.6 %` of the excess. Deciles 3 and 4 are
**negative**: at low plastic levels the filtered run yields *less*.

## 3. Fewer elements plastify, and the extreme tail drops

| Quantity | unfiltered | filtered |
|---|---:|---:|
| active area at `1e-4` | `73.364 %` | `73.169 %` |
| active elements | `81 874` | `81 657` |

Net `-217` elements: `98` newly active, `315` newly inactive.

| Level | unfiltered | filtered | change |
|---|---:|---:|---:|
| q0.99 | `2.5708e-2` | `2.5853e-2` | `+0.57 %` |
| q0.999 | `4.0620e-2` | `4.0737e-2` | `+0.29 %` |
| q0.9999 | `6.0128e-2` | `5.7996e-2` | **`-3.55 %`** |
| maximum | `7.3608e-2` | `7.2394e-2` | **`-1.65 %`** |

The broad upper range rises while the extreme tail falls. That is why the mean
can rise `0.64 %` while the peak falls `1.65 %` — they are measuring different
parts of the distribution.

## Reading

The coherent account is that high-frequency boundary content below the noise
floor was doing two things at once: seeding scattered marginal plastic events
at low levels across the core, and producing isolated extreme excursions at a
few points. Removing it lets the imposed deformation channel into the bands
instead — more plasticity where the material localises, less scattered
elsewhere, and no noise-driven outliers.

This is consistent with the band structure ratio of `3.95` already measured for
the filter's own effect, and with the loading-path result that PEEQ
redistribution is a band-concentrated phenomenon.

It also means the earlier framing of a "noise ratchet" was too simple for this
case. An additive ratchet predicts a uniform reduction when noise is removed.
What is observed is a transfer: down at low levels and in the extreme tail, up
across the band range.

## Claim boundary

This explains a numerical observation about two computed fields. It licenses
**no** statement about which field is more physically correct. A more
concentrated plastic field is not thereby a more accurate one; the filter is
justified by the measured noise floor, not by the shape of the field it
produces, and choosing a filter because its output looks cleaner would be the
same circularity the project rejects elsewhere.

## Reproduction

Archived fields only, no solver:

```python
from fem_inhouse.workflows.multistep_path_dependence import core_slice
w = core_slice((1290, 1950, 780, 1390), (1440, 1800, 930, 1240))
# filtered:   validation/reference_data/dic_multistep_modal3_p0043_v1/PEEQ.npy
# unfiltered: validation/reference_data/dic_multistep_predictor_fix_p0043_v1/PEEQ.npy
# 40-increment control: validation/reference_data/dic_multistep_proportional40_p0043_v1/PEEQ.npy
# 20-increment control: results/constitutive-local-p0043-pad150/partitions/0043/PEEQ.npy
```
