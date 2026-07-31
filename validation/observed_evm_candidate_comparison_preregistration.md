# Observed-EVM candidate comparison — preregistration

Date: 2026-07-31
Lot 5 of the observed-EVM comparison specification.

> **VALIDATED by the project owner on 2026-07-31**, including amendments 1 and 2.
> The campaign of lot 6 is authorised to run on the archived results.
> No new constitutive calculation is authorised.

**This document was validated before the campaign was run.** No candidate has been analysed with the
lot 2 to 4 tooling; nothing in this file is written after seeing its output.

## What is blind here, and what is not

Stated first because it limits what this preregistration can claim.

**Not blind.** The four candidates' *global* observed scores are already
archived in `dic_symmetric_observation_p0043_v1` and have been discussed:

| Case | relative L2 | q90 active fraction | top-10 % IoU |
|---|---:|---:|---:|
| local | `0.4858` | `0.1613` | `0.2987` |
| `alpha = 1` | `0.3542` | `0.1500` | `0.2890` |
| `alpha = 2` | `0.3197` | `0.1354` | `0.2817` |
| `alpha = 4` | `0.2917` | `0.1026` | `0.2527` |

Any bound chosen to sit just above or below one of these numbers would be
preregistration theatre. **No elimination bound below is derived from this
table.** Every bound comes from the measurement chain — its noise floor, its
resolution — or from a geometric property of the bands themselves.

**Blind.** Nothing per band, nothing multiscale, nothing bootstrapped has ever
been computed for these candidates. Band geometry, centreline error, the three
widths, continuity, FSS, residual structure, pairwise probabilities and the
Pareto front are all new. That is where this document has force.

## Candidates

| Label | Campaign | Role |
|---|---|---|
| `local` | `results/constitutive-local-p0043-pad150` | no coupling |
| `alpha1` | `results/constitutive-nonlocal-p0043-pad150-a100` | coupling candidate |
| `alpha2` | `results/constitutive-nonlocal-p0043-pad150-a200` | coupling candidate |
| `alpha4` | `results/constitutive-nonlocal-p0043-pad150-a400` | coupling candidate |
| `homogeneous` | `results/control-homogeneous-local-p0043-pad150` | negative reference, background |
| `translated` | `results/control-translated-maps-local-p0043-pad150` | negative reference, localisation |

All six are converged, share P43 with padding 150, 20 increments and the
proportional path.

**Excluded, deliberately.** The measured-history, proportional-40 and
modal-filtered runs of 2026-07-30 are *not* candidates. They differ in loading
path and increment count, and mixing them in would confound the constitutive
question with the path question. They are already known to be indistinguishable
on this observable; revisiting them is a separate campaign.

## Observation operator

`legacy_script_2021` primary, `declared_medium_v4` sensitivity, applied
identically to every candidate through `replay_dic_observation`. The raw
comparison is computed and archived as a **known-biased control** and never
used for selection.

Comparison support: the P43 core, `core_bounds = [1440, 1800, 930, 1240]`,
padding excluded. Identical mask, crop, EVM operator and non-finite policy for
every candidate.

## Band geometry, from the DIC alone

**The threshold that defines the geometry and the thresholds that define
"active" for the FSS are different things and are not the same here.** The
first version of this document left the geometry threshold unspecified, which
was a hole: it decides how many bands exist.

Geometry threshold: **q80**, fixed after a DIC-only measurement recorded in the
amendment below. Minimum object area
**256 px**, which is `4 x 64` — narrower than the `49 px` MTF-50 of the chain in
one direction, so no object the chain could not resolve survives, and it is not
tuned to any candidate.

Centreline: Zhang-Suen thinning, spur pruning at **16 px** (a third of the
MTF-50), Euclidean shortest path between extremities, moving-average smoothing
with window **9**, resampling every **4 px**.

Normal sections: half-length **40 px**, sampling step **1 px**, border margin
**4 px**. Corridor half-width **12 px**; the background is taken outside it.

**Selection is automatic. There is no manual step.** An object is a band when
its extracted centreline is at least as long as the chain's MTF-50, `49 px`: a
region whose main axis is shorter than the resolution length cannot be asserted
to be a band. Measured on the DIC alone, this separates cleanly at every
threshold, by a factor above six between the shortest retained band and the
longest discarded fragment:

| Threshold | Retained centrelines | Longest discarded |
|---|---|---:|
| q80 | `355`, `175 px` | `26 px` |
| q90 | `273`, `111 px` | `18 px` |
| q95 | `236 px` | `8 px` |

The rule is fixed by the instrument, not by inspection, and it is reproducible
without a human in the loop.

## Metrics

Per band and per section: background level and spread, peak and centroid
position, FWHM, integral width, second-moment width with their statuses, peak,
q95, mass, corridor mean, profile correlation, normalised L1 and L2, asymmetry.
Per band: detected fraction, detected length, gap count, longest gap.
Summaries always report median, IQR, p90, **worst decile**, valid and missing
fractions — never a mean alone.

Widths are reported three ways with **none preferred**.

FSS at q80, q90 and q95, thresholds from the DIC applied to the candidate
unchanged, at scales **1, 2, 4, 8, 16, 24, 32, 48, 64, 96 px**, with the first
scale attaining 0.5, 0.7 and 0.9.

Residual: signed map on a common scale, corridor and background energy, radial
spectrum, directional variograms, associations with the reference and its
**signed** derivatives, and the heuristic typology, labelled as a diagnostic.

## References

- **positive**, the chain's own reproducibility: the repeated final pair gives a
  spurious EVM RMS of `1.363e-4`, that is `4.52 %` of the final DIC EVM RMS.
  This is chain reproducibility, **not** full experimental uncertainty;
- **negative, background**: `homogeneous`;
- **negative, localisation**: `translated`.

Normalised skill is reported alongside raw distances, never instead of them.

## Bootstrap

Paired block bootstrap over sections. Block length **8 sections** (`32 px`,
below the `38.2 px` DIC coherence length so blocks are not artificially
independent), sensitivity at **4** and **16**. Draws **10 000**, seed
**20260731**. Bands resampled separately, averaged at equal weight.

Decision vocabulary at the registered thresholds: `> 0.95` robustly better,
`0.80`–`0.95` probably better, `0.20`–`0.80` indistinguishable, `0.05`–`0.20`
probably worse, `< 0.05` robustly worse.

A conclusion that does not hold at all three block lengths is reported as
scheme-dependent, not as a result.

## Amendment before validation, 2026-07-31

Three changes, made before the campaign is run and before any candidate is read
with the lot 2 to 4 tooling.

### 1. The geometry threshold is q80, chosen on a DIC-only measurement

Segmenting the DIC alone at the three thresholds, minimum area 256 px:

| Threshold | Objects | Second band |
|---|---:|---|
| q80 | 3 | full length, `5639 px`, centreline `175 px` |
| q90 | 2 | reduced to its core, `1666 px`, centreline `111 px` |
| q95 | **1** | **absent** |

At q95 the two-band premise of section 3.4 collapses outright. At q90 the second
band survives only as a fragment, and every worst-band conclusion would rest on
it. **q80 is the only threshold at which both bands exist over their full
length.** The third q80 object, `361 px` with a `26 px` centreline, is what the
manual selection is there to discard.

No candidate field was read to reach this. q80, q90 and q95 all remain in use as
FSS activity thresholds, which is a different question.

![DIC segmented at the three thresholds](figures/observed_evm_band_geometry_p0043_v1/dic_band_thresholds.png)

Blue outlines the objects, orange the pruned skeleton, green the extracted
centreline.

### 2. The regions are ragged, not networks — an earlier reading corrected

An earlier version of this amendment said the skeletons carried hundreds of
loops and were therefore networks. **That was a defect in the measure, not a
property of the data**, and it is corrected here.

The loop count had been taken as `E - V + C` on the eight-connected pixel graph
of the skeleton. Under eight-connectivity three pixels in a corner form a
triangle, so that formula scores every corner of a one-pixel-wide path as a
loop. It reported `661` loops for the first band. Counted properly, on the
region with four-connected background inside eight-connected foreground, the
same band has **63 holes, the largest `32 px`** — and none reaches the `256 px`
resolvable bound. There is no cellular structure.

The branch statistics fail the same way. The first band has `1472` skeleton
branches of which `96 %` are at most `2.4 px`: medial-axis noise from a ragged
boundary, not band intersections. Only `28` reach `16 px`. Branch orientation
on the unfiltered set returns modes at `7, 52, 97, 142` degrees, which are the
four lattice directions and nothing else; orientation is therefore read only
from branches at or above the resolvable length, and the second band has none.

What survives as a real, usable measure of shape:

| Measure | What it says |
|---|---|
| `main_path_share` | how ribbon-like the region is. `0.13` and `0.15` here: wide and ragged, not a thin band |
| `enclosed_holes` with the resolvable bound | whether a cellular pattern exists. Here: none |
| `resolvable_branch_count` | how many genuine sub-branches exist. `28` and `0` |
| area, axis length, elongation | size and extent |

The centreline remains the band axis, which is what the normal sections need.
No statement about connectivity or sub-structure may be read from the raw
skeleton.

### 3. E5 is demoted, and the FSS criterion is replaced

- **E5 is no longer an elimination criterion.** The q90 active fractions are
  already known, and a non-blind criterion must not be able to eliminate a
  candidate on a number known in advance. It is computed and reported.
- **The Pareto FSS criterion becomes the minimum scale reaching FSS 0.7 at
  q90**, lower is better, replacing FSS at a fixed 16 px. The fixed scale was an
  unjustified choice; the attaining scale is the natural summary of the curve.
  A candidate never reaching 0.7 returns `nan` and is reported as such rather
  than ranked.

## Elimination criteria

Each bound and its origin. **None is derived from the archived score table.**

| # | Criterion | Bound | Origin of the bound |
|---|---|---|---|
| E1 | campaign converged, plane-stress residual valid | as archived | solver contract |
| E2 | each DIC band detected in at least half its sections | `0.50` | below half, "the band is reproduced" is not defensible in plain language |
| E3 | median centreline error smaller than the DIC median integral width | band's own width | a band displaced by more than its own width is a different band |
| E4 | no spurious band of area above the minimum object area inside the core and outside every corridor | `256 px` | same resolution bound as E |
E1 to E4 have **never been computed** for any candidate.

E5, the q90 active fraction, was an elimination criterion in the first version
and is **withdrawn** to reported-only status by the amendment above: the active
fractions are already known, and a non-blind criterion must not decide.

## Pareto criteria

A reduced, non-redundant set. Amplitude, position, width, morphology and
continuity each enter once:

| Criterion | Sense |
|---|---|
| integrated mass error, worst band | lower is better |
| median centreline error, worst band | lower is better |
| median integral-width error, worst band | lower is better |
| minimum scale reaching FSS 0.7 at q90 | lower is better |
| corridor fraction of residual energy | lower is better |
| detected fraction, worst band | higher is better |

Every one is taken on the **worst band**, as a vector, never summed. No
weighted sum is formed at any point.

## Interpretation rules

- a candidate is eliminated, dominated, non-dominated, or statistically
  indistinguishable from another. Those are the only verdicts;
- **no winner need exist.** "Several non-dominated" and "no candidate passes"
  are results;
- the primary profile decides; a disagreement with the sensitivity profile is
  reported, never averaged away;
- the `16 %` loading-path systematic on core PEEQ applies uniformly and is
  cited; it is not applied as a correction;
- **no micromorphic parameter is selected by this campaign**, whatever the
  front looks like. Selecting one needs the separately registered
  identification campaign.

## Figures

`dic_and_observed_candidates`, `signed_residuals`, `band_centerlines`,
`normal_profiles_band_<id>`, `width_along_band_<id>`,
`position_error_along_band_<id>`, `fss_heatmaps`, `fss_curves`,
`residual_autocorrelation`, `residual_spectra`, `pareto_matrix`,
`pairwise_probability_matrix`, `falsification_metric_response`.

Every multi-candidate figure uses a common scale.

## Failure conditions

Registered in advance, each a legitimate outcome rather than a defect:

1. **no candidate passes E1 to E5** — reported, and the campaign stops;
2. **the Pareto front holds more than half the candidates** — the criteria do
   not discriminate at this resolution; reported, no ranking published;
3. **the two profiles disagree on a verdict** — reported as profile-dependent;
4. **a conclusion holds at only one block length** — reported as
   scheme-dependent;
5. **the falsification bench mis-ranks a registered defect order** — that metric
   is withdrawn from the decision before any candidate is read.

## Claim boundary

Agreement of an observed EVM field. Nothing about internal stresses, nothing
about PEEQ, which is not a DIC observable. One ROI, one test, one loading path;
a non-dominated candidate is not thereby identified, and no transferable
material internal length follows from any outcome here.

## Deliverable

`validation/observed_evm_candidate_comparison_results.md` and
`reference_data/observed_evm_candidate_comparison_p0043_v1/`, in a **separate
commit** from this one.

## Amendment 2, 2026-07-31: Otsu segmentation and regionprops morphology

The quantile approach of amendment 1 is **replaced**. Three reasons, the first
of them an admission.

**1. The quantile thresholds were arbitrary and the skeleton work was
inconclusive.** Nothing justified q80 over q90 beyond "it keeps two bands", and
the skeleton topology I built on top of it turned out to measure lattice
artefacts rather than structure. That line of analysis is dropped.

**2. Otsu is chosen by the data and separates better.** It maximises
between-class variance, so no quantile has to be argued for. On the DIC it
gives `4.535e-03`, equivalent to q74.5, and yields exactly two objects above
`256 px` with a gap of a factor **34** to the next fragment, against a factor
15 at q80.

| | threshold | objects >= 256 px | gap to next fragment |
|---|---|---:|---:|
| q80 | `4.90e-03` | 3 | `5639` vs `361`, x15 |
| **Otsu** | **`4.535e-03`** | **2** | **`8060` vs `234`, x34** |

The threshold is computed **once on the DIC** and applied unchanged to every
candidate. Recomputing it per field would let each candidate rescale its own
notion of "active" and would hide an amplitude loss entirely; a test locks that.

**3. Morphology comes from `regionprops`**, a standard descriptor set — area,
perimeter, eccentricity, solidity, extent, major and minor axis, orientation,
Euler number — rather than hand-rolled skeleton statistics. This adds
`scikit-image` as a dependency, declared in `pyproject.toml`.

### Why morphology and not area, demonstrated on the negative references

Run on the two negative references only, so no scientific candidate is spent:

| Field | active | objects >= 256 px | eccentricity | minor axis | orientation |
|---|---:|---:|---:|---:|---:|
| DIC | `26.2 %` | 2 | `0.94`, `0.93` | `104`, `72 px` | `-58`, `-46 deg` |
| homogeneous | `0.0 %` | **0** | — | — | — |
| translated | `27.0 %` | **1** | `0.65` | `269 px` | `-15 deg` |

The homogeneous control collapses outright: nothing reaches the DIC threshold.

The translated control is the decisive case. Its active fraction matches the
DIC to within **one point**, `27.0 %` against `26.2 %`, and its morphology is
unmistakably different: one cellular object instead of two elongated bands, an
eccentricity of `0.65` against `0.94`, a minor axis nearly four times larger,
and an orientation `30` to `43` degrees away.

**An area-based score cannot separate these two fields. Morphology separates
them immediately.** That is the specification's premise, now measured rather
than asserted.

![Otsu morphology on the negative references](figures/observed_evm_band_geometry_p0043_v1/otsu_morphology_controls.png)

### What this changes in the registered design

- geometry threshold: **Otsu on the DIC**, replacing q80;
- object descriptors: **regionprops**, replacing the hand-rolled metrics and the
  withdrawn network statistics;
- automatic selection is unchanged in spirit — objects below `256 px` are not
  bands — and no manual step is reintroduced;
- q80, q90 and q95 remain **only** as FSS activity thresholds, a separate use;
- centrelines and normal sections are unchanged: they are still needed for the
  per-section width, position and amplitude metrics, and the Otsu mask feeds
  them instead of the q80 mask.

The four scientific candidates have **not** been segmented with this method.
