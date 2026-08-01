# ROI qualification for a diffusion-length experiment — results

Date: 2026-08-01
Filter: the ten conditions of the 2026-08-01 review, implemented in
`fem_inhouse.workflows.qualify_roi`.
Machine-readable result:
`reference_data/roi_qualification_v1/roi_qualification.json`.
Paired fields: `reference_data/roi_qualification_v1/fields/`.

Reads the DIC and one **local** run per ROI. No coupled computation was used to
produce any conclusion here. The three new local runs cost 18 minutes each
because the nonlocal fixed point is absent rather than zero-valued.

## Short answer

**All four ROIs are rejected, on both DISFlow profiles.** The problem is not the
choice of P43. In every ROI tested, including the three the automatic ranking
preferred, the local model already produces bands **as wide as or wider than the
DIC**, so the defect a scalar regularisation of `p` could correct does not
exist.

| ROI | rank | objects DIC/local | `w_DIC` | in MTF-50 | `w_local` | `w_DIC/w_local` | failed |
|---|---:|---|---:|---:|---:|---:|---:|
| P017 | 1 | 5 / 3 | `19.9` | `0.41` | `21.2` | `0.87` | 7 of 7 |
| P084 | 2 | 2 / 1 | `20.7` | `0.42` | `19.9` | `1.06` | 6 of 7 |
| P058 | 3 | 2 / 1 | `20.0` | `0.41` | `21.6` | `1.04` | 7 of 7 |
| P043 | 7 | 2 / 1 | `19.4` | `0.40` | `21.1` | `0.92` | 7 of 7 |

The filter needs `w_DIC/w_local >= 1.33`. The best any ROI reaches is `1.06`.

## The width is the same everywhere, and that is the finding

Four ROIs with very different DIC morphologies — 2, 5, 2 and 2 objects, Otsu
minor axes ranging from `18` to `259 px` — give a band width of
`19.4` to `20.7 px`, a spread of `7 %`. The local model gives `19.4` to
`21.6 px` on the same sections.

That is `0.40` to `0.42 MTF-50` in every case. The measured bands sit **at the
resolution limit of the measurement chain**, whose MTF-50 is `49 px`.

A width that does not vary with the structure, and that both a speckled
measurement and a smooth simulation reproduce to within `10 %`, is not
characterising the material. It is characterising the observation.

### This was checked, and it is not an estimator artefact

The integral-width estimator was tested on synthetic Gaussian bands of known
FWHM before this was concluded:

| true FWHM | `7.1` | `14.1` | `23.6` | `35.3` | `58.9` | `94.2` | `141.3` |
|---|---:|---:|---:|---:|---:|---:|---:|
| measured | `7.5` | `15.0` | `24.0` | `29.8` | `33.3` | `34.6` | `35.0` |

It is linear to about `24 px` and saturates near `35 px`, the ceiling set by the
`40 px` section half-length and the background taken beyond `+/-12 px`. **The
measured `20 px` sits in the linear range**, so those numbers are real widths,
not a floor. The hypothesis that the constancy was an analysis artefact was
tested and rejected.

## Why the directional condition fails

Identification of `ell` from a width needs

```
r_w = w_local - w_DIC        the defect to correct
s_l = d w_model / d log ell  the direction coupling moves in
```

with `r_w * s_l < 0`. Coupling can only widen, so `s_l > 0`, and the condition
needs `r_w < 0`: the local model must be **too narrow**.

Measured, `r_w` is positive or near zero in all four ROIs. The local model is
already `2` to `9 %` too wide. There is nothing for widening to correct, and the
P43 matrix measured exactly that: width error `3.64 px` for local against
`5.64`, `6.19`, `7.05` for `alpha = 1, 2, 4`.

## Two problems in the filter as specified

**The MTF-50 width criterion is not measurable.** It asks for
`w_DIC > 1.5 MTF-50 = 73.5 px`, but the estimator that would test it saturates
at about `35 px` with the registered section geometry. No data can satisfy it as
written. Either the section half-length and the background corridor are enlarged
first, or the criterion is restated — and given the measured widths of `20 px`,
enlarging them would find nothing.

**The width has three definitions in this repository**, differing by a factor
five on the same field: section integral `19.4 px`, selection heuristic
`36.4 px`, Otsu minor axis `104 px` on P43. A threshold in MTF-50 therefore
means three different things. This filter uses the section integral, because it
is the operator a later campaign would test with, and reports the others beside
it.

## What the ranking cannot see

The automatic ranking scores DIC morphology: aspect ratio, area fraction,
contrast. It ranked P017 first and P043 seventh. On the qualification filter,
both fail identically, and P017 fails slightly worse because its DIC segments
into **five** objects against the local model's three.

Band width correlates with aspect ratio at `-0.61` across the hundred
partitions: the structures that look most band-like are the narrowest. The
ranking therefore selects **against** the property this experiment needs. That
is a defect of the selection criterion, not of any ROI.

## Profile agreement

The last condition of the filter — the same conclusion under both DISFlow
profiles — is satisfied, in the sense that both reject all four. Widths agree to
within `12 %` between profiles and no verdict changes.

## What this licenses

- **P43 is not an unlucky ROI.** Three better-ranked ROIs fail the same way, so
  the result is a property of this dataset at this partition size, not of a
  choice;
- **no coupled campaign should be launched on any of these four**;
- the ranking criterion must be replaced before another ROI is proposed: it
  currently prefers narrow bands, which is the opposite of what is needed;
- widening the search is legitimate — a different partition size might contain
  wider structures, since the `10 x 10` layout fixes the core at
  `360 x 310 px` — but nothing in the present hundred partitions qualifies.

## What it does not license

Nothing here refutes the micromorphic formulation. It says this measurement,
on this specimen, at this partition size, does not contain the contrast needed
to identify a diffusion length from band width. A dataset whose bands were
resolved well above MTF-50, and whose local model under-predicted their width,
would be a different experiment.

Nor does it say the coupling does nothing: the P43 matrix showed it corrects
integrated amplitude and mass robustly. It corrects the wrong defect for this
purpose.
