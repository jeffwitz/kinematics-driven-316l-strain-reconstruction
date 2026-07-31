# Band geometry and profile metrics

**Category: Reference.** Exact definitions for the object-based comparison of
observed EVM fields. This documents lot 2 of the observed-EVM comparison
specification: the geometry extracted from the DIC field, the metrics read
across it, and the synthetic defects used to test those metrics.

Nothing here runs mechanics, selects a material parameter, or reads a candidate
field while defining geometry.

## Why geometry comes from the DIC field alone

Band corridors, centrelines, normal sections and thresholds are built from the
experimental EVM and then frozen. Re-detecting bands independently in each
candidate would let a candidate move the objects it is judged against, which
silently re-aligns the fields and hides a position error.

## Array convention

Fields are `(nx, ny)` on canonical `(x, y)` axes, so index 0 is x. Geometry is
expressed in pixels of the comparison support.

The EVM support is element-centred while the image crop is nodal, so the two
lattices differ by half a pixel — see the grid contract in
{doc}`observation_operator`. Any centreline coordinate is on the EVM lattice.

## Thresholds

`quantile_thresholds` computes absolute thresholds at q80, q90 and q95 on the
**valid DIC values only**. Including invalid entries would move the threshold
and change every downstream object. No single threshold is treated as truth;
all three are carried.

## Objects

`label_band_objects` thresholds the field, labels eight-connected components so
a diagonal band is one object rather than a dotted line, and drops objects
below `minimum_area_pixels`. That bound is the one place where a size choice
can change which bands exist, so it belongs in a preregistration.

Each surviving object reports area, centroid, orientation, major and minor
axes, elongation, compactness and bounding box. Orientation comes from the
second moments; axis lengths are `4 sqrt(lambda)`. Objects are renumbered
largest first, so "the main bands" is stable across thresholds.

## Centreline

1. `zhang_suen_thinning` reduces the band mask to a one-pixel skeleton. It is
   implemented in this repository rather than taken from scikit-image, which is
   not a dependency. It is deterministic.
2. `prune_skeleton_spurs` removes branches shorter than a declared bound.

   :::{note}
   Under eight-connectivity a pixel sitting beside the trunk touches three
   trunk pixels, so its degree is three and the walk stops before it. One
   residue therefore survives each spur. That is documented rather than
   worked around, because the next step makes it harmless.
   :::

3. `order_centreline` locates the two extremities with breadth-first search,
   then takes the **Euclidean shortest path** between them. Hop count alone is
   not enough: a one-pixel residue beside the trunk offers a diagonal detour
   with the same number of nodes, and a hop-count longest path takes it,
   putting a spurious kink in the centreline. Weighting edges by length keeps
   the centreline straight.
4. `smooth_centreline` applies a declared moving average with the endpoints
   pinned, so smoothing cannot shorten the band. Both raw and smoothed
   polylines are kept.
5. `resample_polyline` gives a regular arc-length step; `tangents_and_normals`
   returns unit tangents and left normals.

## Normal sections

`sample_normal_profile` samples a field along a normal by bilinear
interpolation. A section is returned **marked invalid with a reason** rather
than silently clipped:

| Reason | Meaning |
|---|---|
| `leaves_support` | the section reaches within the border margin |
| `crosses_invalid` | the section touches a masked pixel |
| `nonfinite_sample` | the interpolation produced a non-finite value |

Each section carries a stable identifier.

## Background

`estimate_background` takes the local level from the profile tails, outside the
corridor. The background is **not assumed zero**: a band sits on a spatially
varying field, and subtracting nothing inflates every width and mass. The
estimate reports level, spread and sample count.

`excess_profile` returns $E^+(n) = \max(E(n) - E_\mathrm{bg}, 0)$. Width and
mass are always measured on the excess.

## Width, three definitions, none preferred

All three are reported side by side. They disagree on multimodal and
heavy-tailed profiles, and that disagreement is information rather than noise.

| Definition | Formula |
|---|---|
| FWHM | width at half the excess peak, linearly interpolated |
| integral | $W_\mathrm{int} = \int E^+ \,dn \big/ \max E^+$ |
| second moment | $W_2 = 2\sqrt{\int (n-\bar n)^2 E^+ dn \big/ \int E^+ dn}$ |

`measure_width` returns a status, and the status is what makes a missing width
usable downstream:

| Status | Meaning |
|---|---|
| `ok` | a single peak with two half-maximum crossings |
| `multimodal` | more than one peak above a quarter of the maximum |
| `no_crossing` | the band is wider than the sampled window; the integral definitions still return values |
| `too_weak` | no excess above the declared minimum |
| `peak_at_edge` | the maximum sits at a profile end, so the section is not centred on the band |
| `empty` | the section was excluded |

A section with `no_crossing` is a different failure from one whose band is
absent, and averaging over both would be wrong.

## Position, amplitude, shape, continuity

`measure_position` returns the peak offset, the excess centroid and a detection
flag. `measure_amplitude` returns peak, q95, integrated mass and corridor mean
of the excess. `compare_profiles` returns profile correlation, normalised L1
and L2, and an asymmetry difference.

`continuity_metrics` returns the detected fraction, detected length, gap count
and **longest gap**. The longest gap is what separates a band detected
everywhere but weakly from one lost over a stretch — two failures a fraction
alone cannot distinguish.

## Summaries

`summarise` returns median, mean, interquartile range, p90, **worst decile**,
and the valid and missing fractions. A mean alone is never published: the worst
decile is what catches a band well reproduced along most of its length and lost
at one end.

## Falsification cases

`falsification_cases` builds fields with known defects, so a metric's response
can be checked before it is trusted on real candidates. A metric that cannot
rank a deliberately broken field cannot rank models.

| Generator | Defect |
|---|---|
| `translate_field` | position, sub-pixel capable |
| `scale_amplitude` | amplitude, geometry untouched |
| `change_band_width` | width, with the peak renormalised so amplitude does not move too |
| `remove_region` | a missing band |
| `interrupt_region` | a discontinuous band |
| `add_spurious_band` | a band the reference does not contain |

`standard_cases` assembles the minimal set covering position, amplitude, width,
missing band and continuity.

## Limits

- the valid mask is currently a declaration: no invalid region exists in this
  dataset, so `crosses_invalid` has never fired on real data;
- a corridor placed near the crop edge is exposed to the warp border mode, see
  {doc}`observation_operator`;
- these are field-level tools. They measure agreement of an observed EVM; they
  say nothing about internal stresses, and PEEQ is not a DIC observable.
