# EBSD-derived microstructure product: blocking QA

Status: **candidate not validated; do not use for microstructural screening or promote to golden**.

This audit concerns the local candidate generated from
`/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5`. The source
orientation fields are documented as per-pixel grain-mean fields. The raw
Euler-component neighbour difference, including the previously reported
`1410 deg`, is not a crystallographic misorientation.

## Segmentation identity

The counts that must not be conflated are:

| object | count |
| --- | ---: |
| exact stored Euler triplets | 29,238 |
| four-connected components of exact triplets | 224,795 |
| final IDs in the current candidate | 29,238 |

The current builder uses the exact triplet label directly as `grain_id`. It
therefore merges spatially disconnected components carrying the same triplet;
the previous connectivity audit found 195,557 additional components. This is
not an acceptable grain-area contract until the upstream segmentation/label
provenance explains those duplicates. The current 29,238 IDs are source colour
regions, not independently audited connected grains.

The previous exact-connectivity statistics were: 32.9% of exact labels had one
component; 87.7% of components had one pixel, 95.0% had at most four pixels,
and those small components contained 1.77% and 2.16% of pixels respectively.
This discrepancy requires inspection, not silent aggregation.

The twenty largest current colour regions were each one connected component in
the audited candidate. The largest was colour/ID `24186`, area 235,628 px²,
bbox `(696,1508,1458,2105)`, not touching the global border, Euler
`(249.0227,46.4284,55.2224)`. Other large regions were IDs `29203`
(182,399 px²), `26286` (147,233), `1568` (139,325), and `1791` (134,526).
The complete top-20 audit is retained in the JSON companion and must be
repeated after a provenance-backed grain-ID decision.

## Source validity

The source HDF5 contains displacement U/V, Euler `phi1/Phi/phi2`, maximum
Schmid factor, and topography. It contains no phase ID, indexed/unindexed
mask, confidence/index-quality field, material mask, or original grain-ID
dataset. All source Euler values are finite; the current `valid_mask` is thus
an assumption (`all finite Euler triplets`) rather than a provenance-backed
material mask. The value 1449 occurs as a finite triplet in only 46 pixels,
so it must remain an explicitly audited source value rather than being silently
reclassified.

## Invariants that passed on the completed candidate

On the completed candidate before a later interrupted local regeneration:

* non-ambiguous nearest-boundary incidence was 100% (0 bad pixels over
  10,738,426 non-ambiguous pixels);
* `nearest_neighbor_grain_id != grain_id` on that support;
* deterministic 1,000-pixel checks reproduced the boundary-side `mprime_max`
  and residual-Burgers fields exactly (maximum error 0);
* cubic misorientation was 0.001636--62.3763 degrees;
* `mprime_max` was 0.4188--1.0 and residual Burgers was
  1.21e-5--0.7622, within their expected bounds;
* a deterministic crystallographic A/B check on 20 interfaces gave zero
  transpose discrepancy for both the Luster--Morris and residual-Burgers
  matrices. A larger sample should be run when the product is regenerated.

## Remaining blockers

1. The final grain-ID semantics are unresolved: exact colour labels currently
   aggregate disconnected components, while exact connectivity produces a
   radically different population. No grain-size or nearest-GB result should
   be interpreted physically until this is resolved from source provenance.
2. The source has no indexed/material mask. `valid_mask = ones` must remain
   explicitly `ASSUMED`; it is not evidence that all finite Euler pixels are
   indexed material.
3. The current trace fields use one global PCA tangent per grain pair and are
   propagated across the interface. They are not local tangents at the nearest
   boundary point. `slip_trace_to_gb_angle` and `crossing_factor` must not be
   promoted to golden in this form.

Consequently the product is **FAIL / candidate not validated** for the
microstructural \(k_\perp\) screening. The crystallographic formulas are
mathematically bounded, but that does not repair the unresolved segmentation,
validity, and local-trace contracts.

No mechanics, Krylov, FEMU, SRIX, registration, or \(k_\perp\) screening was
run in this QA.
