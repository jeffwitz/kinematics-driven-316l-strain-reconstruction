# Working EBSD microstructure product — A_min = 8 px²

Status: **WORKING_DERIVED_PRODUCT / NOT_GOLDEN**.

The source Euler map is cleaned with the previously recorded one-pass,
seed-only topological rule. Components with (A>8) px² are immutable seeds;
components with (A\le8) px² are assigned only to seed neighbours using shared
boundary contact first and cubic misorientation as a tie-break. Ambiguous
components remain explicit and are excluded from nearest-neighbour and local
trace descriptors.

## Product summary

- Source grid: `3600 × 3100`
- Raw 4-connected components: `224,795`
- Working grains before exact-orientation canonicalisation: `4,963`
- Working grains after exact-orientation canonicalisation: `4,958` (`5` adjacent same-orientation regions merged)
- Pixels reassigned: `2.5083%`
- Ambiguous components: `67` (`6.0e-6` of pixels)
- Interfaces before canonicalisation: `11,473`; after: `11,465`
- Zero-degree interfaces before canonicalisation: `4`; after: `0`
- M20 working grains: `7`
- M20 areas are inherited from the global working map.

The raw orientation labels and raw connected-component map are retained in
`/raw`; the cleanup provenance is retained in `/cleanup`, including source
component, target, contact fraction, reason and merge misorientation.

## Recomputed indicators

All indicators below were recomputed from `working_grain_id`; no old direct-label
fields were patched.

| indicator | range / value |
| --- | ---: |
| grain area | 1–238,187 px² (median 14 px²) |
| cubic boundary misorientation | 0.03596–62.2129° (median 38.6733°) |
| nearest-neighbour (m'_{\max}) | 0.4215–1.0000 (median 0.7868) |
| nearest-neighbour residual Burgers | 2.19e-4–0.7638 (median 0.3743) |
| valid local trace descriptors | 96.8118% of pixel×system entries |

Pixels marked by `cleanup_ambiguous` or whose nearest boundary point is
ambiguous/triple-junction have
undefined nearest-neighbour crystallographic descriptors. Their raw and
working IDs remain available for inspection.

The nearest-point masks are explicitly propagated from the distance-transform
indices (`nearest_boundary_ambiguous` and `nearest_triple_junction`), rather
than testing the ambiguity of the current interior pixel.

## Local grain-boundary geometry

Each interface point receives a local PCA tangent/normal from points of the same
interface within radius 3 px. Interior pixels inherit the geometry of the
actual nearest boundary point, stored in:

```text
nearest_gb_point_rc
nearest_gb_local_tangent_xy
nearest_gb_local_normal_xy
nearest_gb_local_trace_quality
```

A deterministic radius check on 100 interfaces gave median local-trace quality
`0.8402`, `0.9032`, `0.9516` for radii 2, 3 and 5 px. The median unsigned tangent
change between 2→3 and 3→5 px was about `2.06°`; the maximum was `53.1°` on
short/poorly resolved interfaces. Radius 3 is therefore retained as a
declared working choice, not as a physical parameter. Low-quality and
triple-junction regions are masked by `slip_trace_descriptor_valid`.

## QA gates

- nearest boundary is incident to the current working grain on 100% of
  11,062,671 non-ambiguous/non-triple-junction pixels;
- nearest neighbour is never the current grain on that support;
- local tangents and normals have unit norm to float32 precision;
- trace angles lie in `[0°, 90°]`;
- cubic misorientation lies in `[0°, 62.8°]`;
- (m'\) lies in `[0,1]`;
- residual Burgers lies in `[0,\sqrt{2}]`;
- M20 is cropped from the global map and inherits global grain areas.

The source provides no material/indexed mask, so `valid_mask` is explicitly
marked `ASSUMED` (all finite Euler pixels). The source-advertised EBSD scale is
not treated as independently verified; geometric fields remain in pixels.

## Local artifact

The HDF5 payload is intentionally not committed:

```text
/tmp/derived_ebsd_microstructure_working_A8_v1/derived_ebsd_microstructure_working_A8_v1.h5
SHA256 17fda698c61b3669b33367cb6cee354d652e87d7481d1571f159e6830007b5be
```

Figures are in `/tmp/derived_ebsd_microstructure_working_A8_v1/figures/`.
No mechanics, Krylov, FEMU, SRIX or (k_\perp) screening was run.

The product remains a working derived product, not a golden dataset. The
canonicalisation uses exact final Euler triplets plus 4-connectivity after the
reversible A_min=8 cleanup; no angular merge threshold is applied.
