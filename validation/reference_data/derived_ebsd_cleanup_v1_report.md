# Conservative EBSD grain-map cleanup screening

This is a reversible screening product, not a golden EBSD dataset. The source
Euler fields are treated as upstream grain-mean colours. Exact colours are
split with `skimage.measure.label` using 4-connectivity; no angular clustering
or physical misorientation threshold is introduced.

Small components with (A\le A_{\min}) are handled in one pass. Components with
\(A>A_{\min}\) are immutable seeds. A small component is assigned only to a seed
neighbour, first by shared 4-neighbour boundary length and then, when needed, by
minimum cubic misorientation. Components without an unambiguous seed target are
retained and marked ambiguous. Small-to-small propagation is deliberately
forbidden. Raw orientation labels and raw connected-component IDs are stored in
every local compressed variant.

## Source and connectivity

- Source: `/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5`
- Grid: `3600 × 3100`
- Exact stored Euler labels: `29,238`
- 4-connected components: `224,795`
- 8-connected components (diagnostic): `145,505`

The 8-connectivity diagnostic shows that raster diagonals explain part, but not
all, of the 4-connected fragmentation. The cleanup therefore remains explicitly
4-connected and topological.

## Variant comparison

| \(A_{\min}\) (px²) | final grains | reassigned pixels | reassigned fraction | ambiguous components | M20 clean grains |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 27,609 | 197,186 | 1.7669% | 46 | 7 |
| 2 | 18,916 | 214,565 | 1.9226% | 39 | 7 |
| 4 | 11,190 | 240,737 | 2.1571% | 46 | 7 |
| 8 | 4,963 | 279,923 | 2.5083% | 67 | 7 |

The \(A_{\min}=4\) variant therefore reassigns only 2.16% of pixels, in line
with the prior component-area audit. This is a screening result, not a claim
that every removed component was physically spurious.

For the eligible small components, the (f_{\max}) contact fractions are
reported in the JSON companion. Fusion misorientation quantiles are also
recorded; no universal angular cutoff was used.

## Reversibility and use

Local artifacts are written below `/tmp/derived_ebsd_cleanup_v1/`:

- `raw_maps.npz`: exact orientation labels, raw connected components and raw
  component metadata;
- `amin_1.npz`, `amin_2.npz`, `amin_4.npz`, `amin_8.npz`: each cleaned map plus
  target component, reason, ambiguity, contact fraction and fusion
  misorientation;
- `report.json`: complete machine-readable statistics and provenance.

The source orientation of a reassigned pixel is replaced by the receiving
seed's grain-mean orientation. No weighted orientation average is performed.
The next step, after choosing a conservative variant, is to recompute all
grain areas, boundaries, distances, misorientation, (m'), Burgers and local
trace descriptors from that final map. No such mechanical or constitutive
calculation is performed by this screening.

## Current recommendation

`A_min = 4 px²` is a reasonable candidate for review because it changes only
2.16% of pixels and leaves the M20 crop with seven cleaned grain IDs. It must
remain a candidate until the reassignment maps and the ambiguous/tie cases are
visually inspected. No variant is promoted to golden by this report.
