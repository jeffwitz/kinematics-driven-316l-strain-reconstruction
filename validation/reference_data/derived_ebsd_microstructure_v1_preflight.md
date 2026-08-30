# Exact EBSD plateau segmentation QA

The previous “quantisation gap” gate was removed. Raw Euler-component differences are retained only as an export diagnostic; they are not crystallographic misorientations and do not gate segmentation.

## Source audit

- Source: `/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5`
- SHA256: `e2684b5353a53b03871c8ced5ed457c3d2de88de3fb8b7560071bf6d3cda28fb`
- Grid: `3600 × 3100`
- Orientation datasets: `phi1`, `Phi`, `phi2`; `float64`, degrees, finite
- Export metadata: rows `400:4000`, columns `1211:4311` of `[4400, 5400]`; advertised `pixel_size_um = 1.84` remains semantically unresolved
- Rotation check: maximum orthogonality error `6.66e-16`; determinant range `[0.9999999999999994, 1.0000000000000007]`

The raw diagnostic has `96.0815%` exact neighbour equality and a smallest positive component difference of `0.001811°`. The reported `1410°` is only a naive Euler-component difference, not a physical misorientation.

## Exact-label connectivity result

Using exact stored triplets followed by four-connected components gives:

- `29,238` exact orientation labels;
- `224,795` connected components;
- `195,557` additional components beyond the labels;
- only `32.916%` of labels have one component;
- median component area `1 px²`, mean `49.645 px²`;
- `87.739%` of components have area `1 px²`, `95.043%` area ≤`4 px²`.

The resulting global QA maps are visibly fragmented into narrow strips and isolated pixels. This exact segmentation is therefore a **pathological candidate** for a grain-mean microstructure product, despite the high exact-neighbour equality. It should not be promoted into a grain/interface HDF5 product without auditing the export semantics.

## True crystallographic neighbour check

A deterministic sample of 2,000 distinct-Euler neighbour pairs gives cubic minimum misorientations from `0.001636°` to `61.9104°` (median `29.7733°`), within the expected cubic range. The rotation convention is internally valid; it does not rescue the connectivity pathology.

QA figures were generated locally under `/tmp/derived_ebsd_microstructure_v1/segmentation_figures/` and are not versioned.

No HDF5 product, descriptor table, mechanical calculation, or (k_\perp) screening was generated. The next action is source-export/plateau curation, not an angular threshold or physical interpretation of the fragmented labels.
