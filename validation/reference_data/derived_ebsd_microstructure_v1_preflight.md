# EBSD grain-region product preflight

The source `CP_dataset.h5` is documented as a per-pixel grain-mean EBSD
export. The Euler triplets are therefore treated as the already supplied
region colours; this workflow does not infer grains from a physical angular
threshold. The derived product materialises those regions and computes the
requested geometric/crystallographic indicators.

## Source

- File: `/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5`
- SHA256: `e2684b5353a53b03871c8ced5ed457c3d2de88de3fb8b7560071bf6d3cda28fb`
- Orientation arrays: `3600 × 3100`, `float64`, degrees
- Stored orientation colours/regions: `29,238`
- Raw Euler-component neighbour difference is diagnostic only; the earlier
  `1410°` value is not a crystallographic misorientation.

## Derived global indicators

- Source regions: `29,238`
- Internal interfaces: `77,773`
- Equivalent-diameter area range: `1`–`235,628 px²` (median `6 px²`)
- Cubic neighbour misorientation: `0.0016°`–`62.3763°`
- Luster–Morris `m'`: `0.4188`–`1.0` (median `0.9084`)
- Sign-invariant residual Burgers: `1.21e-5`–`0.7622` (median `0.2622`)
- M20 crop: `15` source regions

The minimum/median grain-area values describe the supplied region map and are
not used to reject it. Any later physical interpretation must retain the
source segmentation provenance and the global-border flag.

## Product

The candidate HDF5 is generated locally because its payload is large (about
`371 MB`). It is not promoted to golden and is not committed. The committed
builder and report/manifest provide deterministic reconstruction and the
source SHA256.

No mechanics, inverse calculation, registration, or `k_perp` screening was
run.
