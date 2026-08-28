# Compare DIC and FEM fields

**Category: How-to.**

## Prerequisites

- co-registered DIC and FEM displacement;
- a completed campaign;
- manifest-defined core bounds;
- a declared observation operator and thresholds.

## Validate a coupled campaign

```bash
fem-inhouse validate-coupled-nonlocal --help
```

Pass the prepared input, local campaign, coupled campaign, partition and output
paths required by the installed revision. The validator reconstructs the same
historical EVM from DIC and FEM displacements and evaluates only the core.

## Generate common-scale figures

```bash
fem-inhouse plot-coupled-alpha-fields --help
```

Provide one local and the compared coupled campaigns. Use common EVM and PEEQ
color scales and a symmetric signed-error scale. The metadata records source
hashes, core bounds, percentiles and explicitly confirms that no EVM
post-filter was applied.

## Run the output-only diagnostic

Use `diagnose-nonlocality` only when the question is whether an existing local
field lacks spatial width. Its filtered output is diagnostic and must not
replace raw coupled EVM in the primary comparison.

## Test what the material maps contribute

Create controls without modifying the canonical input:

```bash
fem-inhouse prepare-material-map-control \
  --input data/processed/case_study \
  --output data/processed/control-homogeneous \
  --mode homogeneous \
  --yield-stress-mpa 124 \
  --hardening-coefficient-mpa 380

fem-inhouse prepare-material-map-control \
  --input data/processed/case_study \
  --output data/processed/control-translated \
  --mode translated \
  --shift-x-pixels 600 \
  --shift-y-pixels 500
```

Run both derived inputs with the same local solver configuration as the
mapped reference, then compare the three completed campaigns:

```bash
fem-inhouse validate-material-map-controls \
  --input data/processed/case_study \
  --mapped-campaign results/local-mapped \
  --homogeneous-campaign results/local-homogeneous \
  --translated-campaign results/local-translated \
  --partition-id 43 \
  --output validation/material-map-controls
```

The translated control moves `sigma_y` and `K` jointly, preserving their
distributions and pixel-wise pairing. A drop relative to the mapped reference
therefore measures spatial information rather than a change in marginal
material statistics. PEEQ remains a model output and is not compared with DIC
in amplitude.

See {doc}`../reference/scientific/observation_operator` for comparison rules and
{doc}`../explanation/missing_spatial_interaction` for interpretation.
