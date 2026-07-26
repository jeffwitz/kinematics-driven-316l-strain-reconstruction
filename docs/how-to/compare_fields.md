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

See {doc}`../reference/observation_operator` for comparison rules and
{doc}`../explanation/missing_spatial_interaction` for interpretation.
