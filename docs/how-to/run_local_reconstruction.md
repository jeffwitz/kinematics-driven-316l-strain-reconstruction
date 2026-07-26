# Run a local reconstruction

**Category: How-to.**

## Prerequisites

- a prepared case from {doc}`prepare_case`;
- the MFront environment and compiled behaviour;
- PyPardiso/MKL reported by `fem-inhouse backend`.

## Create or inspect the campaign

```bash
fem-inhouse partition \
  --input data/processed/case-study \
  --output results/local-reconstruction \
  --parts-x 10 --parts-y 10 \
  --padding 150 \
  --increments 20 \
  --constitutive-backend mfront-native-plane-stress \
  --list-pending
```

## Solve

Run one explicit partition:

```bash
fem-inhouse --verbose partition \
  --input data/processed/case-study \
  --output results/local-reconstruction \
  --parts-x 10 --parts-y 10 \
  --padding 150 \
  --increments 20 \
  --constitutive-backend mfront-native-plane-stress \
  --mfront-threads 8 \
  --partition-id PARTITION_ID
```

Or use `--solve-pending` under an external resource manager. Completed
partitions with matching fingerprints are reused.

## Stitch a field

```bash
fem-inhouse partition \
  --input data/processed/case-study \
  --output results/local-reconstruction \
  --parts-x 10 --parts-y 10 \
  --padding 150 \
  --increments 20 \
  --stitch PEEQ
```

Do not change campaign-defining options between calls. A manifest mismatch,
missing MFront library, unavailable PyPardiso or non-converged increment is an
explicit error.

See {doc}`inspect_campaign` for checks and
{doc}`../reference/output_contract` for outputs.
