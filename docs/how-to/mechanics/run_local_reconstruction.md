# Run a local reconstruction

**Mode:** how-to  
**Domain:** reconstruction

## Prerequisites

Prepare the inputs with {doc}`../data/prepare_dic_case`, build the MFront
behaviour if needed, and verify `fem-inhouse backend` before starting.

## List and solve partitions

```bash
fem-inhouse partition \
  --input data/processed/case-study \
  --output results/local-reconstruction \
  --parts-x 10 --parts-y 10 --padding 150 --increments 20 \
  --constitutive-backend mfront-native-plane-stress --list-pending

fem-inhouse --verbose partition \
  --input data/processed/case-study \
  --output results/local-reconstruction \
  --parts-x 10 --parts-y 10 --padding 150 --increments 20 \
  --constitutive-backend mfront-native-plane-stress \
  --mfront-threads 8 --partition-id PARTITION_ID
```

Completed partitions are reused only when their fingerprints match. Keep the
input manifest, commit and backend options unchanged between calls.

## Verify and stitch

Inspect the report for converged increments, true mechanical residual,
cutbacks, plane-stress residual and the manifest-defined core. Then stitch a
field with:

```bash
fem-inhouse partition --input data/processed/case-study \
  --output results/local-reconstruction --parts-x 10 --parts-y 10 \
  --padding 150 --increments 20 --stitch PEEQ
```

Missing libraries, manifest mismatches and non-converged increments are
explicit failures; do not repair them by changing the recorded history.
