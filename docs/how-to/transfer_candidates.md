# Transfer selected candidates

**Category: How-to.**

## Prerequisites

- no more than three non-dominated candidates selected from calibration;
- all physical and numerical parameters frozen;
- an independent band-containing region;
- the same DIC observation operator and metric definitions.

## Prepare the transfer manifest

```bash
fem-inhouse identify-nonlocal prepare-transfer-validation \
  --config configs/joint_nonlocal_identification.yaml \
  --dry-run
```

Review that $\ell$, $\alpha$, $H_\chi$, $H_{\mathrm{ref}}$, local material
parameters, loading snapshots and solver policy are unchanged. Only
region-specific geometry, boundary data and local descriptor maps may differ.

## Execute explicitly

Run each manifest entry with the ordinary coupled reconstruction command.
Never optimize parameters on the transfer region. Collect the same amplitude,
localization and spatial-scale metrics and compare candidate ranking between
calibration and transfer.

## Interpret

An unchanged candidate that remains effective supports transferability. A
candidate requiring retuning is an effective reconstruction parameter for the
calibration region, not an identified material internal length.

See {doc}`../explanation/scope_and_prediction` for the claim boundary.
