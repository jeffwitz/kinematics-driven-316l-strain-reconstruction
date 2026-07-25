# Held-out Helmholtz confirmation on partition 42

This immutable campaign tests the `58.88 µm` candidate selected on partition
48. Partition 42 was declared before P48 was calculated, and the length is
applied without adjustment.

Only two fields are compared:

```text
ell = 0
ell = 58.88 µm
```

The filter covers the complete `660×610`-element padded domain. Metrics use
only the `360×310` retained core. All candidates satisfy
`padding / ell >= 4`.

## Automatic confirmatory result

| Pre-registered requirement | Observed | Threshold | Result |
|---|---:|---:|---|
| correlation gain | `+0.30292` | `>=0.05` | pass |
| relative-L2 reduction | `65.43%` | `>=5%` | pass |
| top-10% IoU gain | `+0.14254` | `>=0.02` | pass |
| relative mean drift | `4.269e-16` | `<=1e-10` | pass |

The built-in confirmatory workflow reports
`criteria_met_on_this_partition = true`.

## Additional absolute-threshold result

| Metric | Raw FEM | Filtered FEM |
|---|---:|---:|
| RMSE | `2.6677e-3` | `9.2209e-4` |
| Pearson correlation | `0.40068` | `0.70360` |
| top-10% IoU | `0.13340` | `0.27594` |
| DIC-q90 absolute-threshold IoU | `0.17735` | `0.25726` |
| predicted active fraction at DIC q90 | `20.41%` | `7.74%` |

The absolute q90 IoU gain is `+0.07990`, above the pre-registered `0.02`.
The filtered active fraction lies inside the declared `[5%,20%]` interval.
The additional anti-collapse requirements therefore pass.

## Stage-1 conclusion

> **Spatial-width hypothesis supported.**

The same `58.88 µm` candidate substantially improves amplitude and spatial
metrics on the P48 selection partition and the held-out P42 confirmation
partition. This supports the stage-1 claim that a width mismatch explains a
significant part of the FEM-DIC discrepancy.

The selected point remains the upper boundary of the original sweep, and only
one held-out partition has been evaluated. The result does not identify a
material internal length and does not validate a coupled nonlocal constitutive
model.

The machine-readable assessment is in `confirmation-report.json`.

## Reproduction

```bash
.venv/bin/fem-inhouse diagnose-nonlocality \
  --input data/processed/case_study \
  --campaign results/reconstruction-100 \
  --partition-id 42 \
  --output results/nonlocality-confirmation-p0042 \
  --lengths-um 0 58.88 \
  --include-peeq \
  --mode confirmatory \
  --decision-thresholds validation/nonlocality_p42_decision_thresholds.yaml \
  --top-fractions 0.05 0.10 0.20 \
  --dic-quantiles 0.80 0.90 0.95 \
  --minimum-padding-length-ratio 4 \
  --save-fields all
```

