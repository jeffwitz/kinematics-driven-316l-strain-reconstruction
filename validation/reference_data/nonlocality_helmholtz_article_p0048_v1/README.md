# Helmholtz nonlocality diagnostic on selection partition 48

This immutable campaign applies the pre-registered stage-1 diagnostic to
partition 48. The filter is solved on the complete `660×610`-element padded
domain, and metrics are evaluated only on the retained `360×310` core.

The DIC and FEM `EVM_HISTORICAL` fields are independently reconstructed from
their nodal displacements with the same operator. Only FEM is filtered. PEEQ
is stored as a separate internal localization indicator and is never compared
to DIC EVM by amplitude RMSE or MAE.

## Frozen sweep

The tested lengths are `0, 1, 2, 4, 8, 16, 32` physical pixels, or
`0–58.88 µm`. All positive lengths are usable under the declared
`padding / ell >= 4` criterion; the smallest ratio is `4.6875`.

The maximum full-domain mean drift is `2.168e-18`, and the maximum relative
discrete filter residual is `6.074e-13`.

## Primary pre-registered evidence

| Metric | Raw FEM | FEM at `58.88 µm` | Change |
|---|---:|---:|---:|
| RMSE | `2.6897e-3` | `9.5182e-4` | `-64.61%` |
| relative L2 | `0.80956` | `0.28649` | `-64.61%` |
| Pearson correlation | `0.29828` | `0.61597` | `+0.31768` |
| top-10% IoU | `0.15984` | `0.28224` | `+0.12240` |
| DIC-q90 absolute-threshold IoU | `0.16757` | `0.30852` | `+0.14096` |
| predicted active fraction at DIC q90 | `19.66%` | `14.09%` | DIC: `10%` |

Correlation, equal-fraction top-10% IoU, and the absolute DIC-q90 IoU all
increase monotonically over the tested sweep and select `58.88 µm`. The same
candidate also maximizes the absolute-threshold IoU at DIC quantiles 80% and
95%. Unlike partition 0, the q90 active area does not collapse to zero.

## Selection and interpretation

`58.88 µm` is selected for unchanged application to held-out partitions. It
is nevertheless the upper sweep boundary, so this campaign does not bracket
an optimum.

The evidence strongly supports a spatial-width contribution **on the
selection partition**. The stage-level conclusion remains:

> **Spatial-width hypothesis partially supported pending held-out
> confirmation.**

No material internal length is identified. A held-out failure must also
trigger investigation of differences between the historical Abaqus
reconstruction visible in the article and the current in-house implementation.

The machine-readable decision is in `selection-report.json`.

## Reproduction

```bash
.venv/bin/fem-inhouse diagnose-nonlocality \
  --input data/processed/case_study \
  --campaign results/reconstruction-100 \
  --partition-id 48 \
  --output results/nonlocality-diagnostic-p0048 \
  --lengths-um 0 1.84 3.68 7.36 14.72 29.44 58.88 \
  --include-peeq \
  --mode exploratory \
  --top-fractions 0.05 0.10 0.20 \
  --dic-quantiles 0.80 0.90 0.95 \
  --minimum-padding-length-ratio 4 \
  --save-fields all
```

