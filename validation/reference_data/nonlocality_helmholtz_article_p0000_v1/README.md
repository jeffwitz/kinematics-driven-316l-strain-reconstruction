# Helmholtz nonlocality diagnostic on article partition 0

This immutable exploratory campaign filters the saved MFront solution of
article partition 0 on its complete `510×460`-element solved region, including
the 150-pixel padding. Metrics are evaluated only on the retained
`360×310`-element core.

The DIC and FEM `EVM_HISTORICAL` fields are both reconstructed from their
nodal displacements with:

```text
strain_from_displacement
  -> plane_stress_equivalent_strain(engineering shear)
  -> cell_average
```

Only FEM fields are filtered. DIC is never filtered. PEEQ is retained as a
separate internal localization indicator; no PEEQ-to-DIC-EVM amplitude error
is reported.

## Sweep

The tested lengths are `0, 1, 2, 4, 8, 16, 32` physical pixels, or
`0–58.88 um`. The minimum artificial padding is 150 pixels (`276 um`).
Consequently, every positive length satisfies the campaign criterion
`padding / ell >= 4`; the smallest ratio is `4.6875`.

The filter preserves the full-domain mean to at most `8.674e-19`. The maximum
relative discrete residual over both EVM and PEEQ is `5.575e-13`.

## Main EVM result

| Metric | Raw FEM | FEM filtered at 58.88 um |
|---|---:|---:|
| RMSE | `2.6859e-3` | `1.3577e-3` |
| relative L2 | `0.81743` | `0.41319` |
| Pearson correlation | `-0.02920` | `0.09257` |
| top-10% IoU | `0.05030` | `0.13116` |
| standard-deviation ratio to raw | `1.000` | `0.2316` |
| peak ratio to raw | `1.000` | `0.1225` |

RMSE and relative L2 decrease by `49.45%`. Correlation gains `0.1218`, and
top-10% IoU gains `0.08085`. All three exploratory rankings therefore select
the largest tested length, `58.88 um`; the optimum is not bracketed within
this sweep.

The DIC-derived absolute-threshold comparison gives a different warning. At
the DIC 90th-percentile threshold, the best IoU occurs at `14.72 um`
(`0.07740`) and its predicted active fraction is `9.98%`, close to the DIC
reference `10%`. At `58.88 um`, peak attenuation is so strong that no FEM cell
remains above that absolute threshold.

## Permitted interpretation

**Spatial-width hypothesis partially supported on this exploratory
partition.**

Scalar diffusion clearly reduces excessive concentration and improves the
main amplitude and quantile-location metrics. It does not reproduce the DIC
texture: the best correlation remains only `0.0926`, the strongest length
over-attenuates peaks, and the best point lies at the sweep boundary. No
material internal length is identified, and no confirmatory conclusion is
possible without selecting a length on another partition and applying it
unchanged to held-out partitions.

## Reproduction

```bash
.venv/bin/fem-inhouse diagnose-nonlocality \
  --input data/processed/case_study \
  --campaign validation/reference_data/article_100p_pad150_p0000_mfront_v1 \
  --partition-id 0 \
  --output validation/reference_data/nonlocality_helmholtz_article_p0000_v1 \
  --lengths-um 0 1.84 3.68 7.36 14.72 29.44 58.88 \
  --include-peeq \
  --mode exploratory \
  --top-fractions 0.05 0.10 0.20 \
  --dic-quantiles 0.80 0.90 0.95 \
  --minimum-padding-length-ratio 4 \
  --save-fields all
```

The command refuses to overwrite a non-empty campaign unless `--overwrite`
is supplied explicitly.
