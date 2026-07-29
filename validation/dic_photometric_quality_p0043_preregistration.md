# P43 photometric quality versus FEM/DIC agreement — preregistration

Date: **2026-07-29**

Status before execution: **registered, no result inspected**

## Question

Does local violation of image brightness constancy explain a measurable part
of the remaining FEM/DIC EVM discrepancy after symmetric image-level
observation?

This is a measurement-quality diagnostic. It does not alter DISFlow, the
mechanics, the observation replay or any micromorphic parameter.

## Immutable inputs

- reference frame: `000294.tif`;
- final frame: `000334.tif`;
- prepared canonical final displacement:
  `data/processed/case_study/displacement_x_mm.npy` and
  `displacement_y_mm.npy`;
- P43 V3 primary-profile replays for `alpha=0,1,2,4` under
  `validation/reference_data/dic_symmetric_observation_p0043_v1/`;
- physical sampling: `0.00184 mm/pixel`;
- retained P43 core: manifest-defined, never inferred from array shape.

## Photometric residual

At source-image pixel \(x\), evaluate

\[
r_I(x)=
\left|
I_{40}\!\left(x+u_{\mathrm{DIC}}(x)\right)-I_0(x)
\right|.
\]

The final image is sampled bilinearly at the direct DIC destination
coordinate. Pixels whose destination leaves the recorded crop are invalid.
No intensity normalisation, local contrast correction, spatial filtering or
post-hoc illumination fit is allowed in the primary analysis.

The canonical/image conversion must use the verified project contract:
canonical `ux` is image-row displacement and canonical `uy` is image-column
displacement. The residual is reported in 8-bit grey levels.

## Agreement map

For each replay, define

\[
e_\varepsilon(x)=
\left|
\varepsilon_{\mathrm{eq}}^{\mathrm{FEM,observed}}(x)
-
\varepsilon_{\mathrm{eq}}^{\mathrm{DIC}}(x)
\right|.
\]

Both arrays come directly from the immutable V3 replay. PEEQ is not used.

## Pre-registered read-outs

For each `alpha`:

1. Pearson correlation between \(r_I\) and \(e_\varepsilon\);
2. Spearman rank correlation;
3. mean absolute EVM error in ten fixed photometric-residual deciles;
4. unmasked RMSE, relative L2 and Pearson FEM/DIC agreement;
5. the same three metrics after excluding only the worst 10 % of pixels
   according to \(r_I\).

The q90 threshold is defined once from the photometric residual on the full
P43 core and applied unchanged to all four candidates. No alternative
threshold is selected after inspecting the result.

## Interpretation

There is no pass/fail threshold in this first baseline.

- Positive residual/error association supports a measurement-quality
  contribution to local discrepancy.
- A negligible association is a negative result and must be reported.
- A change in model ranking after q90 masking is reported as sensitivity, not
  as permission to discard pixels from the primary scientific metric.
- Unmasked metrics remain the primary reported comparison.

## Outputs

```text
validation/reference_data/dic_photometric_quality_p0043_v1/
  report.json
  photometric_residual.npy
  valid_mask.npy
  decile_metrics.csv
validation/figures/dic_photometric_quality_p0043_v1/
  photometric_quality_and_error.png
  photometric_deciles.png
validation/dic_photometric_quality_p0043_results.md
```

No mechanical calculation and no non-local identification may be launched by
this campaign.
