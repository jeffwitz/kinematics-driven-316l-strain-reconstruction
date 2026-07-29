# P43 DIC uncertainty propagation — preregistration

Date: **2026-07-29**

Status before execution: **registered, no propagated result inspected**

## Question

How sensitive are the archived V3 FEM/DIC metrics to a displacement
perturbation with the amplitude and spatial structure measured by the
candidate repeated-state DISFlow pair?

This is a light observation-level propagation. It does not rerun material-map
identification, mechanics or micromorphic coupling.

## Immutable inputs

- final and candidate-repeat images: `000334.tif` and `000335.tif`;
- prepared canonical DIC displacement:
  `data/processed/case_study/displacement_x_mm.npy` and
  `displacement_y_mm.npy`;
- primary V3 P43 replays for `alpha=0,1,2,4` under
  `validation/reference_data/dic_symmetric_observation_p0043_v1/`;
- DISFlow profile: `legacy_script_2021`;
- pixel size: `0.00184 mm`;
- solve and core bounds: read from the immutable replay reports.

The candidate repeat is not certified as a static pair. Its displacement is
therefore an **upper-bound residual field**, not a random-noise sample.

## Residual displacement

Run DISFlow once from `000334.tif` to `000335.tif` over the canonical recorded
crop with `legacy_script_2021`. Query and save the actual OpenCV settings.
Subtract the spatial mean of each flow component. Convert the centred OpenCV
flow to canonical millimetres using the verified axis convention.

The generated artefact must reproduce the component standard deviations and
the spurious-EVM scale reported by the legacy corrected-warp measurement
campaign within numerical tolerance. Stop if it does not.

## Surrogate ensemble

Use exactly **256** samples with NumPy seed **20260729**.

For each sample:

1. draw one row shift uniformly over the full recorded crop;
2. draw one column shift uniformly over the full recorded crop;
3. draw one sign from `{-1,+1}`;
4. apply the same periodic shift and sign to both centred flow components;
5. extract the manifest-defined P43 solve support;
6. add the perturbation to the prepared DIC displacement;
7. reconstruct perturbed DIC EVM with `reconstruct_historical_evm`;
8. compare every fixed archived observed-FEM EVM field to that perturbed
   reference on the unchanged core.

The common shift preserves component amplitudes, cross-component structure
and spatial correlation while randomising alignment with the localisation
bands. Periodicity is a stationarity assumption and must be stated as a
limitation.

## Metrics

For each `alpha` and sample, record:

- RMSE;
- relative L2;
- Pearson correlation;
- relative top-10% IoU;
- absolute DIC-q90 IoU;
- absolute DIC-q90 predicted active fraction.

Report the baseline, ensemble mean, standard deviation, median, 2.5% and
97.5% quantiles. Report the fraction of samples for which each candidate has
the best value of each metric.

The quantile interval is named a **surrogate sensitivity interval**, never a
confidence interval.

## PEEQ boundary

PEEQ is a mechanical model output. It is unchanged in this light propagation
and no PEEQ uncertainty is computed. A non-zero PEEQ uncertainty would require
propagating image uncertainty through:

```text
DIC correlation
-> local-map identification
-> boundary history
-> complete mechanical solve
```

That full propagation is explicitly outside this campaign. `peeq_max` must be
reported with status `not_propagated_requires_mechanical_rerun`, not with a
zero uncertainty.

## Interpretation

There is no acceptance threshold in this first propagation baseline.

- Narrow intervals support robustness of the reported metric.
- Overlapping candidate intervals prohibit a unique ranking by that metric.
- A stable ranking across this surrogate ensemble is not material-parameter
  identification.
- The result does not replace the image-level symmetric observation operator.
- No new `alpha`, `Hchi` or `ell` may be selected.

## Outputs

```text
validation/reference_data/dic_uncertainty_propagation_p0043_v1/
  report.json
  samples.csv
  centred_repeat_flow_pixels.npy
validation/figures/dic_uncertainty_propagation_p0043_v1/
  metric_intervals.png
  ranking_probabilities.png
validation/dic_uncertainty_propagation_p0043_results.md
```

No mechanical solve or non-local identification may be launched.
