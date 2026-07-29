# Symmetric image-level observation on P43 — preregistration

Date frozen: **2026-07-29**

Status: **frozen before the first archived FEM replay**

## Scientific question

How do the amplitude, active area and localization ranking of the archived P43
fields change when FEM displacement is passed through the same declared
image/DISFlow observation family as the experimental displacement?

No mechanical calculation and no parameter identification is authorised.

## Immutable FEM sources

| alpha | campaign |
|---:|---|
| 0 | `results/constitutive-local-p0043-pad150` |
| 1 | `results/constitutive-nonlocal-p0043-pad150-a100` |
| 2 | `results/constitutive-nonlocal-p0043-pad150-a200` |
| 4 | `results/constitutive-nonlocal-p0043-pad150-a400` |

Every `U.npy` must match the SHA-256 in the partition `status.json`.

## Observation protocol

- partition: P43;
- solve bounds and core bounds: campaign manifest only;
- pixel size: `0.00184 mm`;
- reference image: `000294.tif`;
- reference image crop: solve nodal support, including padding;
- FEM interpolation to image: coincident nodal samples, no smoothing;
- displacement conversion:
  canonical `ux -> image drow`, canonical `uy -> image dcolumn`;
- warp: `iterative_forward_inverse`, tolerance `1e-5 px`;
- mask: deterministic `declared_all_valid`;
- primary DISFlow profile: `legacy_script_2021`;
- sensitivity: `declared_medium_v4`;
- EVM: shared `reconstruct_historical_evm`;
- physical metrics: core only;
- no EVM post-filtering.

The unavailable historical mask and OpenCV binary remain provenance
limitations. The primary profile is selected from the supplied source, not
from agreement with DIC.

## Metrics

For raw and observed FEM against the identical DIC core:

- RMSE and MAE;
- relative L2 error;
- Pearson correlation;
- mean, standard deviation and quantiles 50, 90, 95, 99;
- relative top-10 % IoU, Dice, precision and recall;
- absolute DIC-q90 IoU and predicted active fraction;
- gradient RMS and total variation.

Band profiles are descriptive and use fixed central sections in V1. They are
not used to select parameters.

## Decision rule

The replay is descriptive. It must report:

1. change in amplitude error;
2. change in spatial overlap and active area;
3. whether alpha ranking changes;
4. whether mechanical PEEQ redistribution remains a separate fact;
5. whether renewed non-local identification is authorised.

Identification remains frozen if the observation changes the ranking,
materially changes q90 active area, or leaves the optimum on an explored
boundary. No single L2 improvement is sufficient.

## Outputs

```text
validation/reference_data/dic_symmetric_observation_p0043_v1/
validation/figures/dic_symmetric_observation_p0043_v1/
validation/dic_symmetric_observation_p0043_results.md
```
