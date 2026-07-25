# Pre-registration: held-out Helmholtz confirmation on partition 42

Date: 2026-07-25  
Pre-calculation source revision: `975def53c94c6e0293cde7c7a92e583edc7cbda2`

## Independence from selection

Partition 42 was proposed as a secondary region before partition 48 was
calculated. It was not used to select the Helmholtz length. The selected value
from P48 is now frozen:

```text
ell = 58.88 µm = 32 physical pixels
```

No intermediate length may be fitted on P42. The confirmation compares only
the raw field (`ell=0`) with the frozen candidate.

## Frozen region

| Property | Value |
|---|---|
| partition ID | `42` |
| partition index | `(4, 2)` |
| core bounds | `x=[1440,1800)`, `y=[620,930)` |
| core shape | `360×310` elements |
| solved bounds | `x=[1290,1950)`, `y=[470,1080)` |
| solved shape | `660×610` elements |
| padding | 150 pixels on all four sides |

These bounds were verified with the repository's `PartitionLayout`
implementation before calculation.

## Frozen mechanical calculation

- native MFront plane-stress backend;
- 20 increments;
- 8 MGIS threads;
- unchanged PyPardiso/MKL FEM solve;
- all mechanical and diagnostic outputs must be retained, including failed
  diagnostics if convergence is not obtained.

## Frozen automatic decision thresholds

The existing confirmatory workflow must pass:

```yaml
decision_thresholds:
  minimum_correlation_gain: 0.05
  minimum_relative_l2_reduction: 0.05
  minimum_iou_gain: 0.02
  maximum_relative_mean_drift: 1.0e-10
```

The IoU gain is evaluated on the pre-registered top-10% localization mask.

## Additional absolute-threshold requirements

At the DIC 90th-percentile numerical threshold:

- IoU gain from raw to filtered FEM must be at least `0.02`;
- the filtered predicted active fraction must remain between `0.05` and
  `0.20`, compared with the DIC reference fraction `0.10`.

These additional requirements prevent an apparent success obtained by erasing
all peaks or by activating most of the field.

## Interpretation

- Passing all automatic and absolute-threshold requirements supports transfer
  of the P48 spatial-width candidate to this held-out partition.
- A mixed result partially supports transfer and must identify which metric
  fails.
- Failure of correlation and localization transfer is evidence against a
  simple universal scalar width correction.
- A discrepancy with the agreement visible in the published figure must also
  trigger investigation of the historical Abaqus/current-solver difference.
- The candidate remains a diagnostic scale, never an identified material
  length.

