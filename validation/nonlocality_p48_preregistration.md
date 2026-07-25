# Pre-registration: Helmholtz diagnostic on partition 48

Date: 2026-07-25  
Pre-calculation source revision: `77267015e02b2f3b87545c00c0169a9ee5b2b058`

## Purpose

Partition 48 replaces partition 0 as the length-selection region for the
stage-1 scalar Helmholtz diagnostic. Partition 0 remains preserved as an
exploratory implementation campaign and must not be used to select the length
for subsequent confirmation.

The selection is motivated by indicators extracted independently from the
published figures 6 and 8 before running the in-house solver on this region.
Those indicators are approximate because the figures are saturated at 1%,
compressed, and less resolved than the raw arrays. They are used only to
select a representative window, not as quantitative validation data.

## Frozen region

| Property | Value |
|---|---|
| partition ID | `48` |
| partition index | `(4, 8)` |
| core bounds | `x=[1440,1800)`, `y=[2480,2790)` |
| core shape | `360×310` elements |
| solved bounds | `x=[1290,1950)`, `y=[2330,2940)` |
| solved shape | `660×610` elements |
| padding | 150 pixels on all four sides |
| physical pixel | `1.84 µm` |

The bounds above were verified with the repository's `PartitionLayout`
implementation before calculation.

## Frozen calculation

- constitutive backend: native MFront plane stress, the production default;
- 20 global increments;
- 8 MGIS threads;
- PyPardiso/MKL global linear solver;
- no modification of the mechanical solver, material law, Newton method, or
  plane-stress condensation;
- all converged raw partition fields, diagnostics, logs, hashes, and resource
  measurements must be preserved.

## Frozen Helmholtz sweep

Lengths:

```text
0, 1.84, 3.68, 7.36, 14.72, 29.44, 58.88 µm
```

Equivalent physical-pixel lengths:

```text
0, 1, 2, 4, 8, 16, 32 pixels
```

The filter is applied to the complete padded domain. Metrics are evaluated
only on the retained core. DIC is not filtered. PEEQ remains separate from DIC
equivalent strain and receives no cross-observable amplitude error.

## Frozen interpretation rules

1. P48 is the only selection partition for this stage.
2. P0 is excluded from length selection.
3. Boundary-contaminated candidates are excluded from the main selection.
4. The primary evidence is:
   - Pearson correlation;
   - top-10% localization IoU;
   - IoU and active-area agreement at the absolute DIC 90th-percentile
     threshold.
5. RMSE and relative L2 are supporting amplitude metrics, not the sole
   selection criterion.
6. A useful candidate must improve the primary spatial evidence without a
   physically misleading collapse of the absolute-threshold active area.
7. If the primary metrics favour different candidates, report a range and the
   Pareto trade-off rather than silently choosing the RMSE optimum.
8. After selection on P48, the chosen length or explicitly declared candidate
   range must be applied unchanged to held-out partitions.
9. No selected length is to be interpreted as a material internal length.
10. If raw P48 remains weakly correlated despite the agreement visible in the
    published figures, the report must identify possible disagreement between
    the historical Abaqus reconstruction and the current implementation,
    rather than attributing the result only to ROI selection.

## Allowed stage-1 conclusions

- spatial-width hypothesis supported;
- spatial-width hypothesis partially supported;
- spatial-width hypothesis insufficient.

A “supported” conclusion still requires confirmation on partitions not used
for selection.

