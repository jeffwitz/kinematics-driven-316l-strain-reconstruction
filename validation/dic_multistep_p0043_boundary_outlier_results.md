# P43 early measured-boundary outlier audit

## Short answer

State 3 is genuinely early: it is image 3 of 40 and its EVM RMS is only
`4.79 %` of the final EVM RMS. The transition to state 4 nevertheless does
**not** contain a visible DIC spike at the location where Newton fails.

The converged final calculation and the failed measured-history calculation
do not follow the same loading path:

- the historical calculation ramps proportionally from zero to the final
  displacement field;
- the measured replay follows the 40 direct-reference DIC states;
- at state 4, the affine transverse contraction has reached `7.29 %` of its
  final value, while the affine axial extension has reached only `1.09 %`.

Convergence of the first path therefore does not imply convergence of the
second. The evidence points to nonlinear globalisation near the first local
plastic transition, not to an isolated DIC boundary outlier.

## What was checked

The immutable repaired displacement history was inspected for states 1--6.
For each state, the audit:

1. reconstructs EVM with the common validated operator;
2. separates boundary motion into an affine part and a non-affine residual;
3. measures tangential gradients and high-frequency boundary content;
4. evaluates exact CPS4 Gauss strains from the measured displacement
   increment in elements 402245 and 402246;
5. measures the direct brightness-constancy residual against the raw image.

The two elements are adjacent upper-boundary elements:

```text
402245 -> (ix, iy) = (305, 609)
402246 -> (ix, iy) = (306, 609)
```

## State 3 really is early

| Quantity | State 3 | State 4 | Final |
|---|---:|---:|---:|
| image index fraction | 7.5 % | 10.0 % | 100 % |
| EVM RMS fraction of final | 4.79 % | 6.53 % | 100 % |
| EVM maximum | `7.78e-4` | `9.71e-4` | `1.31e-2` |

Image index is not a force-synchronised load fraction, but the reconstructed
EVM independently confirms that state 3 is an early low-strain state.

## No boundary spike explains the failure

For the state-3-to-state-4 increment:

| Diagnostic | Value |
|---|---:|
| boundary displacement RMS | `1.631 µm` |
| non-affine boundary residual RMS | `0.00445 µm` |
| non-affine / total RMS | `0.273 %` |
| maximum tangential boundary gradient | `2.97e-4` |
| maximum measured Gauss strain in rejected elements | `8.89e-5` |

The state-4 boundary increment is more than `99.7 %` affine in RMS. The upper
edge is smooth through the two rejected elements; no pixel-scale spike is
present there.

Across states 1--6, the largest measured Gauss strain in those elements is
`1.59e-4`. The smallest of the recorded rejected Newton strains is `58.0`.
The rejected trial is therefore at least `3.64e5` times larger than the
measured strain at the same locations.

This directly rules out an imposed DIC strain of the recorded magnitude. The
enormous strain is created by the unconverged interior Newton correction next
to the prescribed boundary.

## No photometric anomaly at state 4

| State | Global residual RMS (grey levels) | Failure-neighbourhood RMS |
|---:|---:|---:|
| 1 | 3.31 | 3.23 |
| 2 | 4.22 | 4.42 |
| 3 | 5.11 | 5.83 |
| 4 | 5.06 | 5.81 |
| 5 | 4.72 | 5.61 |
| 6 | 5.24 | 5.94 |

State 4 is neither the largest global residual nor the largest local residual.
This test does not prove perfect optical flow, but it finds no image-quality
event capable of explaining the unique mechanical failure.

## Why the final proportional solve can still converge

```{figure} figures/dic_multistep_p0043_boundary_outlier_v1/p0043_measured_vs_proportional_path.png
:alt: Measured affine strain path compared with the straight proportional path to the final displacement field
:width: 100%

The dashed path is the calculation that reaches the final field. The coloured
points are the measured DIC history. The right panel magnifies states 0--6;
the red interval is the failed transition.
```

The final calculation does not solve each measured state. It follows a
straight displacement path and crosses plastic activation with a different
ratio of transverse and axial strain. The measured path initially contains
mostly transverse contraction and very little axial extension. With
heterogeneous yield maps this changes which points activate and can make the
Newton active-set transition harder, even at a smaller overall strain.

This is algorithmically possible and no longer paradoxical. A robust nonlinear
solver should nevertheless handle it; the failure remains a numerical
limitation.

## Consolidated diagnostic

```{figure} figures/dic_multistep_p0043_boundary_outlier_v1/p0043_early_boundary_outlier_diagnostic.png
:alt: Six-panel audit of early EVM, boundary residuals, local gradients, photometric residuals and the state-4 incremental EVM
:width: 100%

The red crosses locate the rejected elements. Scales are not adjusted per
state to hide peaks.
```

## Conclusion and next action

The DIC-outlier hypothesis is **not supported** for the state-3-to-state-4
failure:

- state 3 is early in both image order and measured strain;
- states 3 and 4 are photometrically ordinary relative to neighbouring
  states;
- the imposed boundary is smooth at the failed elements;
- the measured local strains are over five orders of magnitude below the
  rejected Newton trials;
- replacing state 3 or state 4 by a transparent temporal bridge did not move
  the limiting pseudo-time.

Further deletion or interpolation of measured frames is not justified by
these data. The next diagnostic should instrument the free-DOF Newton
correction immediately before the first rejected constitutive evaluation:
correction norm, elementwise strain increment, tangent conditioning and
residual localisation. Any future globalisation must limit or reject the
interior correction before it creates an inadmissible constitutive trial,
without changing the measured boundary or interpolating internal variables.

## Claim boundary

The image sequence is not force-synchronised. This audit does not establish
that every early measured state is a physically exact loading state. It
establishes only that no local spatial, temporal or photometric DIC outlier was
found that explains the recorded Newton strain explosion.

Machine-readable sources:

- `reference_data/dic_multistep_p0043_boundary_outlier_v1/report.json`;
- `reference_data/dic_multistep_p0043_boundary_outlier_v1/state_metrics.csv`.
