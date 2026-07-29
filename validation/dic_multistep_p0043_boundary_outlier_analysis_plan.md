# P43 early measured-boundary outlier audit — analysis plan

## Status

This is an **exploratory numerical diagnosis**, not a preregistered
confirmatory campaign. Preliminary scalar checks were performed before this
file was written to decide which diagnostics could distinguish a measured DIC
outlier from a nonlinear-solver overshoot. No constitutive parameter,
mechanical tolerance, measured field or convergence result is changed.

## Question

The measured-boundary calculation commits states 1--3 and fails while moving
towards state 4. Is that failure explained by a spatial, temporal or
photometric outlier in the measured DIC boundary conditions, especially next
to elements 402245 and 402246 where MFront rejects the Newton trials?

State numbers are ordered direct-reference image indices. They are not
synchronised load fractions.

## Immutable sources

- repaired direct-reference displacement history:
  `reference_data/dic_multistep_history_p0043_repaired_v1/`;
- failed local measured-history calculation:
  `reference_data/dic_multistep_mechanics_p0043_measured_repaired_v1/`;
- raw images `000294.tif` to `000300.tif`;
- pixel spacing: `1.84 µm`;
- P43 solved support and failure locations read from the existing reports.

## Fixed diagnostics

For states 1--6:

1. reconstruct state and incremental historical EVM with the validated common
   operator;
2. fit one affine displacement field to all boundary nodes and separate the
   non-affine residual;
3. measure tangential boundary-displacement gradients and the high-frequency
   residual-energy fraction at wavelengths no larger than 16 pixels;
4. evaluate the exact CPS4 engineering strains at all Gauss points of the two
   rejected elements from the measured displacement increment;
5. evaluate the direct brightness-constancy residual from the reference image
   to each current image, on the full solved support and near the rejected
   elements;
6. compare all measured values to the rejected Newton trial strains already
   recorded in the immutable failure report.

## Interpretation rule

The audit may report that no DIC outlier was observed only if state 4 is not an
isolated maximum across these spatial and photometric diagnostics and if the
measured Gauss strains at the rejected elements remain many orders of
magnitude below the rejected Newton strains.

This cannot prove that the measured pseudo-time history is force-synchronised
or exact. It can only determine whether the recorded failure is explained by
an evident measured spike at the failure location.

## Outputs

```text
validation/reference_data/dic_multistep_p0043_boundary_outlier_v1/
  report.json
  state_metrics.csv
validation/figures/dic_multistep_p0043_boundary_outlier_v1/
  p0043_early_boundary_outlier_diagnostic.png
validation/dic_multistep_p0043_boundary_outlier_results.md
```
