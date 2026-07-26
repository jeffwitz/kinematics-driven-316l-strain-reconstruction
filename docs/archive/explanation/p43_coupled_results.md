# Historical record: detailed coupled campaign

:::{admonition} Superseded
:class: warning
Historical record. Superseded for current scientific interpretation.
:::

P43 is the first region in this project where the coupled micromorphic model
is tested on a DIC field containing several spatially resolved deformation
bands rather than an almost homogeneous field or an isolated hotspot. It is
therefore the most informative coupled calculation completed so far.

This page separates:

- facts measured from the completed calculations;
- interpretations supported by several diagnostics;
- provisional conclusions that still require an independent validation
  campaign.

No Helmholtz filter is applied to the FEM fields shown below. The mechanical
solution itself changes through the energetic term
\(H_\chi(p-\chi)\).

```{figure} ../../validation/figures/p0043-alpha-comparison/p0043_alpha_summary.png
:name: p43-coupled-summary
:alt: Summary of P43 DIC EVM, raw FEM EVM and internal PEEQ for alpha 0, 1, 2 and 4
:align: center
:width: 100%

Overview of the P43 campaign. The first row compares the DIC total equivalent
strain with the raw converged FEM fields. The second row shows the internal
PEEQ redistribution. EVM and PEEQ deliberately use different common colour
scales because they are different physical quantities.
```

## Why P43 is informative

P43 is partition `(4, 3)` of the `10 x 10` decomposition. Its retained core is

```text
global x elements: [1440, 1800)
global y elements: [ 930, 1240)
core shape:         360 x 310 = 111,600 elements
physical x range:   [2.6496, 3.3120] mm
physical y range:   [1.7112, 2.2816] mm
```

The mechanical domain includes 150 padding pixels on each side and contains
`660 x 610 = 402,600` elements. At the fixed candidate length
\(\ell=58.88\,\mu\mathrm m=32\) pixels, the padding-to-length ratio is
`4.6875`.

The DIC field contains a broad diagonal high-strain structure on the right,
secondary oblique structures through the centre, and a heterogeneous
background. This matters for two reasons:

1. peak attenuation alone cannot reproduce all visible structures;
2. a model can improve amplitude errors while still placing or widening the
   bands incorrectly.

P43 is consequently much more discriminating than P154. It is not selected
because it gives the best numerical score, but because its morphology exposes
the distinction between suppressing a peak and reproducing a band.

## Frozen campaign design

The local calculation gives

\[
H_\mathrm{ref}=5168.147582748343\ \mathrm{MPa}.
\]

The length is held fixed and only the coupling ratio varies:

\[
\alpha=\frac{H_\chi}{H_\mathrm{ref}}\in\{0,1,2,4\}.
\]

| Alpha | \(H_\chi\) (MPa) | Interpretation |
|---:|---:|---|
| 0 | 0 | local J2 reference |
| 1 | 5,168.148 | coupling comparable to the median local Ludwik tangent |
| 2 | 10,336.295 | twice that reference tangent |
| 4 | 20,672.590 | strongest pre-registered candidate |

Every run uses 20 increments, the same DIC displacement boundary history, the
same heterogeneous \(\sigma_{y0}\) and \(K\) maps, the native MFront
plane-stress J2 behaviour, and eight MFront threads. Alpha is the only
scientific parameter changed.

The comparison observable is constructed identically on both sides:

```text
nodal displacement
    -> engineering strain from displacement
    -> historical plane-stress equivalent total strain
    -> element-centre average
```

Metrics are evaluated only on the unpadded P43 core.

## Total equivalent strain: spatial reading

```{figure} ../../validation/figures/p0043-alpha-comparison/p0043_total_evm_comparison.png
:name: p43-total-evm-comparison
:alt: P43 DIC and raw FEM total equivalent strain maps for alpha 0, 1, 2 and 4
:align: center
:width: 100%

Total equivalent strain reconstructed from DIC and FEM nodal displacements.
All five maps share the same 99.5th-percentile colour limit. Values above the
limit are saturated identically in every panel.
```

### Local reference, alpha 0

The local model captures part of the oblique network, but concentrates the
response into narrow, high-amplitude paths. The upper-right diagonal is much
sharper and stronger than in DIC. Several grain-scale lines in the centre are
also overemphasised, while broader neighbouring regions remain too weak.

The mismatch is therefore not merely a displacement of one band. It combines:

- excessive peak amplitude;
- insufficient band width;
- excessive fine-scale contrast;
- incomplete agreement on which secondary paths remain active.

This is precisely the failure mode that motivated the nonlocal coupling.

### Alpha 1

The strongest local ridges broaden and their peak amplitude falls. Previously
dark regions adjacent to the main bands acquire intermediate strain. The
large upper-right overprediction is substantially reduced, while the central
oblique network remains visible.

This is already a mechanically useful change: the model has not simply
post-filtered the final map. Plastic evolution has been redistributed and the
equilibrium displacement field has changed.

### Alpha 2

The central and upper-right structures become more continuous and less
fragmented. Fine lines that dominated the local field lose contrast, while
the intermediate-strain background becomes more populated. The top-10%
localisation overlap reaches its maximum at this value.

Alpha 2 therefore retains slightly more of the high-strain spatial ranking
than alpha 4, even though its global amplitude error is larger.

### Alpha 4

The field is the smoothest of the tested solutions. The dominant diagonal
structure is still present, but narrow peaks have largely disappeared.
Correlation, RMSE, relative L2, and absolute-threshold IoU are all best at this
value.

The improvement is not complete. The FEM field still contains oblique
structures that are weaker or less evident in DIC, and the agreement remains
moderate rather than excellent (`r = 0.504`). Alpha 4 should consequently not
be described as a calibrated solution.

## Field errors: what improves and what remains

```{figure} ../../validation/figures/p0043-alpha-comparison/p0043_total_evm_difference.png
:name: p43-total-evm-errors
:alt: Signed raw FEM minus DIC equivalent-strain errors on P43 for alpha 0, 1, 2 and 4
:align: center
:width: 100%

Signed FEM-minus-DIC EVM errors. Red denotes FEM overprediction and blue FEM
underprediction. One symmetric 99.5th-percentile scale is shared by all four
panels.
```

The local error map contains intense red ridges along the narrow FEM bands.
Those ridges progressively disappear as alpha increases. The maximum absolute
error falls from `0.04465` to `0.00819`, an `81.7%` reduction.

The remaining error is more spatially diffuse:

- a positive residual persists along parts of the upper diagonal;
- broad blue regions remain below and to the right of the central structures;
- fine signed textures remain around some material-map boundaries.

The signed mean error becomes slightly more negative:

| Alpha | MAE | RMSE | Maximum absolute error | Signed mean error |
|---:|---:|---:|---:|---:|
| 0 | 0.002616 | 0.003988 | 0.044648 | -0.000150 |
| 1 | 0.001989 | 0.002587 | 0.016678 | -0.000219 |
| 2 | 0.001730 | 0.002203 | 0.012013 | -0.000258 |
| 4 | 0.001434 | 0.001819 | 0.008186 | -0.000305 |

Thus the dominant improvement comes from removing severe local
overpredictions. At the same time, the strongest coupling introduces a small
additional global underprediction. This trade-off would be hidden by RMSE
alone.

## Global and localisation metrics

| Alpha | Pearson \(r\) | Relative L2 | top-10% IoU | DIC-q90 IoU | q90 precision | q90 recall | Predicted q90 area |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.37910 | 0.95156 | 0.20727 | 0.20418 | 0.26481 | 0.47142 | 17.80% |
| 1 | 0.46243 | 0.61739 | 0.24304 | 0.24479 | 0.30198 | **0.56380** | 18.67% |
| 2 | 0.48136 | 0.52557 | **0.24860** | 0.25765 | 0.32218 | 0.56263 | 17.46% |
| 4 | **0.50358** | **0.43413** | 0.24616 | **0.26676** | **0.35240** | 0.52330 | 14.85% |

Several complementary effects are visible:

- correlation and amplitude errors improve monotonically over the tested
  range;
- absolute-threshold precision improves monotonically;
- absolute-threshold recall peaks at alpha 1 and then declines;
- top-10% IoU peaks at alpha 2 and falls slightly at alpha 4;
- the predicted area above the DIC-q90 threshold first grows, then contracts.

This prevents a simplistic conclusion that “more coupling is always better”.
Alpha 4 is best for broad field agreement, but alpha 2 better preserves the
ranking of the most active locations. Alpha 2 and alpha 4 are non-dominated
under the registered metrics.

## PEEQ: evidence of constitutive redistribution

```{figure} ../../validation/figures/p0043-alpha-comparison/p0043_peeq_comparison.png
:name: p43-peeq-comparison
:alt: Internal PEEQ maps on P43 for alpha 0, 1, 2 and 4
:align: center
:width: 100%

Internal PEEQ fields on the P43 core. All maps share the same 99.5th-percentile
colour scale. PEEQ is not an experimental DIC observable.
```

The local PEEQ map is dominated by very thin, high-amplitude paths. Increasing
alpha reduces those peaks and populates their neighbourhood with lower
plastic strain. Large inactive areas remain, so the model does not collapse
to a uniform plastic field.

| Alpha | Maximum PEEQ | Mean PEEQ | Standard deviation | Gradient RMS | Total variation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.06419 | 0.003080 | 0.004676 | 1.2731 | 0.2648 |
| 1 | 0.02194 | 0.002946 | 0.003136 | 0.7224 | 0.1806 |
| 2 | 0.01615 | 0.002876 | 0.002669 | 0.5790 | 0.1498 |
| 4 | 0.01161 | 0.002794 | 0.002164 | 0.4423 | 0.1165 |

Between alpha 0 and alpha 4:

- the maximum falls by `81.9%`;
- the standard deviation falls by `53.7%`;
- gradient RMS falls by `65.3%`;
- total variation falls by `56.0%`;
- the mean falls by only `9.3%`.

The much stronger reduction in peaks and gradients than in the mean is direct
evidence of redistribution rather than a uniform suppression of plasticity.

```{figure} ../../validation/figures/p0043-alpha-comparison/p0043_peeq_distributions.png
:name: p43-peeq-distributions
:alt: Empirical PEEQ distributions and cumulative distributions on P43
:align: center
:width: 100%

Empirical PEEQ density and cumulative distribution on the retained P43 core.
The high-value tail contracts as coupling increases.
```

The distribution provides a second, independent reading:

- the extreme local tail is progressively truncated;
- the density at exactly or almost zero decreases;
- more elements occupy the low-to-intermediate PEEQ range;
- the distribution becomes narrower without becoming a delta function.

This is the signature expected from the energetic term:
high-\(p\) locations with \(p>\chi\) are hardened, while nearby locations with
\(p<\chi\) are slightly softened.

## Numerical integrity and computational cost

| Alpha | Wall time | Peak RSS | Newton iterations | Fixed-point iterations | Cutbacks |
|---:|---:|---:|---:|---:|---:|
| 0 | 15 min 43 s | 4.09 GiB | 129 | 0 | 0 |
| 1 | 26 min 40 s | 4.36 GiB | 172 | 577 | 0 |
| 2 | 29 min 56 s | 4.36 GiB | 202 | 616 | 0 |
| 4 | 36 min 14 s | 4.32 GiB | 249 | 679 | 0 |

The stronger coupling increases the number of global Newton iterations and
therefore total runtime. Peak memory remains approximately constant once the
nonlocal state is enabled.

For every coupled campaign:

- all 20 increments converge;
- no cutback occurs;
- no local coupling failure occurs;
- the maximum Gauss-point plane-stress residual remains below
  `8.74e-14 MPa`;
- the maximum Helmholtz residual remains below `7.50e-13`;
- the maximum relative tangent asymmetry remains below `7.11e-16`.

The observed scientific trend is therefore not associated with a loss of
plane stress, a failed fixed point, or a change of linear-solver contract.

## Temporary conclusions

### Supported by the current evidence

1. **The local J2 solution is too concentrated on P43.** It produces narrow
   peaks and excessive fine-scale contrast relative to DIC.
2. **Energetic micromorphic coupling genuinely redistributes plasticity.**
   Peak, variance, gradient energy, and total variation fall much faster than
   mean PEEQ.
3. **The redistribution improves the raw mechanical solution.** All positive
   candidates pass the eight pre-registered criteria without post-filtering.
4. **P43 is scientifically useful.** Unlike P154 and P15, it exposes both
   amplitude reduction and the preservation or loss of several deformation
   bands.
5. **The implementation remains mechanically and numerically controlled.**
   Plane-stress, Helmholtz, tangent-symmetry, and convergence diagnostics stay
   far inside their limits.

### Plausible but not yet established

1. **The fixed length of 58.88 micrometres is plausible**, because it produces
   a useful redistribution on a band-containing ROI. It is not yet an
   identified material length: it was selected by an earlier post-processing
   diagnostic and was not varied here.
2. **The useful coupling range lies around alpha 2 to 4.** Alpha 2 best
   preserves top-ranked active locations; alpha 4 best reduces overall field
   error. The current data do not select one uniquely.
3. **Very strong coupling may eventually over-regularise the response.** The
   decreasing q90 recall and slightly lower top-10% IoU at alpha 4 are early
   indicators, but they are not sufficient to locate that limit.

### Not supported by this campaign

- a unique value of \(H_\chi\);
- a unique or intrinsic material length \(\ell\);
- transferability to another ROI without rechecking;
- replacement of crystal plasticity by the present isotropic model;
- reproduction of every DIC band: the best correlation is still only about
  `0.50`.

## Consequence for the next campaign

The next step should not be an unregistered increase of alpha on P43. That
would favour scalar error metrics after inspecting the trend and would not
separate coupling strength from length.

A defensible next protocol should:

1. freeze a small candidate set containing alpha 2 and alpha 4;
2. vary the length on a pre-declared grid;
3. select with both amplitude and morphology metrics, including q90 recall and
   top-10% IoU;
4. reserve at least one other band-containing partition for confirmation
   without parameter adjustment;
5. reject candidates that improve L2 mainly by flattening the field while
   degrading band localisation.

Only that held-out, two-parameter experiment can begin to separate the role of
\(H_\chi\), which controls the energetic penalty, from the role of \(\ell\),
which controls the spatial interaction range.

## Reproducibility

The calculations, hashes, colour limits, and plotting choices are recorded in:

- `validation/nonlocal_p0043_validation_results.md`;
- `validation/figures/p0043-alpha-comparison/plot_metadata.json`;
- `validation/nonlocal_p0043_preregistration.md`;
- `results/constitutive-local-p0043-pad150`;
- `results/constitutive-nonlocal-p0043-pad150-a100`;
- `results/constitutive-nonlocal-p0043-pad150-a200`;
- `results/constitutive-nonlocal-p0043-pad150-a400`.

The figure command is:

```bash
.venv/bin/fem-inhouse plot-coupled-alpha-fields \
  --input data/processed/case_study \
  --local-campaign results/constitutive-local-p0043-pad150 \
  --coupled-campaign 1 results/constitutive-nonlocal-p0043-pad150-a100 \
  --coupled-campaign 2 results/constitutive-nonlocal-p0043-pad150-a200 \
  --coupled-campaign 4 results/constitutive-nonlocal-p0043-pad150-a400 \
  --partition-id 43 \
  --output validation/figures/p0043-alpha-comparison \
  --strain-vmax-percentile 99.5 \
  --peeq-vmax-percentile 99.5 \
  --difference-vmax-percentile 99.5
```
