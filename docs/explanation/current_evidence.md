# Current evidence

**Category: Explanation.** What is established now, what is only supported,
and what remains unresolved?

```{include} ../_generated/current_conclusion.inc
```

## Demonstrated numerically

- The local finite-element solver and its constitutive backends are coherent
  within their declared tolerances.
- The historical raw-FEM comparison overstates part of the local-model error
  because only the experimental field had passed through DISFlow.
- After symmetric image-level observation, the local case still predicts too
  much area above the absolute DIC q90 threshold.
- Coupled micromorphic feedback redistributes PEEQ; it does not merely filter
  the final EVM.
- Stronger coupling can reduce global L2 and increase correlation while
  degrading absolute-threshold band overlap.
- Equal-$A_\chi$ pairs do not produce identical reduced-fidelity responses.
  Spatial length therefore has an observable effect beyond amplitude alone.

The full-resolution local/coupled evidence quantifies redistribution without
using plot metadata as its primary source:

```{include} ../_generated/micromorphic_redistribution_metrics.inc
```

```{figure} ../_static/evidence/band_roi_evm_comparison.png
:alt: DIC and raw FEM equivalent total strain on a band-containing calibration region for increasing coupling.
:width: 100%

Raw coupled FEM fields compared with DIC on one common scale. No Helmholtz
post-filter is applied to the FEM EVM.
```

## Supported, but not final

- $H_\chi$ and $\ell$ play distinct roles in the present reduced-fidelity
  observations.
- The local formulation retains a model-form localization error after the
  observation asymmetry is removed.
- Reproducing measured band width requires an effective spatial scale.

These statements are **supported**, but the old parameter-objective surface is
not reusable: it scored raw FEM against observed DIC. They require a new
pre-registration using the symmetric operator before high-fidelity
confirmation and transfer.

## What the DIC reproduction chain resolves

The recovered raw image sequence makes a direct observation-chain test
possible. Two profiles are kept distinct:

- `legacy_script_2021` reproduces only the setters present in the supplied
  source: scale 0, patch 4, stride 1 and variational refinement
  \(\alpha=100,\delta=1,\gamma=0,\epsilon=0.002\), 30 iterations. Its
  no-argument OpenCV factory leaves gradient-descent iterations, mean
  normalisation and spatial propagation at factory values, which are queried
  and recorded.
- `declared_medium_v4` explicitly requests the medium preset, scale 0, patch
  8, stride 3, 30 gradient-descent iterations, mean normalisation and spatial
  propagation, with the same variational settings.

Neither profile is chosen by its FEM/DIC score. The legacy-source profile is
primary by provenance; V4 is a sensitivity. Under OpenCV 4.14, the candidate
repeated final-state pair produces a spurious EVM RMS of
\(1.64\times10^{-4}\) with the legacy-source profile and
\(1.36\times10^{-4}\) with V4, respectively **5.42 %** and **4.52 %** of the
final DIC EVM RMS.

This number is an upper bound, not yet a certified random noise floor. The
residual flow remains coherent over about 38 px, and the acquisition log does
not certify that the two last images are the same physical state.

```{include} ../_generated/dic_measurement_chain_metrics.inc
```

```{figure} ../_static/evidence/dic_measurement_null_test.png
:alt: Candidate repeated-state images, recovered flow, spurious EVM and its autocorrelation.
:width: 100%

The EVM amplitude is small relative to the final field, but its long spatial
coherence is incompatible with interpreting the map as white image noise.
```

The synthetic tests also show why measurement resolution cannot be reduced to
one number. With the corrected forward-warp inverse, V4 reaches sinusoidal
MTF-50 around 50 px and the legacy-source profile around 56 px. Their
subpixel FWHM for the same imposed 32 px horizontal band is respectively
25.31 and 18.33 px. The inverse-warp correction itself changes localised-band
metrology substantially even though it moves MTF-50 by less than one pixel.

The imposed and recovered EVM maps, normal sections and explicit FWHM
reference steps are shown and interpreted in
{doc}`dic_synthetic_measurement_tests`.

```{figure} ../_static/evidence/dic_measurement_band_fidelity.png
:alt: Recovered versus imposed DIC strain-band width and peak amplitude.
:width: 85%

At native scale, a 4 px band is resolved at 4 px with about 83 % of its peak.
The 8, 16 and 32 px bands are recovered at 7, 12--13 and 28 px respectively;
the larger bands are narrowed and their differentiated peaks are amplified.
```

```{figure} ../_static/evidence/dic_profile_and_warp_comparison.png
:alt: Synthetic band profiles separating inverse-warp correction and DISFlow-profile effects.
:width: 100%

The first comparison isolates the corrected forward inverse; the second
changes only the DISFlow provenance profile.
```

The measurement chain is demonstrably non-neutral. This evidence does not
invalidate a structural length, but it prohibits fitting one with an
asymmetric raw-FEM versus observed-DIC objective.

## Symmetric observation changes the conclusion

Archived P43 displacement fields for \(\alpha=0,1,2,4\) were replayed through
the image-level chain without rerunning mechanics. DIC, raw FEM and observed
FEM use the same historical EVM operator and the same 360 by 310 core.

```{include} ../_generated/dic_symmetric_observation_metrics.inc
```

```{figure} ../_static/evidence/p43_symmetric_observation_fields.png
:alt: Raw and DISFlow-observed FEM EVM and signed errors for four coupling levels.
:width: 100%

Every row uses common EVM and signed-error limits. The change from raw to
observed FEM is entirely an observation effect.
```

For the local case, relative L2 falls from 0.952 to 0.486 and Pearson
correlation rises from 0.379 to 0.604. The operator therefore explains a
major fraction of the former apparent discrepancy. It does not remove all of
it: the local case still predicts 16.1 % active area above the DIC q90
threshold, whose reference fraction is 10 %.

The candidate ranking is no longer one-dimensional. At \(\alpha=4\), the
observed field has the lowest L2 and highest correlation, and its q90-active
area is close to 10 %. However, absolute-q90 IoU is maximal at
\(\alpha=1\), then decreases to 0.255 at \(\alpha=4\). Lowering global error
by suppressing activity is not equivalent to placing the bands correctly.

```{figure} ../_static/evidence/p43_symmetric_observation_metrics.png
:alt: Raw and image-observed P43 metrics for the legacy-source and declared V4 profiles.
:width: 86%

Both profiles give the same qualitative conflict between amplitude and
localisation. Their spread is observation-chain uncertainty, not a
parameter-selection freedom.
```

PEEQ is unchanged by replay and remains a mechanical model output:

```{figure} ../_static/evidence/p43_peeq_separate.png
:alt: PEEQ model output for the four archived coupling levels.
:width: 100%

Increasing coupling redistributes and attenuates local plasticity. This is not
an experimental PEEQ comparison.
```

## Local image mismatch does not explain the remaining field error

A pre-registered negative control tested whether the direct
brightness-constancy residual,
\(\lvert I_{40}(x+u_{\mathrm{DIC}}(x))-I_0(x)\rvert\), is locally associated
with the remaining observed-FEM/DIC EVM error. The nodal image residual was
averaged to the common element support; no intensity fit, spatial filter or
mechanical rerun was used.

```{include} ../_generated/dic_photometric_quality_metrics.inc
```

```{figure} ../_static/evidence/p43_photometric_quality.png
:alt: P43 photometric residual, local-model EVM error, sensitivity mask, scatter plot and compared EVM fields.
:width: 100%

The high-residual pixels are spatially dispersed and do not reproduce the
structured band-shaped EVM error. The q90 exclusion is a sensitivity only;
the primary metrics remain unmasked.
```

```{figure} ../_static/evidence/p43_photometric_deciles.png
:alt: Mean absolute EVM error versus photometric-residual decile for four coupling levels.
:width: 82%

The decile curves are nearly flat. Pearson residual/error association ranges
from -0.025 to 0.023 and the Spearman result is likewise negligible.
```

Excluding the worst photometric decile retains 90.14% of the core, changes
relative L2 by at most about 1.1%, changes FEM/DIC correlation by less than
0.0015 and does not change candidate ordering. This does **not** support
discarding pixels from the main comparison. It also does not turn the
photometric residual into an uncertainty distribution: propagated
measurement uncertainty remains a separate task.

## An independent structural scale

The grain-mean maximum-Schmid-factor map provides a structural measurement
that never uses FEM/DIC agreement. Under the preregistered exponential-tail
definition, the full-field radial decay is **179.38 µm**. Spatial blocks give
a median of **108.57 µm** with a bootstrap interval of
**[90.92, 122.38] µm**. Directional estimates are **132.93 µm** along x and
**212.31 µm** along y, exposing anisotropy that a scalar radial average hides.

```{figure} ../_static/evidence/ebsd_schmid_correlation.png
:alt: Radial and directional autocorrelation profiles of the grain-mean maximum Schmid factor field.
:width: 85%

The shaded interval and dashed fits were fixed before the field value was
read. The RMS-control definition gives 311.73 µm and is deliberately not
substituted for the primary result.
```

This is an **EBSD/Schmid structural correlation scale**, not a direct
measurement of the micromorphic $\ell$ and not a material internal length.
The input is already averaged per grain, and the native EBSD step and
registration procedure are not archived. Its value is evidence for a
microstructural spatial scale and for directional structure; imposing it in
mechanics requires a separate preregistered hypothesis.

## What the material maps contribute

A homogeneous nominal control and a jointly translated-map control separate
three effects that global agreement otherwise mixes:

```{figure} ../_static/evidence/material_map_controls.png
:alt: DIC EVM and mapped, homogeneous, and translated-map controls with their PEEQ fields
:width: 100%

Full-resolution controls on the band-containing calibration region. EVM uses
one common scale; PEEQ is shown separately as a model output.
```

The homogeneous control gives a lower global L2 error than the mapped model,
but predicts no point above the absolute DIC q90 threshold: it obtains a good
score by suppressing the bands. Conversely, translating both material maps
together while preserving their distributions reduces correlation from
0.379 to 0.140 and top-10% IoU from 0.207 to 0.113. Boundary kinematics explain
much of the smooth background, while the original map placement contributes
genuine localisation information. Neither result establishes transferability.

## Not demonstrated

- one unique value of $H_\chi$;
- one unique value of $\ell$;
- a material internal length for 316L;
- unchanged-parameter transfer to an independent band-containing region;
- complete Abaqus parity;
- prediction before the experiment with the present local descriptor maps.

## Identification consequence

The earlier homogeneous Newton-25 design resolved two interior amplitude
optima while leaving the shortest-length profile censored:

```{include} ../_generated/identifiability_status.inc
```

That numerical censoring remains non-physical. More importantly, V3 now shows
that the entire old objective surface used an asymmetric observation
operator. It is retained as historical evidence that \(H_\chi\) and \(\ell\)
can act differently, not as a valid parameter-identification surface.

No new sweep is authorised until its operator, separate amplitude and
localisation objectives, parameter domain and decision rule are
pre-registered.

The generated {doc}`../reference/claims_matrix` and
{doc}`../reference/evidence_registry` are the source of detailed status and
provenance. Historical campaign narratives are retained outside the public
reading path.

## Conclusion

> Symmetric image-level observation removes a large part of the apparent
> FEM/DIC discrepancy and changes the coupling ranking. Micromorphic
> redistribution remains demonstrated, but neither \(H_\chi\) nor \(\ell\)
> is identified and no material internal length is claimed.

The final chapter, {doc}`scope_and_prediction`, states what the software can
claim and what extension is required next.
