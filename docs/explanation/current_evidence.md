# Current evidence

**Category: Explanation.** What is established now, what is only supported,
and what remains unresolved?

```{include} ../_generated/current_conclusion.inc
```

## Demonstrated numerically

- The local finite-element solver and its constitutive backends are coherent
  within their declared tolerances.
- The local solution is overly concentrated in regions containing coherent
  deformation bands.
- Coupled micromorphic feedback redistributes PEEQ; it does not merely filter
  the final EVM.
- Excessive coupling can reduce global L2 while suppressing relevant band
  morphology.
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
- Some explored regions show interior compromises between amplitude and
  localization objectives.
- Reproducing measured band width requires an effective spatial scale.

These statements are **supported**, not yet independently confirmed at high
fidelity and transferred.

## What the DIC reproduction chain resolves

The newly recovered raw image sequence makes a direct observation-chain test
possible. With the reported DIS variational parameters applied in a declared
OpenCV 4.14 implementation, the candidate repeated final-state pair produces
a spurious EVM RMS of \(7.90\times10^{-5}\), or **2.62 %** of the final DIC
EVM RMS.

This number is an upper bound, not yet a certified random noise floor. The
residual flow remains coherent over about 120 px, and the acquisition log does
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
one number. Zero-mean displacement sinusoids reach 50 % recovered amplitude
only around 126--127 px, whereas an integrated 16 px strain band is recovered
at 15--17 px. The former isolates one spatial frequency; the latter contains
low-frequency displacement content.

```{figure} ../_static/evidence/dic_measurement_band_fidelity.png
:alt: Recovered versus imposed DIC strain-band width and peak amplitude.
:width: 85%

Bands narrower than 8 px are broadened and attenuated. A 16 px band is
recovered within one pixel, while a 32 px band still shows orientation-
dependent width bias.
```

The measurement chain is therefore demonstrably non-neutral. This evidence
does not invalidate a structural length, but it requires the next FEM/DIC
comparison to pass FEM displacements through the same image-level operator
before any new nonlocal identification.

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

## Current numerical limitation

The homogeneous Newton-25 design currently resolves two interior amplitude
optima while leaving the shortest-length profile censored:

```{include} ../_generated/identifiability_status.inc
```

The short-length, high-coupling corner is censored by mechanical convergence in
the homogeneous F1 design. This is not interpreted as a physical boundary.
Until that part of the parameter domain is either solved robustly or excluded
for a physical reason, the objective surface is incomplete.

The generated {doc}`../reference/claims_matrix` and
{doc}`../reference/evidence_registry` are the source of detailed status and
provenance. Historical campaign narratives are retained outside the public
reading path.

## Conclusion

> Current evidence distinguishes a spatial-length effect from coupling
> strength, but it does not identify a transferable material length.

The final chapter, {doc}`scope_and_prediction`, states what the software can
claim and what extension is required next.
