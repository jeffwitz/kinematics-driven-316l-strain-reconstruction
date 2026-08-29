# What DIC observes, and what it does not

**Mode:** explanation  
**Domain:** measurement

DIC observes image motion, not mechanics. The chain is

```text
true mechanical displacement
          ↓
image texture is displaced
          ↓
DIC / optical-flow estimator
          ↓
measured displacement
          ↓
differentiation
          ↓
measured strain-like field
```

Thus a mechanical field is not the same object as a measured DIC field.
Differentiation produces a noisy strain estimate and can amplify selected
noise components; it does not turn the DIC field into a measurement of stress,
plastic slip or equilibrium. Those quantities are inferred by the mechanical
model.

## Resolution and amplitude transfer

The synthetic measurement campaigns use a real reference texture, a known
image warp and the same DIC estimator as the experimental pipeline. A sinusoid
of amplitude 0.5 pixel reaches 50% recovered displacement amplitude at a
wavelength of about 49--50 pixels for the corrected V4 chain, and about 56
pixels for the recorded legacy-source profile. This is a modulation-transfer
scale, not a hard resolution cutoff and not a minimum resolvable strain-band
width.

Localised bands demonstrate why the distinction matters. In the corrected V4
campaign, an imposed 4-pixel FWHM band loses about 17% of its peak amplitude,
whereas a 32-pixel band is recovered with a narrower profile and amplified
peak. Width, amplitude and morphology therefore do not transform together: a
band that appears to have the right width can still have a biased amplitude,
and a displacement transfer number cannot be read as a strain-resolution
number. The detailed profiles and Charbonnier sweep are retained in
{doc}`synthetic_measurement_tests`.

## Noise and processing choices

Repeated final-frame differences in the registered P43 chain have standard
deviations of about 0.06283 pixel (column) and 0.04267 pixel (row). They define
a measurement sensitivity scale for the valid mask; they are not confidence
intervals for constitutive parameters. A differentiated EVM map combines the
mechanical signal, the spatial transfer, differentiation and this noise.

The Charbonnier parameter illustrates the trade-off. At the recorded
$\epsilon=0.002$, a 32-pixel band has about 7.0% along-band variation; increasing
to $0.01$ reduces that variation to 3.0% but changes the width and profile.
Smoothing a visible artefact can therefore move the error into amplitude or
width. The observation profile and its version must be part of the provenance,
not an unrecorded post-processing choice.

For a fair comparison the model follows the opposite path:

```text
mechanical displacement -> observation operator O -> observable prediction
```

The same crop, mask, interpolation, component convention and units must be
used on both sides. Comparing a raw finite-element field with a field already
filtered and resampled by DISFlow can introduce wavelength-dependent amplitude
and localisation bias. This is why the operator is part of the calculation
manifest rather than an implicit plotting step.

For identification the conceptual chain is therefore

```text
theta -> mechanical forward -> u(theta) -> O_DIC -> predicted observable
                                                       ↓
                                                  compare with DIC
```

and not a direct comparison of raw mechanical strain with differentiated DIC
strain. When the observation and weighting are fixed, the corresponding
parametric sensitivity is

$$
S_{\mathrm{DIC}}=W_D O_{\mathrm{DIC}}\,\frac{\partial u}{\partial\theta}.
$$

Repeated-frame differences quantify measurement sensitivity on the valid mask;
they are not confidence intervals for the constitutive parameters.  The
loading path also matters: a displacement observed at one frame cannot stand
in for the unrecorded intermediate history used by a path-dependent material.
No post-filtering of EVM or plastic fields is allowed unless it is declared as
a new observation operator and qualified separately.

The stable contracts are in {doc}`../../reference/scientific/observation_operator`
and {doc}`../../reference/data/dic_axis_conventions`.  Their limits and
available evidence are indexed from the {doc}`../../evidence/index` portal.
The repeated-frame uncertainty procedure and its limitations are recorded in
`validation/dic_uncertainty_propagation_p0043_results.md`; the transfer and
band measurements are summarised in {doc}`synthetic_measurement_tests`.
