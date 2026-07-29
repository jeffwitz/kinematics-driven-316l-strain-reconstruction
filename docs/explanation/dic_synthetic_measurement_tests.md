# Synthetic tests of the DIC measurement chain

**Category: Explanation.** What spatial strain fields are considered
acceptably recovered by the declared DISFlow reproduction?

## What is tested

These tests do not use a synthetic speckle image. A \(1024\times1024\) window
is cut from the real undeformed reference image, then warped by a displacement
field known exactly. DISFlow receives only the original and warped images.
The recovered displacement is differentiated by the same
`reconstruct_historical_evm(...)` operator used for the experimental and FEM
comparisons.

The test therefore includes:

- the real speckle texture;
- image interpolation during the synthetic warp;
- the complete DISFlow estimation;
- differentiation into the historical total equivalent strain, EVM.

It does not include camera noise, illumination drift, out-of-plane motion or
specimen evolution.

## Exact DISFlow settings

The OpenCV 4.14 reproduction uses the `MEDIUM` DIS preset and reaches the
native image grid:

| DIS stage | Value |
|---|---:|
| finest scale | **0** |
| patch size | 8 px |
| patch stride | 3 px |
| gradient-descent iterations | 30 |
| mean normalisation | enabled |
| spatial propagation | enabled |

The variational-refinement stage uses:

| Variational-refinement parameter | Value |
|---|---:|
| iterations | 30 |
| \(\alpha\) | 100 |
| \(\delta\) | 1 |
| \(\gamma\) | 0 |
| Charbonnier \(\epsilon\) | 0.002 |

Both requested values and values read back from the OpenCV object are stored
in the campaign manifest. Stopping at scale 1 is not acceptable here: it
removes the final full-resolution refinement and materially changes the
result.

## Sinusoidal transfer

A sinusoidal displacement of amplitude 0.5 px is imposed along either image
axis at wavelengths from 4 to 128 px. Its recovered amplitude and phase are
fitted away from a 16 px border.

```{figure} ../_static/evidence/dic_measurement_transfer_function.png
:alt: Recovered displacement amplitude divided by imposed amplitude for synthetic sinusoids.
:width: 78%

The recovered displacement reaches 50 % amplitude at a wavelength of about
49 px in both directions. This is a displacement modulation-transfer result,
not a minimum resolvable strain-band width.
```

A sinusoid contains a single spatial frequency. A localised band contains
substantial low-frequency displacement content, so the two experiments must
not be reduced to one interchangeable resolution number.

## Localised EVM bands and the normal section

The imposed strain profile is Gaussian. Its declared size is its full width
at half maximum (FWHM): 4, 8, 16 or 32 px. The corresponding displacement is
the integral of that Gaussian and reaches 1.5 px across the image.

The green rectangular step in the profiles below is **not** the injected
strain. It is an explicit reading aid: its width is the prescribed FWHM and
its height is the peak of the exact imposed EVM. It makes the nominal physical
width visible without hiding the true Gaussian profile.

```{figure} ../_static/evidence/dic_synthetic_band_evm_sections.png
:alt: Exact and DISFlow-recovered synthetic EVM bands with profiles normal to the band.
:width: 100%

Each row uses one common colour scale for imposed and recovered EVM. The cyan
line is the section normal to the band. The right-hand plot superposes the
exact imposed EVM, recovered EVM and the FWHM reference step. The lower axis
is in micrometres and the upper axis in pixels.
```

The measured widths and amplitudes are:

| Imposed FWHM | Physical width | Recovered FWHM | Recovered peak / imposed peak |
|---:|---:|---:|---:|
| 4 px | 7.36 µm | 4 px | 0.823--0.836 |
| 8 px | 14.72 µm | 7 px | 0.997--1.006 |
| 16 px | 29.44 µm | 12--13 px | 1.136--1.152 |
| 32 px | 58.88 µm | 28 px | 1.156--1.251 |

The two values in a range correspond to the two image-axis orientations.
Centroid shifts reach 1, 1--2, 2 and 5 px respectively.

## What “acceptable” means here

There is no binary declaration that every recovered field is “correct”.
Three aspects are reported separately:

1. **width fidelity** — the recovered FWHM relative to the imposed FWHM;
2. **amplitude fidelity** — the recovered peak relative to the imposed peak;
3. **morphology** — lateral waviness, tails and centroid shift visible in the
   EVM maps and normal section.

Under that definition:

- the 4 px band has the correct FWHM, but its peak is attenuated by about
  17 %;
- the 8 px band is close in amplitude and one pixel too narrow;
- the 16 px band is narrowed by 19--25 % and its peak is amplified by about
  14 %;
- the 32 px band is narrowed by 12.5 %, its peak is amplified, and the
  recovered map develops visible along-band texture.

Thus “4 px recovered at 4 px” is only a width statement. It does not establish
perfect amplitude or morphology. Conversely, the 49 px sinusoidal MTF-50 does
not mean that a 32 px localised strain band is invisible.

## Does Charbonnier epsilon remove the waviness?

A targeted sweep varies only the Charbonnier epsilon on the 32 px band. Every
other DIS and variational-refinement setting remains fixed.

```{include} ../_generated/dic_epsilon_band32_metrics.inc
```

```{figure} ../_static/evidence/dic_epsilon_band32_sections.png
:alt: Recovered 32 px synthetic EVM band and normal section for eight Charbonnier epsilon values.
:width: 100%

All recovered maps share one colour scale. Increasing epsilon suppresses the
texture-aligned waviness, but also changes the normal band profile.
```

```{figure} ../_static/evidence/dic_epsilon_band32_metrics.png
:alt: Recovered width, peak gain and along-band waviness versus Charbonnier epsilon.
:width: 88%

Width, amplitude and along-band variation must be read together; no curve is
an optimisation objective by itself.
```

The effect is strong but not free:

- at the production value \(\epsilon=0.002\), the band is 28 px wide, the
  peak is amplified by 16 %, and the along-band coefficient of variation is
  7.0 %;
- at \(\epsilon=0.01\), waviness falls to 3.0 % and the peak becomes nearly
  unbiased, but the band narrows further to 26 px;
- at \(\epsilon=0.02\), waviness falls to 1.1 %, but the band broadens to
  39 px and the peak is attenuated by 27 %;
- larger epsilon values remain visibly over-smoothed.

Therefore epsilon can hide the waviness, but the strong improvement at
\(0.02\) and above is mainly a redistribution of the error into width and
amplitude. This single-band exploratory sweep does not justify changing the
production value.

## Consequence for FEM/DIC comparison

The measurement chain is neither a neutral sampler nor a simple Gaussian
blur. Depending on scale, differentiation can attenuate or amplify peaks and
can narrow a band. Future quantitative comparisons must therefore send FEM
displacements through the same image-level observation operator before
attributing these effects to the constitutive model.

The current numerical evidence and its limitations are summarised in
{doc}`current_evidence`; the operational reproduction command is in
{doc}`../how-to/characterise_dic_measurement_chain`.
