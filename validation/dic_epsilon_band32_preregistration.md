# DISFlow epsilon sensitivity on the 32 px synthetic band

Date: **2026-07-29**

## Question

Does the Charbonnier epsilon used by OpenCV variational refinement materially
change the narrowing, peak amplification or along-band waviness observed for
the synthetic 32 px EVM band?

This is an output-only measurement-chain diagnostic. It does not change the
mechanical or micromorphic models.

## Fixed settings

- real central \(1024\times1024\) speckle window;
- native `finest_scale=0`;
- DIS `MEDIUM` preset;
- patch size 8 px and stride 3 px;
- 30 DIS gradient-descent iterations;
- variational \(\alpha=100\), \(\delta=1\), \(\gamma=0\);
- 30 variational-refinement iterations;
- horizontal 32 px Gaussian-FWHM strain band;
- integrated displacement jump of 1.5 px;
- pixel size 1.84 µm.

Only epsilon varies:

```text
0.0002, 0.002, 0.02, 0.2, 2.0
```

The production reproduction value is `0.002`.

## Quantities reported

- recovered EVM FWHM;
- recovered-to-imposed EVM peak ratio;
- normal-profile relative L2 error;
- centroid shift;
- coefficient of variation of the recovered peak along the band;
- EVM maps and normal profiles on common scales.

The green rectangle in the profile figure is an FWHM reading aid, not the
injected profile.

## Interpretation rule

No epsilon is selected as “optimal” in this exploratory sweep. A change is
called material only if it is larger than the deterministic differences
visible in the reported metrics and maps. Any apparent improvement in
along-band waviness must be reported together with its effects on width and
peak amplitude.

## Pre-registered refinement after the coarse sweep

The coarse sweep located a sharp transition between `0.002` and `0.02`.
Before executing any additional point, the following three intermediate
values are fixed:

```text
0.004, 0.006, 0.01
```

They refine the transition only. The reporting quantities and interpretation
rule above remain unchanged.
