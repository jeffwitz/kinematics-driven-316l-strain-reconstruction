# Declared versus legacy DISFlow profile comparison — preregistration

Date frozen: **2026-07-29**

Status: **frozen before the first A5 correlation**

## Question

How much of the existing synthetic measurement-chain response comes from the
approximate inverse warp, and how much comes from the DISFlow settings
explicitly present in the supplied 2021 script?

This campaign characterises the image observation operator. It does not alter
the mechanics and does not select a micromorphic parameter.

## Fixed inputs

- raw image sequence:
  `/home/jeff/CNRS/Theses/Adil/essais/9_numerical/DIC_images`;
- reference: `000294.tif`;
- repeated-final candidate: `000334.tif` to `000335.tif`;
- registered crop: rows `[400,4000)`, columns `[1211,4311)`;
- synthetic transfer window: fixed central `1024x1024` crop;
- pixel size: `1.84 µm`;
- declared mask: generated all-valid boolean mask;
- no post-filtering of EVM.

The unavailable historical `mask.png` is a provenance limitation. The data
owner authorised continuation with the same declared all-valid support for
all three paths. This does not reproduce the unknown historical mask.

## Frozen comparison

| Case | Profile | Warp | Primary purpose |
|---|---|---|---|
| A | `declared_medium_v4` | `legacy_approximate_inverse` | V4 non-regression |
| B | `declared_medium_v4` | `iterative_forward_inverse` | isolate inverse-warp correction |
| C | `legacy_script_2021` | `iterative_forward_inverse` | isolate profile provenance |

The legacy profile is primary for the subsequent V3 replay if its component
convention and requested settings remain consistent with the prepared
displacements. The declared V4 profile remains a sensitivity. No profile may
be selected from its FEM/DIC score.

## Fixed computations

For every case:

1. repeated-final null diagnostic;
2. displacement sinusoids at wavelengths
   `4, 8, 12, 16, 24, 32, 48, 64, 96, 128 px`;
3. Gaussian-gradient bands with FWHM `4, 8, 16, 32 px`;
4. horizontal and vertical orientations.

The epsilon sweep E-DIC-002 is not repeated.

## Reported metrics

- every requested setting and every getter value;
- spurious EVM RMS and autocorrelation scale;
- sinusoidal gain, phase and interpolated MTF-50;
- legacy integer FWHM;
- subpixel FWHM and crossing status;
- peak gain;
- peak shift;
- positive-background centroid shift;
- relative profile L2 error;
- along-band coefficient of variation where available.

Case A passes the non-regression check if its queried settings and legacy
integer widths equal V4, and its floating values differ by no more than
`1e-6` relative except timestamps and newly added metrics.

No numerical threshold is used to declare one profile scientifically better.

## Artefacts

```text
validation/reference_data/dic_legacy_profile_comparison_v1/
validation/figures/dic_legacy_profile_comparison_v1/
validation/dic_legacy_profile_comparison_results.md
```

Each case occupies a separate subdirectory. Existing V1, V2, V4 and
E-DIC-002 artefacts are read-only.
