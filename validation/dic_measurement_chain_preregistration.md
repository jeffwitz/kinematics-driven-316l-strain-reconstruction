# DIC measurement-chain characterisation — preregistration

Date: **2026-07-29**

Status: **frozen before optical-flow computation**

## Question

How much noise, spatial attenuation and band-width bias does the available
DISFlow reproduction chain introduce before a FEM field is compared with the
DIC equivalent-strain field?

This campaign characterises an observation process. It does not modify the
mechanical model and it does not identify \(H_\chi\) or \(\ell\).

## Source data and frame mapping

The raw source is the external 42-frame TIFF sequence
`000294.tif`--`000335.tif`. The analysed crop is fixed to
`rows[400:4000], columns[1211:4311]`, matching the
\(3600\times3100\) prepared DIC support.

The provisional mapping is:

- reference: `000294.tif`;
- final loaded state: `000334.tif`;
- repeated final-state candidate: `000335.tif`.

This mapping is supported by the sequence length and image registration but
is not backed by an acquisition log. All reports shall retain this limitation.

## Algorithm identity

The historical executable and OpenCV version are unavailable. The campaign
therefore uses a separately identified **DISFlow reproduction implementation**.
Its report must record:

- Python, NumPy and OpenCV versions;
- DIS preset;
- finest scale, patch size and patch stride;
- mean normalisation and spatial propagation flags;
- variational-refinement \(\alpha=100\), \(\delta=1\), \(\gamma=0\);
- 30 variational-refinement iterations;
- every other queryable DIS parameter.

The manuscript value Charbonnier \(\epsilon=0.002\) shall be recorded as
reported but **not applied** if the selected public API does not expose it.
In that case, the result may not be called a bitwise reproduction of the
historical production chain.

## V2.1 — Null test

Compute the flow from `000334.tif` to `000335.tif` on the fixed crop. Report:

1. mean, standard deviation and RMS of both displacement components in pixels;
2. the same standard deviations after removal of the spatial mean translation;
3. RMS and quantiles of the spurious historical plane-stress EVM;
4. the radial autocorrelation profile of the centred EVM and its first
   \(1/e\) crossing in pixels and micrometres;
5. the ratio between spurious-EVM RMS and the step-40 DIC-EVM RMS reconstructed
   by the same operator;
6. local photometric residual statistics after warping, as a descriptive
   quality indicator.

The EVM is reconstructed with the public project operator and
\(\nu=0.3\). No denoising, Helmholtz filtering or mask selected from the
result is permitted.

This is a characterisation, not a pass/fail test. Before observing the result,
the reporting bands are fixed as:

- ratio below 0.1: noise small relative to the compared field;
- ratio from 0.1 to 0.3: non-negligible;
- ratio above 0.3: measurement noise materially limits amplitude claims.

These labels do not replace the numerical value.

## V2.2 — Synthetic transfer and imposed bands

Warp `000294.tif` using smooth, known displacement fields and recover them
with the identical implementation and settings.

### Sinusoidal sweep

Use wavelengths \(4,8,12,16,24,32,48,64,96,128\) pixels. The imposed
displacement amplitude is 0.5 pixel. Analyse horizontal and vertical modes
separately. Recover amplitude by least-squares projection onto the imposed
sine/cosine basis after discarding a border equal to twice the queried patch
size.

Report gain, phase error and an MTF-50 wavelength obtained by monotone
interpolation only if the measured curve brackets 0.5. Otherwise report the
bound explicitly.

### Band-width fidelity

Use Gaussian displacement-gradient bands with full width at half maximum
\(4,8,16,32\) pixels, in horizontal and vertical orientations. Fix the peak
displacement gradient so that the maximum displacement remains below
2 pixels. Estimate the recovered width from the half-maximum crossings of the
median profile over the central half of the orthogonal direction.

Report recovered width, relative width error, peak-gain error and centroid
shift. No post-filtering is allowed.

### Interpretation

The synthetic result measures the algorithmic transfer on the supplied
speckle image. It does not include experimental out-of-plane motion or
load-dependent illumination changes. A nominal micromorphic length may only
be said to exceed the measured algorithmic resolution if its value is larger
than the upper uncertainty bound of the corresponding resolution statistic.

## Artefacts

The campaign shall create:

- `validation/reference_data/dic_measurement_chain_v1/manifest.json`;
- `validation/reference_data/dic_measurement_chain_v1/null_test_report.json`;
- `validation/reference_data/dic_measurement_chain_v1/transfer_report.json`;
- CSV profiles for autocorrelation, MTF and band widths;
- `validation/figures/dic_measurement_chain_v1/null_test.png`;
- `validation/figures/dic_measurement_chain_v1/transfer_function.png`;
- `validation/figures/dic_measurement_chain_v1/band_width_fidelity.png`;
- a concise results page under `validation/`;
- a machine-checked evidence-registry entry.

Large full-resolution flow fields remain generated artefacts and are not
versioned unless their storage is explicitly justified. Reports must contain
source hashes and enough configuration to regenerate them.

## Frozen exclusions

- no micromorphic parameter sweep;
- no change to the local or coupled constitutive laws;
- no threshold fitted after observing a curve;
- no claim that `000334`/`000335` is a certified static pair without the
  acquisition log;
- no claim that the reproduction implementation is the exact historical
  DISFlow binary.
