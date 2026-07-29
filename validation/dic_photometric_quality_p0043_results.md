# P43 photometric quality versus FEM/DIC agreement — results

Date: **2026-07-29**

Pre-registration:
`validation/dic_photometric_quality_p0043_preregistration.md`

Primary data:
`validation/reference_data/dic_photometric_quality_p0043_v1/report.json`

## Question

Does a local violation of brightness constancy explain a measurable fraction
of the remaining FEM/DIC EVM discrepancy after symmetric image-level
observation?

## What was computed

The direct DIC displacement maps each reference-image pixel to the final
image. The absolute 8-bit grey-level residual was evaluated by bilinear
sampling and averaged from the four nodal pixels to each EVM element:

\[
\bar r_{I,e}=\operatorname{cell\_average}
\left(
\left|I_{40}(x+u_{\mathrm{DIC}}(x))-I_0(x)\right|
\right).
\]

No contrast correction, intensity fit, spatial filter or mechanical
calculation was applied. The manifest-defined P43 core contains 111,600
elements and all of them have geometrically valid destination coordinates.
The q90 residual is 20.75 grey levels. Excluding values strictly above it
retains 90.14% of the core.

The unavailable historical `mask.png` was not reconstructed. This campaign
uses only the geometric validity of the direct warp; its primary metrics
remain unmasked.

## Result

| Alpha | Pearson residual/error | Spearman residual/error | L2 unmasked | L2 q90 sensitivity | FEM/DIC r unmasked | FEM/DIC r q90 sensitivity |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.0250 | -0.0192 | 0.4858 | 0.4909 | 0.6042 | 0.6046 |
| 1 | -0.0005 | -0.0084 | 0.3542 | 0.3551 | 0.6512 | 0.6522 |
| 2 | 0.0127 | 0.0014 | 0.3197 | 0.3198 | 0.6579 | 0.6591 |
| 4 | 0.0229 | 0.0085 | 0.2917 | 0.2911 | 0.6636 | 0.6650 |

The association between photometric residual and absolute EVM error is
negligible for every candidate. The decile curves are almost flat. Removing
the worst photometric decile changes relative L2 by at most about 1.1% and
Pearson FEM/DIC correlation by at most 0.0015. It does not change the ordering
of the four archived candidates.

```text
local:  residual/error Pearson = -0.025
alpha1: residual/error Pearson = -0.001
alpha2: residual/error Pearson =  0.013
alpha4: residual/error Pearson =  0.023
```

## Interpretation

This is a **negative result** for the tested explanation: the direct local
brightness-constancy residual does not account for the structured remaining
FEM/DIC error on P43. Discarding high-residual pixels is therefore not
supported as a way to improve the primary comparison.

The result does not prove that all measurement effects are negligible.
Brightness residual mixes speckle mismatch, interpolation, lighting changes
and possible out-of-plane effects; it is not an uncertainty distribution.
The synthetic transfer and symmetric observation results remain necessary.
Monte-Carlo propagation of a separately characterised noise model is still a
different, unfinished task.

## Figures

- `validation/figures/dic_photometric_quality_p0043_v1/photometric_quality_and_error.png`
- `validation/figures/dic_photometric_quality_p0043_v1/photometric_deciles.png`

## Scientific decision

- Keep the unmasked V3 metrics as primary.
- Do not introduce a photometric quality mask into parameter identification
  from this result.
- Do not restart micromorphic identification.
- Retain the conclusion that observation asymmetry matters globally, while
  the residual structured spatial mismatch is not explained by this local
  brightness-residual proxy.
