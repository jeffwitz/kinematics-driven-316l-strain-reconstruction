# Legacy DIC integration audit

Date: **2026-07-29**

Status: **A0 passed with an explicit provenance limitation**

Reference commit requested by the work order:
`c8f84ced8756bdc71e6ee378b1f0de3f2105f27e`.

Actual `HEAD` at audit time:
`c8f84ced8756bdc71e6ee378b1f0de3f2105f27e`.

## Scope and decision

This audit checks whether the archived P43 displacements, raw images,
partition metadata, DIC component convention and historical mask are
sufficient to start the legacy-profile comparison and the symmetric
image-level observation operator.

The mechanical displacements and partition metadata are available. The image
hashes agree with the current experimental inventory. The displacement
component convention is determined consistently by the historical script,
the raw-data contract and the prepared-data manifest.

The historical `mask.png` is not available. The data owner explicitly waived
this stop gate on 2026-07-29 because the missing mask does not prevent the
image-level experiment. New campaigns shall consequently use a declared
all-valid binary mask. They shall not describe that choice as reproduction of
the unknown historical support.

## Repository state before modification

The worktree was not clean before this audit. The index contained a staged
reversal of the E-DIC-002 result artefacts and modifications to generated
documentation, while the working tree still contained the corresponding
files. These pre-existing changes were not reset, staged, overwritten or
included in this audit.

Affected paths reported by `git status --short` included:

- `Claude.md`;
- generated evidence-registry fragments;
- `docs/explanation/dic_synthetic_measurement_tests.md`;
- `validation/documentation_evidence_registry.json`;
- the E-DIC-002 result page, figures and reference data.

This audit adds only the present file.

## Archived P43 campaigns

All four requested campaigns contain a finite final nodal displacement field
and a finite PEEQ field.

| alpha | campaign | `U.npy` SHA-256 | `PEEQ.npy` SHA-256 |
|---:|---|---|---|
| 0 | `results/constitutive-local-p0043-pad150` | `236cff92ce5d5e91ff8067760c367b55e9ba21ff6d87fa72d1581899c08d8b50` | `7b6d7fcb8ce4378fa084cead634501941d17a0d12eab2e52910117629a8b0fe8` |
| 1 | `results/constitutive-nonlocal-p0043-pad150-a100` | `ed2f1df285d4bb593b5bee5950d5617f4a3d6dbc1567b4fa169895899ec89e3b` | `6d07877df235fc866a11274b6c50f6e130b66fad2bda164cf56005379b7c39f1` |
| 2 | `results/constitutive-nonlocal-p0043-pad150-a200` | `3fc21936d4b85ca36a1b55901271cc42dca49e07b0ca314f3a8db35baf6e8781` | `5052eeabbae8e2374ffae1df251e867c06282ec708f0727f84d7d76f1bd8768f` |
| 4 | `results/constitutive-nonlocal-p0043-pad150-a400` | `49609c20c19dcdfd9e2274f214acadee371567a18ce3462090fbcfad0b95a15c` | `2ca3e6d0b83fc3cfac6c05da1743373c28bb5d91aa1b600f018f647ec2466d77` |

For every campaign:

- `U.npy` has shape `(661, 611, 2)`, dtype `float64`, unit millimetre,
  and component order `[u1, u2] = [u_x, u_y]`;
- `PEEQ.npy` has shape `(660, 610)`, dtype `float64`;
- the mesh spacing is `0.00184 mm`;
- the partition is P43 with index `(4, 3)`;
- the core bounds are
  `axis0=[1440,1800)`, `axis1=[930,1240)`;
- the solve bounds are
  `axis0=[1290,1950)`, `axis1=[780,1390)`;
- the padding is 150 pixels;
- the core shape is `(360, 310)` elements;
- the solved shape is `(660, 610)` elements.

The non-local campaigns use `ell=0.05888 mm` and coupling moduli
`5168.147582748343`, `10336.295165496686` and
`20672.59033099337 MPa` for alpha 1, 2 and 4 respectively. These values were
read from the campaign manifests; no mechanical calculation was rerun.

## Prepared and historical displacement data

| Field | Shape | dtype | meaning | SHA-256 |
|---|---:|---|---|---|
| `data/raw/case_study/U_40.npy` | `(3600,3100)` | `float32` | historical traction-axis displacement in pixels, canonical `u_y` | `f9a308b43db2adc5068f4728d9553011715fd5854664fd5f66b7c9cd035e831f` |
| `data/raw/case_study/V_40.npy` | `(3600,3100)` | `float32` | historical transverse displacement in pixels, canonical `u_x` | `d7c7725bc7f60f9de97aadc850753343fe6cbe322f5ef02e86c94386b79df2b0` |
| `data/processed/case_study/displacement_x_mm.npy` | `(3601,3101)` | `float64` | nodal `u_x` in mm | `d385a5f36a891fb990eb641d4808b93d1e2307bd68699d45f07a21704286a2b3` |
| `data/processed/case_study/displacement_y_mm.npy` | `(3601,3101)` | `float64` | nodal `u_y` in mm | `177ef78472722afbee27f7f0a89ae8ec0f0c4bfdca9f12460863b463dea633ee` |

The prepared-data manifest states:

```text
axis 0 = x / transverse
axis 1 = y / tensile
V_40 -> u_x
U_40 -> u_y
1 pixel = 0.00184 mm
```

The legacy optical-flow script returns OpenCV flow component zero as `u`
(`X-displacement`) and component one as `v` (`Y-displacement`). Its downstream
script differentiates the historical pair according to the stored array
axes. Together with the raw-data README, the preparation code and its tests,
this determines the canonical mapping above. No transpose, flip or component
swap is needed to explain the prepared arrays.

## Raw images

The raw image directory contains the 42 TIFF images
`000294.tif`--`000335.tif`. The endpoint files checked by this audit agree
with the hashes in the experimental inventory:

| Image | bytes | SHA-256 |
|---|---:|---|
| `000294.tif` | 23,763,409 | `9a4cf912fcc7f989072e1acc075ae40e02f782faf14c4c70535d48daf52282da` |
| `000334.tif` | 23,763,409 | `04f38d166cc186403b85424ccb14f15621c532fef0f1338e6bd6207ba8b8d830` |
| `000335.tif` | 23,763,409 | `2d086df7381856e6d6252721b2a0451ab020bd0d496718b9011d25856917519d` |

The legacy crop `rows[400:4000], columns[1211:4311]` gives the
`(3600,3100)` prepared support.

## Historical scripts and requested DISFlow settings

The source hashes before any archival move are:

| File | bytes | SHA-256 |
|---|---:|---|
| `dic_analysis/dic_displacement_fields.py` | 5,402 | `fdbd9f6c7a75750eedfabc203e98018631ebe7094e14008017b2a6c93a167855` |
| `dic_analysis/yield_stress_hardening.py` | 15,782 | `d6fdb6d460b0bf6f72d1516cc835893ac4f8de1a298c923ca3fc72a25d3a525c` |
| `dic_analysis/README.md` | 3,307 | `0cfa356f7e07cd77ced1b269c94aa504a0b4fb5f042e72769d94edd45b74ba59` |
| `dic_analysis/requirements.txt` | 261 | `e2f7ec3801a47c82fe6327f83cff2a3f42857c3cb8581193823fb3e3c4b6c8b1` |

The optical-flow script explicitly requests:

```text
factory call = cv2.DISOpticalFlow_create()
finest scale = 0
patch size = 4
patch stride = 1
VR alpha = 100
VR delta = 1
VR gamma = 0
VR epsilon = 0.002
VR iterations = 30
```

It does not explicitly set the factory preset, gradient-descent iterations,
mean normalisation or spatial propagation. Their values in OpenCV 4.14 would
be observed defaults, not certified historical values.

## Mask audit and blocking condition

The legacy script loads:

```text
C:\Users\adil.kilinc\Desktop\Thesis\3_data\21_DIC\mask.png
```

and applies:

```python
im_ref * mask
im_def * mask
```

No mask or mask manifest was found in:

- the repository;
- `data/raw/case_study`;
- `data/processed/case_study`;
- `/home/jeff/CNRS/Theses/Adil/essais`;
- `/home/jeff/CNRS/Theses/Adil/essais/9_numerical`;
- the raw `DIC_images` directory.

Consequently, the following required facts cannot be established:

- mask dtype;
- unique mask values;
- whether the historical multiplication was binary or subject to `uint8`
  modular arithmetic;
- the valid comparison support;
- the mask hash;
- whether the prepared final DIC fields were produced with exactly that mask.

A full-one mask is admissible following the explicit waiver, provided that:

- it is generated deterministically at the image support;
- its dtype, values and hash are recorded;
- manifests identify it as `declared_all_valid`, not as a historical mask;
- the profile-comparison report retains the missing historical mask as a
  provenance limitation;
- the same declared support is applied to DIC and every FEM candidate.

## Baseline validation

Before any source modification:

- the complete test suite passed with the real TFEL/MGIS environment and
  MFront behaviour library: **346 passed in 31.19 s**;
- `ruff check .` reported **37 issues in the two legacy scripts only**
  (`dic_displacement_fields.py`: 3,
  `yield_stress_hardening.py`: 34).

The Ruff result is consistent with the planned A1 exclusion of preserved
legacy sources.

## Waiver and required implementation

The data owner authorised continuation without `mask.png` on 2026-07-29.
The maintained implementation must therefore provide an explicit
`declared_all_valid` mask mode. The unavailable historical mask remains a
documented uncertainty rather than being silently reconstructed.

If the historical mask is supplied later, the audit shall be extended to:

1. verify its dimensions and hash;
2. list its dtype and unique values;
3. reproduce the exact NumPy multiplication semantics on `uint8`;
4. determine the maintained binary-mask equivalent without changing the
   historical reproduction path;
5. record the valid core support;
6. compare its support with the declared all-valid sensitivity.

Until then, profile comparisons and symmetric P43 observation results may use
the declared support, but may not claim bitwise reproduction of the historical
DIC support.
