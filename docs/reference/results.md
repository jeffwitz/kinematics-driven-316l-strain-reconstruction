# Validated results

This page reports preserved evidence. It is not a promise of performance on
another machine.

## Helmholtz spatial-width diagnostic

Campaign:
`validation/reference_data/nonlocality_helmholtz_article_p0000_v1`

The saved article partition-0 solution is filtered on its complete
`510×460`-element solved region, including 150 pixels of padding. All
comparison metrics are evaluated only on the `360×310` retained core. The
sweep uses 0, 1, 2, 4, 8, 16, and 32 physical pixels
(`0–58.88 µm`). Every positive length satisfies the declared
`padding / length >= 4` criterion.

Both DIC and FEM `EVM_HISTORICAL` are reconstructed from nodal displacement
with the same strain, plane-stress invariant, and cell-average operators.
Only FEM is filtered. PEEQ is reported separately as an internal localization
indicator; no amplitude error against DIC EVM is computed.

| Metric | Raw FEM | FEM at `58.88 µm` | Change |
|---|---:|---:|---:|
| RMSE | `2.6859e-3` | `1.3577e-3` | `-49.45%` |
| relative L2 | `0.81743` | `0.41319` | `-49.45%` |
| Pearson correlation | `-0.02920` | `0.09257` | `+0.12177` |
| top-10% IoU | `0.05030` | `0.13116` | `+0.08085` |
| standard-deviation ratio | `1.000` | `0.2316` | — |
| peak ratio | `1.000` | `0.1225` | — |

All three exploratory rankings select `58.88 µm`, the upper sweep boundary;
the optimum is therefore not bracketed. At the absolute DIC 90th-percentile
threshold, `14.72 µm` gives the best IoU (`0.07740`) and predicts an active
fraction of `9.98%` against the DIC reference `10%`. At `58.88 µm`, no FEM
cell remains above that threshold.

The full-domain mean drift is at most `8.674e-19`, the maximum relative
discrete residual is `5.575e-13`, and the smallest padding-to-length ratio is
`4.6875`.

The recorded interpretation is **spatial-width hypothesis partially
supported on this exploratory partition**. Smoothing reduces excessive
concentration and improves the main errors and quantile overlap, but the
remaining correlation is weak and the strongest candidate over-attenuates
peaks. No material internal length is identified. A confirmatory decision
requires an unchanged length on held-out partitions.

See {doc}`../explanation/nonlocality_diagnostic` for the rationale and
{doc}`../how-to/diagnose_nonlocality` for the command.

## Three-backend FEM benchmark

Campaign:
`validation/reference_data/plane_stress_backend_performance_100x100_v1`

All three material paths were run on the same central `100×100`-element crop
of the versioned DIC data, with 20 increments. Each measurement starts in a
fresh process; the order is alternated over three repetitions per backend.
All runs use two MKL threads and, where applicable, two MGIS threads. GNU
`time` records complete-process peak RSS. Complete fields and diagnostics are
preserved for every run.

| Backend | Median process wall | Median FEM solver | Median constitutive | Median peak RSS | Global Newton |
|---|---:|---:|---:|---:|---:|
| Python J2 | `134.36 s` | `133.31 s` | `99.33 s` | `248.96 MiB` | 183 |
| native MFront plane stress | `27.03 s` | `25.89 s` | `9.52 s` | `269.65 MiB` | 93 |
| condensed 3D MFront | `83.43 s` | `82.30 s` | `65.44 s` | `320.30 MiB` | 93 |

All nine runs converge without cutback or local failure. Relative to native
MFront, Python takes `4.97×` more wall time and uses `7.7%` less peak RSS; the
condensed path takes `3.09×` more wall time and uses `18.8%` more peak RSS.
The condensed path is nevertheless `1.61×` faster than Python because both
MFront paths converge in 93 global Newton iterations, versus 183 for Python.

Native and condensed MFront are equivalent to numerical precision: their
maximum differences are `9.171e-15 mm` in displacement, `2.307e-07 MPa` in
stress, `3.197e-12` in total strain, `3.267e-12` in plastic strain, and
`2.427e-12` in PEEQ. The Python backend is an independent implementation and
is equivalent within the declared case-study tolerances rather than bitwise:
its corresponding maximum differences from native MFront are
`2.121e-09 mm`, `6.763e-02 MPa`, `4.471e-07`, `6.037e-07`, and `4.759e-07`.

The maximum Gauss-point transverse residual is `0` for Python,
`9.107e-14 MPa` for native MFront, and `3.745e-08 MPa` for condensed MFront.
The condensed local solve needs at most four iterations, with maximum observed
`cond(Cbb) = 3.984`.

For the current isotropic J2 law, native MFront is therefore the production
choice. Python remains useful as an independent scientific and regression
oracle. Condensed 3D MFront is usable and validated, but its purpose is to
accept a genuinely three-dimensional constitutive law without changing the
two-dimensional FEM solver.

## Native versus condensed 3D J2 on the DIC 10×10 case

Campaign:
`validation/reference_data/mfront_3d_condensed_dic_10x10_v1`

The same J2/Ludwik model was run through native MFront `PlaneStress` and
through MFront `Tridimensional` followed by local three-component
plane-stress condensation.

| Measure | Native plane stress | Condensed 3D |
|---|---:|---:|
| global Newton iterations | 66 | 66 |
| cutbacks | 0 | 0 |
| maximum Gauss-point transverse residual (MPa) | `5.575e-14` | `2.705e-08` |
| maximum local iterations | 0 | 4 |
| mean local iterations | 0 | `2.666` |
| local failures | 0 | 0 |
| maximum `cond(Cbb)` | 0 | `1.896` |

Maximum absolute backend differences are `6.245e-16 mm` for displacement,
`4.804e-08 MPa` for in-plane stress, `5.101e-13` for total 3D strain,
`6.117e-13` for plastic 3D strain, and `4.038e-13` for PEEQ. All declared
field and invariant checks pass.

The condensed run takes `2.805 s`, of which `1.804 s` is constitutive, versus
`1.310 s` and `0.377 s` for the native path on this host. This expected
validation cost is not a reason to replace the faster native backend for the
current isotropic law.

## Complete-tensor reconstruction on the DIC 10×10 case

Campaign:
`validation/reference_data/plane_stress_tensor_reconstruction_dic_10x10_v1`

The same real DIC crop was solved with the unchanged Python and MFront
plane-stress paths. Historical fields were compared with the pre-feature
campaign, while complete tensors were compared between backends.

| Consistency measure | Python | MFront |
|---|---:|---:|
| reconstruction source | analytical | native `AxialStrain` |
| maximum `abs(S33)` (MPa) | `0` | `1.046e-14` |
| maximum `abs(trace(PE))` | `0` | `1.406e-19` |
| maximum `abs(E - EE - PE)` | `8.132e-20` | `1.355e-19` |
| maximum native/analytical total-strain difference | `0` | `1.220e-19` |

Backend comparison:

| Field | maximum absolute difference | relative L∞ |
|---|---:|---:|
| `S_3D` | `3.338e-2 MPa` | `1.437e-4` |
| `E_3D` | `9.581e-8` | `6.654e-5` |
| `EE_3D` | `1.979e-7` | `1.759e-4` |
| `PE_3D` | `2.937e-7` | `3.256e-4` |
| `EVM_RECONSTRUCTED_3D` | `1.339e-7` | `6.685e-5` |
| 3D von Mises stress | `6.930e-2 MPa` | `2.882e-4` |

All declared thresholds pass. Python historical fields are byte-for-byte
unchanged. The largest MFront historical-field difference is
`4.263e-14 MPa` in stress, which is floating-point round-off. The report stores
input and library hashes, all thresholds, diagnostics, complete field NPZs,
and the explicit `EVM_HISTORICAL`/`EVM_RECONSTRUCTED_3D` distinction.

## Article-sized MFront partition

Campaign:
`validation/reference_data/article_100p_pad150_p0000_mfront_v1`

| Property | Value |
|---|---:|
| global ROI | `3600 × 3100` elements |
| layout | `10 × 10`, 100 partitions |
| partition | corner partition 0 |
| retained core | `360 × 310` elements |
| solved region | `510 × 460` elements |
| solved element count | `234600` |
| increments | 20/20 converged |
| cutbacks | 0 |
| Newton iterations | 112 total, 6 maximum |
| final relative residual | `2.207e-8` |
| solver time | `648.402 s` |
| complete process wall time | `650.08 s` |
| peak RSS | `4163308 KiB` |
| swap | 0 |

Boundary DIC displacement is satisfied to `4.163e-17 mm` maximum error. The
reaction-balance ratio is `3.961e-14`.

The maximum PEEQ is `0.06496`. The legacy `0.2` cap was therefore not active
in this partition, although it is absent from the nominal law.

## Comparison with the historical Python table

| Measure | Python table | MFront analytical | Change |
|---|---:|---:|---:|
| process wall time | `1089.80 s` | `650.08 s` | `-40.35%` |
| solver time | `1088.126 s` | `648.402 s` | `1.678×` faster |
| constitutive time | `575.906 s` | `83.409 s` | `6.905×` faster |
| Newton iterations | 113 | 112 | -1 |
| peak RSS | `3768132 KiB` | `4163308 KiB` | `+10.49%` |

The MFront path does not construct the 1000-point Python table. The complete
process still uses more peak memory because MGIS states and tangents, sparse
matrices, and PyPardiso dominate the measurement.

Relative-L2 field differences from the historical run are:

| Field | Relative L2 |
|---|---:|
| `U` | `1.575e-5` |
| `E` | `0.721%` |
| `PE` | `0.910%` |
| `PEEQ` | `0.868%` |
| `S` | `0.759%` |
| `RF` | `0.752%` |

These are model differences between an analytical law and its historical
interpolation, not a bitwise-parity claim.

## DIC/FE comparison on the solved region

| Metric | Full `511 × 461` nodal region | Retained core nodes |
|---|---:|---:|
| equivalent-strain RMSE | `0.254` percentage points | `0.277` percentage points |
| equivalent-strain MAE | `0.186` percentage points | `0.203` percentage points |
| spatial correlation | `0.016` | `-0.028` |
| relative L2 error | `0.792` | `0.841` |

The error magnitude is comparable to the article’s whole-ROI values, but one
corner partition and weak spatial correlation do not establish article
reproduction.

```{image} ../../validation/reference_data/article_100p_pad150_p0000_mfront_v1/preview.png
:alt: DIC equivalent strain, FE equivalent strain, signed difference, and von Mises stress for the saved corner partition.
:width: 100%
```

## Constitutive benchmark

The preserved one-minute benchmark evaluates 200,000 heterogeneous material
points over 20 increments, twice:

| Backend | Median |
|---|---:|
| Python/NumPy | `12.347 s` |
| MFront serial | `13.333 s` |
| MFront, 8 threads | `3.527 s` |

Eight-thread MFront is `3.500×` faster than Python for this constitutive kernel.
This benchmark excludes finite-element assembly and PyPardiso.

## Current evidence boundary

Validated:

- material-point paths;
- consistent-tangent integration;
- complete MFront/Newton coupling;
- a real 10 × 10 DIC crop;
- one article-sized corner partition;
- atomic persistence and resumption.

Not yet validated:

- all 100 article partitions and global stitching;
- padding sensitivity at 50/100/150/200;
- exact article mask and final whole-ROI metrics;
- external Abaqus parity with original `.inp` and ODB data;
- the section thickness needed for reaction parity.
