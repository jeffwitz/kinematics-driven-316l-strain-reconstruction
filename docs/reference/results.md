# Validated results

This page reports preserved evidence. It is not a promise of performance on
another machine.

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

