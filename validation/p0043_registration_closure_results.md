# E-SRIX-P43-REGISTRATION-CLOSURE-002 — result

## What is closed

- The `StructuredMesh` EBSD element-order defect is reproduced by a 3×5
  sentinel and fixed by explicit F-order assignment.
- The corrected mapping changes the M20 classical prior field materially.
- The HDF5 crop, orientation-map shape, pixel size and source crop are
  archived.
- An independent Schmid-map check was run on the P43 crop.
- A corrected-F RAW M20 forward/Jacobian/optimization diagnostic was completed.

## What remains unknown, exactly

The dataset does not contain the acquisition metadata or a co-registered
reference image needed to prove:

1. EBSD image-axis identity and sign relative to canonical DIC axes;
2. physical origin/registration of the P43 DIC and EBSD crops;
3. the fixed EBSD sample-frame rotation relative to FEM transverse/tensile/normal axes.

The internal Schmid diagnostic selects Euler-derived axis 1 on the M20 crop
(correlations with the stored Schmid field: 0.267, 0.519, 0.006 for axes 0,
1, 2), but this cannot replace acquisition provenance.

## Corrected-F RAW M20 diagnostic

Using the provisional global rank-7 basis and the F-order projected shadows:

| quantity | value |
|---|---:|
| prior raw displacement RMS | 4.06717e-6 mm |
| final raw displacement RMS | 3.57640e-6 mm |
| reduction | 12.07 % |
| Gauss–Newton predicted reduction | 19.76 % |
| optimizer evaluations | 3 |
| optimizer status | positive directional derivative |
| final verification residual | 5.80e-7 |

The run is informative but not a converged identification: several physical
bounds are active and the final equilibrium verification is not at the prior
forward tolerance. The historical C result remains a numerical control only.

## Status

```text
ebsd_element_order_correct = true
ebsd_dic_axis_identity_proven = false
ebsd_dic_axis_direction_proven = false
ebsd_dic_origin_registration_proven = false
ebsd_dic_crop_registration_proven = false
ebsd_fem_sample_frame_proven = false
raw_m20_f_identification_completed = true
raw_m20_f_minimum_converged = false
historical_c_results_physically_valid = false
historical_c_results_retained_as_control = true
```

No constitutive conclusion or M100 authorization follows from this result.
