# EBSD/Schmid structural correlation length

Date: 2026-07-27  
Status: completed independent measurement; no mechanical calibration

## Result

The preregistered estimator was applied to the grain-mean maximum-Schmid-factor
field stored in `CP_dataset.h5`. Of 11,160,000 pixels, 66 were excluded by the
frozen physical/Euler mask.

| Quantity | Value |
|---|---:|
| Radial exponential decay length | 179.38 µm |
| Spatial-block median | 108.57 µm |
| Bootstrap 95% interval of block median | [90.92, 122.38] µm |
| x-direction decay length | 132.93 µm |
| y-direction decay length | 212.31 µm |
| Directional anisotropy ratio | 1.60 |
| RMS-control length | 311.73 µm |

The full-field radial exponential fit is numerically regular
($R^2=0.9964$), but it is not spatially representative of every subregion:
the sixteen block estimates range from 68.21 to 180.79 µm. The directional
difference is also substantial. An isotropic scalar length would therefore
discard measurable structure.

The RMS control is 1.74 times the radial exponential estimate. This is not
resolved by choosing the value closest to the previous mechanically fitted
58.88 µm. It confirms that the numerical definition of a structural scale
must remain explicit.

## Interpretation boundary

The input is a per-pixel **grain-mean** Schmid map, not a raw intragranular
orientation field. The measured scale consequently combines grain-domain
extent, spatial orientation correlations and map construction. The HDF5
metadata declares a 1.84 µm grid, but the native EBSD step and registration
procedure are unavailable.

This result is therefore:

- independent of FEM/DIC agreement;
- useful as a structural prior and anisotropy diagnostic;
- not a direct measurement of the micromorphic parameter `ell`;
- not a material internal length;
- not permission to choose among definitions after seeing their values.

No coupled solve was launched. A separate preregistration must decide whether
and how the measured statistic is imposed in mechanics.

## Reproduction

```bash
.venv/bin/fem-inhouse measure-ebsd-structural-length \
  --input /home/jeff/CNRS/Theses/Adil/essais/CP_dataset.h5 \
  --output validation/reference_data/ebsd_structural_length_v1
```

Primary machine-readable evidence:
`validation/reference_data/ebsd_structural_length_v1/report.json`.
