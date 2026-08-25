# P43 pixel/mm audit

## Conclusion

No pixel/mm conversion error was found in the audited chain.

The registered scale is `0.00184 mm/pixel`. The DIC temporal-noise report stores
`0.0510928919420213 pixel` and `9.40109211733192e-5 mm`; their conversion agrees
to machine precision. The archived identification scalar `9.40e-5 mm` is
therefore consistent with approximately `0.0511 pixel`.

## Checks

| check | result |
|---|---:|
| robust noise conversion error | `0.0 mm` |
| image-flow pixel→mm→pixel round trip | `2.8e-17 px` max |
| FEM strain `B u` mm-vs-pixel representation | `1.7e-15` max |
| archived M20 prior whitened RMS, mm/pixel calculation | `0.04606497498` / `0.04606497498` |

The source report is
`validation/reference_data/dic_boundary_loading_subspace_p0043_v1/report.json`.
The machine-readable audit is
`validation/reference_data/p0043_pixel_mm_audit_v1/report.json`.

## Remaining limitation

This validates units, not the statistical adequacy of the whitening. The
scalar `1/sigma I` used by the experimental M20 script is a registered
per-state scale, not a full covariance model. Existing repeat-frame and
temporal diagnostics indicate substantial affine/coherent and spatially
correlated DIC noise. Thus the M20 NO-GO cannot be attributed to a pixel/mm
mistake, but its scalar RMS should not be interpreted as a fully whitened
likelihood without a covariance-qualified whitener.
