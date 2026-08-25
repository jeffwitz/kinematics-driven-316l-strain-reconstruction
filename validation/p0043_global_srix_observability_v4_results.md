# P43 global SRIX observability — Sobol campaign v4

## Status

The eight-point campaign is incomplete. Four points (including the prior) were
completed; the fifth point (`sample_index=4`) failed during the base forward
with `3D MFront integration failed with status -1`.

The partial Jacobians are archived in
`validation/reference_data/p0043_global_srix_observability_v4/partial.npz`.
The failure metadata is in `failure.json`. No global Hessian or global SVD is
reported from this incomplete campaign.

## Failed point

The physical parameters reconstructed from the failing Sobol coordinate are:

```text
C11 = 188026.67 MPa    C12 = 113612.57 MPa    C44 = 139053.48 MPa
tau0 = 51.9007 MPa      R = 8.5212 MPa          Q = 17.0635 MPa
b = 4.4210              C = 15990.65 MPa        d = 2620.26
```

All are inside the pre-registered domain. This is therefore a constitutive
forward robustness failure at an admissible point, not a reason to silently
narrow the Sobol domain.

```text
global_sobol_campaign_complete = false
global_svd_qualified = false
experimental_identification_authorized = false
```

The failure must be diagnosed or the point must be explicitly classified as an
infeasible part of the pre-registered domain before a global rank-7 basis can
be promoted.
