# P43 material-map controls preregistration

Date: 2026-07-27

## Question

How much of the spatial agreement with DIC comes from the imposed boundary
kinematics, and how much comes from the spatial correspondence of the
pixel-wise `sigma_y` and `K` maps?

The mapped local P43 campaign remains the reference. No micromorphic coupling
is enabled in either control.

## Frozen controls

### Homogeneous nominal material

```text
sigma_y = 124 MPa
K       = 380 MPa
n       = 0.245
```

These are the nominal macroscopic values already documented by the project.
They are not fitted to P43. Elasticity, displacement boundary conditions,
mesh, increments, backend and solver tolerances remain identical to the
mapped local reference.

### Spatially translated maps

Both maps are translated together using:

```text
shift axis 0 / x = +600 pixels = +1.104 mm
shift axis 1 / y = +500 pixels = +0.920 mm
boundary rule    = periodic toroidal roll
```

This transformation preserves exactly:

- the marginal distributions of `sigma_y` and `K`;
- their pixel-wise pairing;
- every value in both maps.

It destroys their original spatial correspondence with the DIC field. The
periodic seams are an explicit limitation and must be visible in provenance.

## Frozen numerical campaign

| Quantity | Value |
|---|---:|
| partition | P43 |
| partition layout | 10 by 10 |
| padding | 150 pixels |
| increments | 20 |
| maximum Newton iterations | 15 |
| backend | native MFront plane stress |
| MFront threads | 8 |
| nonlocal plasticity | disabled |

## Metrics and interpretation

Primary EVM is reconstructed from raw FEM displacement and compared with DIC
on the retained core using the existing validated operator. No Helmholtz
post-filter is allowed.

Report for mapped, homogeneous and translated cases:

- correlation, relative L2, RMSE and MAE;
- relative and absolute-q90 overlap metrics;
- displacement error;
- PEEQ diagnostics as model outputs only;
- convergence and plane-stress diagnostics;
- generalized section-equilibrium baseline.

No pass/fail threshold is assigned before these first controls. The effect
sizes are interpreted directly:

- homogeneous versus mapped measures the combined information added by map
  heterogeneity and spatial placement;
- translated versus mapped isolates spatial placement while preserving map
  distributions and local pairing;
- translated versus homogeneous indicates whether heterogeneity contributes
  even when it is spatially misregistered.

These controls do not validate the maps as microstructural descriptors. They
only quantify their incremental information under the present reconstruction
protocol.
