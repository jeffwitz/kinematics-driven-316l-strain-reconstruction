# Historical record: scientific goal

:::{admonition} Superseded
:class: warning
Historical record. Superseded for current scientific interpretation.
:::

## The central question

Digital image correlation measures a displacement field on the specimen
surface. Differentiating it provides a strain field, but two difficulties
remain:

1. differentiation amplifies measurement noise;
2. a measured kinematic field does not automatically satisfy mechanical
   equilibrium for any constitutive model.

The project asks whether a deliberately simple material description can
transform those measured kinematics into a mechanically admissible field while
retaining the dominant microscale localization patterns.

```{image} ../_static/workflow.*
:alt: Workflow from measured DIC displacement to mechanically admissible reconstructed fields.
:width: 100%
```

## What is reconstructed

The solver receives:

- measured nodal displacements $u_x(x,y)$ and $u_y(x,y)$;
- a local initial-yield-stress map $\sigma_y(x,y)$;
- a local hardening-coefficient map $K(x,y)$.

DIC displacement is prescribed **only on the boundary** of each solved
subdomain. The interior displacement, strain, plastic strain, stress, and
reaction fields are computed from:

- kinematic compatibility of the CPS4 interpolation;
- the J2/Ludwik constitutive law;
- global force equilibrium.

This is why the output is a reconstruction rather than a smoothed copy of the
DIC strain.

## Meaning of local material maps

The pixel-wise values $\sigma_y(x,y)$ and $K(x,y)$ are **effective
reconstruction descriptors**. They encode the spatial organization observed
under one loading path, one DIC resolution, one filtering pipeline, and one
constitutive assumption.

They must not be interpreted as:

- intrinsic properties of individual grains;
- loading-path-independent material constants;
- a substitute for crystallographic orientation or slip-system parameters.

This distinction is central. The model is intentionally lighter than crystal
plasticity because its target is mechanically consistent field
reconstruction, not a complete microstructural mechanism.

## Why a simple law is useful here

The Ludwik–Hollomon law has three intuitive quantities:

$$
\sigma_\mathrm{flow}
  = \sigma_y + K\,p^n,
$$

where $p$ is equivalent plastic strain. The article keeps $E$, $\nu$,
and $n$ homogeneous and varies $\sigma_y$ and $K$ pixel by pixel.

This is the minimum constitutive complexity needed to test the coupling:

- $\sigma_y$ controls where plasticity starts;
- $K$ controls local hardening amplitude;
- $n$ controls the common curvature;
- equilibrium redistributes stress and smooths incompatible high-frequency
  kinematics.

Adding crystal plasticity would answer a different question and require
orientations, slip systems, and a substantially larger validation programme.

## Geometry and plane stress

The cropped ROI contains `3600 × 3100` pixels at $1.84\,\mu\mathrm{m}$ per
pixel, corresponding to $6.624\times5.704\ \mathrm{mm}^2$. The article
reports a 2 mm specimen thickness and observes the surface, so the supported
model uses plane stress:

$$
\sigma_{zz}=0.
$$

The out-of-plane strain is not zero. In an elastoplastic FEM state it is the
sum of the elastic plane-stress response and the isochoric J2 plastic response.
It is reconstructed only after convergence, without adding a 3D unknown or
changing the 2D solve. See {doc}`plane_stress_tensors`.

## Shared equivalent-strain measure

Both DIC and FE displacement fields are differentiated with the same operator:

$$
\epsilon_{xx}=\frac{\partial u_x}{\partial x},\qquad
\epsilon_{yy}=\frac{\partial u_y}{\partial y},\qquad
\epsilon_{xy}=\frac{1}{2}
\left(\frac{\partial u_x}{\partial y}
      +
      \frac{\partial u_y}{\partial x}\right).
$$

The comparison scalar is

$$
\epsilon_\mathrm{vM}
=\sqrt{\frac{2}{3}\,\boldsymbol{\epsilon}' :
                         \boldsymbol{\epsilon}'},
$$

The historical article comparison completes both DIC and FE total strains with
the purely elastic closure
\(\epsilon_{zz}=-\nu(\epsilon_{xx}+\epsilon_{yy})/(1-\nu)\). Computing both
sides from nodal displacement avoids comparing one differentiation convention
with an Abaqus-specific extrapolation convention, but this closure does not
identify the transverse strain after plastic flow.

The software therefore keeps this scalar as `EVM_HISTORICAL` and separately
computes `EVM_RECONSTRUCTED_3D` from the accepted complete FEM tensor. A single
final DIC image cannot supply an analogous elastoplastic 3D tensor without its
loading history and local constitutive integration.

## Relationship to the article

The supplied manuscript, *Kinematics-Driven Reconstruction of Microscale
Strain Localization in 316L Stainless Steel*, defines the scientific scope and
reports whole-ROI comparisons. This repository currently reproduces the
DIC-driven computation from versioned arrays and has validated one
article-sized partition.

It does not yet claim complete article reproduction because:

- the 100 partitions have not all been solved and stitched;
- the exact article mask and final metric conventions remain to be applied;
- the original Abaqus input and ODB extraction are unavailable;
- only DIC step 40, not the historical baseline steps, is versioned.

The software therefore separates “the configured problem converged” from “the
scientific article was reproduced”.
