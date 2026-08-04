# Full-Dirichlet displacement-based spectral mechanics

## Scope and thesis

This section develops a displacement-based, full-Dirichlet spectral solver for
small-strain crystal plasticity with a three-dimensional SRIX material reduced
to plane stress. Its central question is whether one SRIX history per pixel can
be retained without sacrificing mechanical fidelity.

The evidence separates three effects: the one-point stencil has a high-
frequency near-null mode; TET2 removes that spatial defect but uses two local
histories; EBI-TET keeps one history and converges accurately, yet changes the
plastic evolution.

The documented scope is the registered homogeneous SRIX case, recorded
non-affine loading, and refinements through 24x24. It is not a universal claim
about every SRIX loading path.

## Primary sources

* [Amouzou-Adoun et al., *A robust and versatile parallel FFT-based
  mechanical solver for general non-periodic and periodic boundary
  conditions*](https://arxiv.org/abs/2607.05929).
* [Gélébart, *A modified FFT-based solver for the mechanical simulation of
  heterogeneous materials with Dirichlet boundary conditions*](https://doi.org/10.5802/crmeca.54).
* Gehrig and Schneider, *Element-Based Internal Variable Formulations for
  Finite Element Discretizations in FFT-Based Homogenization Methods*,
  [doi:10.1002/nme.70170](https://onlinelibrary.wiley.com/doi/10.1002/nme.70170).
* Frigo and Johnson, *The Design and Implementation of FFTW3*.

```{graphviz}
digraph spectral_pipeline {
  rankdir=LR;
  node [shape=box, style=rounded];
  dic [label="DIC boundary\\nu* extension"];
  fluct [label="zero-boundary\\nfluctuation u^f"];
  dtt [label="DST-I / B0\\npreconditioner"];
  material [label="plane-stress\\nSRIX material"];
  residual [label="adjoint residual\\nR(u)"];
  dic -> fluct -> dtt -> residual;
  fluct -> material -> residual;
}
```

```{toctree}
:maxdepth: 1

scientific_question
full_dirichlet_formulation
discrete_kinematics
dtt_green_operator
operator_derivation_appendix
one_point_instability
tet2_newton_gmres
ebi_variational_formulation
ebi_srix_falsification
constrained_dic_extension
conclusions
```
