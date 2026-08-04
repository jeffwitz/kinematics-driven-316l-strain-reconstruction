# Scientific question

The solver must satisfy four requirements simultaneously:

1. compatible displacement kinematics;
2. discrete mechanical equilibrium;
3. exact full-boundary DIC displacement constraints;
4. transactional constitutive evolution.

The recorded comparison is:

```text
cell-centred one-point   one material state per pixel
TET2                     two triangles and two SRIX states per pixel
EBI-TET                  two triangle samples and one shared SRIX state
```

The target is a matrix-free, stabilization-free equilibrium solve. The
scientific decision is not based on convergence alone: spatial stability,
constitutive fidelity and cost are reported separately.

:::{admonition} Literature result
The non-periodic FFT literature identifies DTT-based displacement solvers and
reports greater robustness for TETRA2 than for one-point HEX1 in difficult
crystal-plasticity calculations.
:::

:::{admonition} Repository adaptation
The present code uses a two-dimensional full-Dirichlet DST-I implementation,
an original TET2 stencil, and a plane-stress adapter around the existing 3D
SRIX bridge. These choices are not claimed to be a line-by-line reproduction
of AMITEX.
:::
