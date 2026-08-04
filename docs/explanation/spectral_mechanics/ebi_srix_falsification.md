# Causal falsification of one-state EBI-TET

The decisive comparison keeps the same TET2 kinematics and solver, changing
only the number of constitutive histories. At 24x24:

| comparison | displacement | stress | accumulated slip | nodal equilibrium |
|---|---:|---:|---:|---:|
| TET2 vs CPS4 | 0.11% | 0.19% | 0.72% | 0.87% |
| EBI vs TET2 | 1.06% | 1.11% | 5.39% | 5.45% |
| EBI vs CPS4 | 1.11% | 1.23% | 5.76% | 6.01% |

```{figure} ../../_static/spectral_mechanics/error_decomposition.png
:alt: Accumulated-slip error decomposition at 24 by 24 pixels.
:name: spectral-error-decomposition

The same-kinematics comparison isolates the state-sharing contribution.
```

```{figure} ../../_static/spectral_mechanics/refinement_accumulated_slip.png
:alt: Accumulated-slip error under spatial refinement.
:name: spectral-refinement-slip

TET2 approaches CPS4, while the EBI/TET2 gap remains several percent.
```

The EBI Newton-GMRES solve itself is accurate: at 12x12 it reached a verified
residual of $1.12\times10^{-12}$, with a maximum Hookean prerequisite error
of about $3.8\times10^{-14}$. The mismatch is therefore not a convergence
failure or a high-frequency artifact.

The demonstrated result is that state sharing dominates the registered error:
the spatial stencil, global solver and convergence tolerance are held fixed,
while only the number of constitutive histories changes. A compatible local
interpretation is non-commutation of nonlinear history evolution:

```{math}
\mathcal U(z,\tfrac12(\varepsilon_1+\varepsilon_2))
\ne \tfrac12[\mathcal U(z,\varepsilon_1)+\mathcal U(z,\varepsilon_2)].
```

The two-state and one-state updates are more precisely

```{math}
z_e^{n+1}=\mathcal U(z_e^n,\bar\varepsilon_e^{n+1}),
\qquad
z_{eq}^{n+1}=\mathcal U(z_{eq}^n,\varepsilon_{eq}^{n+1}).
```

The archive does not contain a system-by-system active-set comparison.
Different active slip sets are therefore an interpretation, not a separately
demonstrated claim. What is demonstrated is that one state cannot recover the
two local histories with the observed accuracy.

:::{admonition} Project numerical result
The status is `experimental_falsified_for_registered_SRIX_case`: the result is
bounded to the registered homogeneous orientation, recorded non-affine load,
and refinements through 24x24. It is not a universal falsification of EBI for
all constitutive laws or load paths.
:::
