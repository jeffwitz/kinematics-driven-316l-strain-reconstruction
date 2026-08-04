# Causal falsification of one-state EBI-TET

The decisive comparison keeps the same TET2 kinematics and solver, changing
only the number of constitutive histories. At 24x24:

| comparison | displacement | stress | accumulated slip | nodal equilibrium |
|---|---:|---:|---:|---:|
| TET2 vs CPS4 | 0.11% | 0.19% | 0.72% | 0.87% |
| EBI vs TET2 | 1.06% | 1.11% | 5.39% | 5.45% |
| EBI vs CPS4 | 1.11% | 1.23% | 5.76% | 6.01% |

The EBI Newton-GMRES solve itself is accurate: at 12x12 it reached a verified
residual of \(1.12\times10^{-12}\), with a maximum Hookean prerequisite error
of about \(3.8\times10^{-14}\). The mismatch is therefore not a convergence
failure or a high-frequency artifact.

The causal mechanism is non-commutation of nonlinear history evolution:

\[
\mathcal U(z,\tfrac12(\varepsilon_1+\varepsilon_2))
\ne \tfrac12[\mathcal U(z,\varepsilon_1)+\mathcal U(z,\varepsilon_2)].
\]

Two local histories can also activate different slip systems. A single state
driven by the mean strain cannot retain that intra-pixel history split.

:::{admonition} Project numerical result
The status is `experimental_falsified_for_registered_SRIX_case`: the result is
bounded to the registered homogeneous orientation, recorded non-affine load,
and refinements through 24x24. It is not a universal falsification of EBI for
all constitutive laws or load paths.
:::
