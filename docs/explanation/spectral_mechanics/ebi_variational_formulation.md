# EBI-TET and its plane-stress adaptation

Traditional TET2 has two internal states $z_{e1},z_{e2}$. EBI replaces them
with one element state $z_e$, driven by the weighted mean strain

```{math}
\bar\varepsilon_e=\tfrac12(\varepsilon_{e1}+\varepsilon_{e2}).
```

For the Hookean construction, one constitutive call gives
$\bar\sigma_e$ and $C_{\mathrm{alg},e}^{ps}$. Local stresses are rebuilt
with the fixed elastic tangent:

```{math}
\sigma_{eq}^{EBI}=\bar\sigma_e+C_{e}^{ps}
(\varepsilon_{eq}-\bar\varepsilon_e).
```

The corresponding tangent action is

```{math}
\delta\sigma_{eq}^{EBI}=C_{\mathrm{alg},e}^{ps}\delta\bar\varepsilon_e
+C_e^{ps}(\delta\varepsilon_{eq}-\delta\bar\varepsilon_e).
```

This is an adaptation of the Hookean EBI construction to the repository's
plane-stress SRIX bridge. It is not presented as a direct plasticity result of
the EBI paper.

In the elastic regime, $C_{\mathrm{alg}}^{ps}=C_e^{ps}$, so reconstruction
is exactly equal to the traditional two-sample response.

The shared-state incremental potential is schematically

```{math}
\Pi_e^{EBI}=\sum_{q=1}^{2}w_q\,\psi(\varepsilon_{eq},z_e)
             +\mathcal D(z_e,z_e^n).
```

The single state equation is therefore driven by the weighted element
contribution. The mean-strain call is exact for the Hookean construction
because the elastic part is affine in strain and the weighted fluctuation has
zero mean. It remains an adaptation hypothesis for SRIX.

:::{admonition} Literature result
EBI assigns internal variables to elements and reduces the constitutive
history count for suitable generalized-standard/Hookean formulations.
:::
