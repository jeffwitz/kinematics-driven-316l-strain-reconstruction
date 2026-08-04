# EBI-TET and its plane-stress adaptation

Traditional TET2 has two internal states \(z_{e1},z_{e2}\). EBI replaces them
with one element state \(z_e\), driven by the weighted mean strain

\[
\bar\varepsilon_e=\tfrac12(\varepsilon_{e1}+\varepsilon_{e2}).
\]

For the Hookean construction, one constitutive call gives
\(\bar\sigma_e\) and \(C_{\mathrm{alg},e}^{ps}\). Local stresses are rebuilt
with the fixed elastic tangent:

\[
\sigma_{eq}^{EBI}=\bar\sigma_e+C_{e}^{ps}
(\varepsilon_{eq}-\bar\varepsilon_e).
\]

The corresponding tangent action is

\[
\delta\sigma_{eq}^{EBI}=C_{\mathrm{alg},e}^{ps}\delta\bar\varepsilon_e
+C_e^{ps}(\delta\varepsilon_{eq}-\delta\bar\varepsilon_e).
\]

This is an adaptation of the Hookean EBI construction to the repository's
plane-stress SRIX bridge. It is not presented as a direct plasticity result of
the EBI paper.

In the elastic regime, \(C_{\mathrm{alg}}^{ps}=C_e^{ps}\), so reconstruction
is exactly equal to the traditional two-sample response.

:::{admonition} Literature result
EBI assigns internal variables to elements and reduces the constitutive
history count for suitable generalized-standard/Hookean formulations.
:::
