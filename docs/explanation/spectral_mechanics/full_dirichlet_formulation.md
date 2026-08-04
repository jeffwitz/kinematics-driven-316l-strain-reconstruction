# Full-Dirichlet displacement formulation

Let \(\Omega\subset\mathbb R^2\) be a rectangle. At small strain,

\[
\varepsilon(u)=\tfrac12(\nabla u+\nabla u^T),
\qquad
\varepsilon_a=(\varepsilon_{11},\varepsilon_{22},\gamma_{12})^T,
\]

with \(\gamma_{12}=2\varepsilon_{12}\), and
\(\sigma_a=(\sigma_{11},\sigma_{22},\sigma_{12})^T\). The internal power is
\(\sigma_a^T\varepsilon_a\), with no additional factor on the engineering
shear pair.

The prescribed displacement is decomposed as

\[
u=u^\ast+u^f,
\qquad u^\ast|_{\partial\Omega}=u^{\rm DIC},
\qquad u^f|_{\partial\Omega}=0.
\]

The default \(u^\ast\) is a harmonic extension of the measured boundary
values. The transform therefore acts only on the zero-boundary fluctuation.
Equilibrium is \(\nabla\cdot\sigma=0\), or equivalently

\[
\int_\Omega \varepsilon(v):\sigma\,d\Omega=0
\quad\forall v|_{\partial\Omega}=0.
\]

:::{admonition} Repository adaptation
The three-dimensional SRIX response is condensed locally so that
\(\sigma_{33}=\sigma_{13}=\sigma_{23}=0\). The global solver sees only the
three in-plane engineering components and never accesses SRIX variables.
:::
