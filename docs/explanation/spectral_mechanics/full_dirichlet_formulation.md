# Full-Dirichlet displacement formulation

Let $\Omega\subset\mathbb R^2$ be a rectangle. At small strain,

```{math}
\varepsilon(u)=\tfrac12(\nabla u+\nabla u^T),
\qquad
\varepsilon_a=(\varepsilon_{11},\varepsilon_{22},\gamma_{12})^T,
```

with $\gamma_{12}=2\varepsilon_{12}$, and
$\sigma_a=(\sigma_{11},\sigma_{22},\sigma_{12})^T$. The internal power is
$\sigma_a^T\varepsilon_a$, with no additional factor on the engineering
shear pair.

The prescribed displacement is decomposed as

```{math}
u=u^\ast+u^f,
\qquad u^\ast|_{\partial\Omega}=u^{\rm DIC},
\qquad u^f|_{\partial\Omega}=0.
```

The default $u^\ast$ is a harmonic extension of the measured boundary
values. The transform therefore acts only on the zero-boundary fluctuation.
Equilibrium is $\nabla\cdot\sigma=0$, or equivalently

```{math}
\int_\Omega \varepsilon(v):\sigma\,d\Omega=0
\quad\forall v|_{\partial\Omega}=0.
```

The constitutive layer supplies the local map

```{math}
\sigma_{n+1}=\mathcal M(\varepsilon_{n+1},z_n,\Delta t).
```

For plane stress, partition the 3D tangent into in-plane
$a=(11,22,12)$ and out-of-plane $b=(33,13,23)$ blocks. The local constraint
gives

```{math}
d\sigma_b=C_{ba}d\varepsilon_a+C_{bb}d\varepsilon_b=0,
\qquad
d\varepsilon_b=-C_{bb}^{-1}C_{ba}d\varepsilon_a.
```

Consequently,

```{math}
C^{ps}=C_{aa}-C_{ab}C_{bb}^{-1}C_{ba}.
```

The elastic condensed tangent, the algorithmic condensed tangent and the
isotropic reference tangent $B_0$ remain three distinct objects. A rejected
global trial is reverted without changing $z_n$; only the independently
verified converged trial is committed.

:::{admonition} Repository adaptation
The three-dimensional SRIX response is condensed locally so that
$\sigma_{33}=\sigma_{13}=\sigma_{23}=0$. The global solver sees only the
three in-plane engineering components and never accesses SRIX variables.
:::
