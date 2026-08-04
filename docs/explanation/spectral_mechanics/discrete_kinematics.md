# Discrete kinematics

For $n_x\times n_y$ pixels, displacements live at the $(n_x+1)\times
(n_y+1)$ nodes. Each TET2 pixel is split along one diagonal into two equal
triangles.

For the lower-left triangle,

```{math}
\varepsilon_{11}^{(1)}=(u_x^{br}-u_x^{bl})/h_x,
\quad
\varepsilon_{22}^{(1)}=(u_y^{tl}-u_y^{bl})/h_y,
```
```{math}
\gamma_{12}^{(1)}=(u_x^{tl}-u_x^{bl})/h_y+(u_y^{br}-u_y^{bl})/h_x.
```

For the upper-right triangle,

```{math}
\varepsilon_{11}^{(2)}=(u_x^{tr}-u_x^{tl})/h_x,
\quad
\varepsilon_{22}^{(2)}=(u_y^{tr}-u_y^{br})/h_y,
```
```{math}
\gamma_{12}^{(2)}=(u_x^{tr}-u_x^{br})/h_y+(u_y^{tr}-u_y^{tl})/h_x.
```

Both samples have weight $w_q=1/2$. The discrete residual is

```{math}
R(u)=-\sum_e\sum_{q=1}^2 w_qA_eB_{eq}^T\sigma_{eq}.
```

The defining contract is the adjoint identity

```{math}
\sum_{e,q}w_qA_e\,\varepsilon_{eq}(v)^T\sigma_{eq}
=-\sum_a v_a^TR_a.
```

Writing the two triangle gradients as matrices gives

```{math}
\varepsilon_{eq}=B_{eq}u,
\qquad
R=-\sum_e\sum_qw_qA_eB_{eq}^{T}\sigma_{eq}.
```

The divergence is therefore the negative transpose of the assembled gradient
under the weighted discrete inner product, rather than an independently
chosen stencil.

:::{admonition} Project numerical result
The TET2 stencil passes the adjoint and directional-tangent tests and is the
spatial oracle used to isolate the later EBI state-sharing error.
:::
