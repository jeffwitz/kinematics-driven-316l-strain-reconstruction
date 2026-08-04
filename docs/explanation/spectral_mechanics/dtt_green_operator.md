# DST-I and the reference Green operator

For zero fluctuation on all four sides, the interior modes are the orthonormal
DST-I basis

```{math}
\phi_{kl}(i,j)=\sqrt{2/n_x}\sqrt{2/n_y}
\sin(\pi ki/n_x)\sin(\pi lj/n_y),
```

with $k=1,\ldots,n_x-1$ and $l=1,\ldots,n_y-1$. The implementation uses
`scipy.fft.dstn`/`idstn`, type 1, with `norm="ortho"`.

For TET2, with $\theta_x=\pi k/n_x$ and $\theta_y=\pi l/n_y$,

```{math}
L_x=4(h_y/h_x)\sin^2(\theta_x/2),\quad
L_y=4(h_x/h_y)\sin^2(\theta_y/2),\quad L=L_x+L_y.
```

The isotropic reference operator $B_0$ is diagonal in this basis:

```{math}
d_x=2\mu_0L+\lambda_0L_x,\qquad
d_y=2\mu_0L+\lambda_0L_y,
```
```{math}
\widehat{G_0p}_x=-\widehat p_x/d_x,
\qquad
\widehat{G_0p}_y=-\widehat p_y/d_y.
```

$B_0$ is a preconditioner, not the SRIX constitutive law. The projected
reference parameters are recorded separately from the elastic and algorithmic
plane-stress tangents.

The assembled reference stiffness is

```{math}
A_0=-\operatorname{div}_D\left(\mathbb R_0:\nabla_D^s\right)
=\sum_{e,q}w_qA_e B_{eq}^{T}C_0B_{eq}.
```

For a DST-I mode, the assembled operator gives the diagonal reference factors

```{math}
\widehat{A_0u}_x=d_x\widehat u_x,
\qquad
\widehat{A_0u}_y=d_y\widehat u_y.
```

The Green action inverts these modal factors with the solver's residual sign.
It is therefore a GMRES preconditioner, not a claim that the heterogeneous
plastic Jacobian is diagonal in the DST basis.

:::{admonition} Literature result
DTT reference operators are the standard mechanism for non-periodic
FFT-based mechanics. The present implementation is a SciPy prototype, not a
native FFTW/2DECOMP implementation.
:::
