# Operator derivation appendix

This appendix makes the local matrices and the modal construction explicit.
The notation is restricted to the two-dimensional TET2 adaptation used by
this repository.

## Local ordering

Use the nodal order `(bl, br, tl, tr)` and the local displacement vector

```{math}
q_e=(u_x^{bl},u_y^{bl},u_x^{br},u_y^{br},
     u_x^{tl},u_y^{tl},u_x^{tr},u_y^{tr})^T.
```

The two engineering-strain samples are

```{math}
\varepsilon_{e1}=B_1q_e,
\qquad
\varepsilon_{e2}=B_2q_e.
```

For the lower-left triangle,

```{math}
B_1=\begin{bmatrix}
-1/h_x&0&1/h_x&0&0&0&0&0\\
0&-1/h_y&0&0&0&1/h_y&0&0\\
-1/h_y&-1/h_x&0&1/h_x&1/h_y&0&0&0
\end{bmatrix}.
```

For the upper-right triangle,

```{math}
B_2=\begin{bmatrix}
0&0&0&0&-1/h_x&0&1/h_x&0\\
0&0&0&-1/h_y&0&0&0&1/h_y\\
0&0&-1/h_y&0&0&-1/h_x&1/h_y&1/h_x
\end{bmatrix}.
```

These matrices reproduce the finite-difference expressions in the main
text. They also show that an affine displacement has identical strains in
both samples.

## Adjoint residual and two reference operators

With `A_e=h_x h_y` and `w_1=w_2=1/2`, a symmetric isotropic elastic stiffness
would be

```{math}
K_{e,\mathrm{iso}}=\sum_{q=1}^{2}w_qA_eB_q^TC_{\mathrm{iso}}^{ps}B_q.
```

This is not the diagonal Green operator used by the production solver. The
production reference is the Gélébart-type `B_0` operator

```{math}
A_0^{B_0}u=-\operatorname{div}_D\left[
2\mu_0\nabla_Du+\lambda_0(\nabla_Du\odot I)
\right].
```

Its two scalar modal factors are the `d_x` and `d_y` factors given below.
The assembled physical residual is

```{math}
R(u)=-\sum_{e,q}w_qA_eB_{eq}^T\sigma_{eq}.
```

For any zero-boundary virtual displacement `v`,

```{math}
\sum_{e,q}w_qA_e\,(B_{eq}v)^T\sigma_{eq}
=-\sum_a v_a^TR_a.
```

This is the discrete integration-by-parts identity. It is the contract used
by the adjoint tests and fixes the sign of the divergence operator.

## DST-I symbols

Insert the interior mode

```{math}
\phi_{kl}(i,j)=\sin(\theta_x i)\sin(\theta_y j),
\qquad
\theta_x=\pi k/n_x,
\quad
\theta_y=\pi l/n_y.
```

Taking the two triangle differences and forming the weighted adjoint product
gives

```{math}
L_x=4\frac{h_y}{h_x}\sin^2(\theta_x/2),
\qquad
L_y=4\frac{h_x}{h_y}\sin^2(\theta_y/2),
\qquad
L=L_x+L_y.
```

The isotropic reference action is therefore

```{math}
d_x=2\mu_0L+\lambda_0L_x,
\qquad
d_y=2\mu_0L+\lambda_0L_y.
```

For the one-point corner-average stencil, the transverse averaging introduces
the factors `cos^2(theta_y/2)` and `cos^2(theta_x/2)` in `L_x` and `L_y`.
Both factors vanish at the checkerboard corner, which explains the reported
near-null mode without invoking a constitutive or solver defect.

:::{admonition} Repository adaptation
The two-triangle matrices and their DST-I symbols are a two-dimensional
adaptation. They are not claimed to be an identical implementation of the
three-dimensional TETRA2 discretisation in the cited literature.
:::
