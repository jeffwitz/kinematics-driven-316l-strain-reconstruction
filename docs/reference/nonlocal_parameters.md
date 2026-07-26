# Nonlocal-parameter reference

**Category: Reference.**

| Symbol | Configuration meaning | Unit |
|---|---|---|
| $p$ | local accumulated equivalent plastic strain | dimensionless |
| $\chi$ | nonlocal equivalent plastic strain | dimensionless |
| $H_\chi$ | coupling modulus | MPa |
| $\ell$ | spatial length | mm internally |
| $\alpha$ | $H_\chi/H_{\mathrm{ref}}$ | dimensionless |
| $A_\chi$ | $H_\chi\ell^2$ | MPa mm² |
| $\omega$ | fixed-point relaxation | dimensionless |

`H_ref` is derived from the local hardening response and read from material or
campaign metadata. It must not be hard-coded in an identification workflow.

## Helmholtz discretization

For element-centred source $p_e$,

$$
(I+\ell^2L_h)\chi=p_e.
$$

$L_h$ is the positive discrete negative Laplacian with omitted missing-neighbour
terms at a Neumann boundary. The DCT eigenvalues are

$$
\lambda^x_k=\frac{2-2\cos(\pi k/n_x)}{h_x^2},\qquad
\lambda^y_l=\frac{2-2\cos(\pi l/n_y)}{h_y^2}.
$$

At $\ell=0$, the implementation returns an exact copy without applying a DCT.

## Campaign interpretation

- `alpha = 0` is the local reference; $\ell$ has no physical effect.
- F0 is a frozen-field proxy and cannot support a coupled-mechanics claim.
- F1 ranks candidates after explicit validation against existing F2 cases.
- F2 provides scientific confirmation.
- “material internal length” requires unchanged-parameter transfer.
