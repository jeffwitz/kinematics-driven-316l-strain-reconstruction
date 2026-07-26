# Tensor conventions

**Category: Reference.**

## Historical two-dimensional vectors

| Quantity | Ordering |
|---|---|
| strain | $[\varepsilon_{11},\varepsilon_{22},\gamma_{12}]$ |
| stress | $[\sigma_{11},\sigma_{22},\sigma_{12}]$ |
| plastic strain | $[\varepsilon^p_{11},\varepsilon^p_{22},\gamma^p_{12}]$ |

$\gamma_{12}=2\varepsilon_{12}$. Historical fields retain this convention.

## Kelvin representation

Plane stress uses

$$
[\varepsilon_{11},\varepsilon_{22},\varepsilon_{33},
\sqrt2\varepsilon_{12}]
$$

and the analogous stress vector. The six-component three-dimensional order is
validated against MGIS metadata rather than assumed. Kelvin scaling preserves
the tensor double product.

Conversions are centralized in `fem_inhouse.core.tensor_reconstruction`.
Factors of two and $\sqrt2$ must not be reproduced in solver code.

## Complete output tensors

The 2D problem remains unchanged. After convergence, outputs may be completed
as symmetric $3\times3$ tensors. Plane stress means
$\sigma_{33}=\sigma_{13}=\sigma_{23}=0$ within numerical tolerance; it does
not mean $\varepsilon_{33}=0$.

For isotropic J2,

$$
\varepsilon^p_{33}=-(\varepsilon^p_{11}+\varepsilon^p_{22}),
\qquad
\varepsilon^e_{33}=-\frac{\nu}{1-\nu}
(\varepsilon^e_{11}+\varepsilon^e_{22}),
$$

and $\varepsilon_{33}=\varepsilon^e_{33}+\varepsilon^p_{33}$.
Native MFront outputs retain native axial values and the numerical
$\sigma_{33}$ residual.

## Equivalent measures

`EVM_HISTORICAL` uses the historical project convention and is reconstructed
from both DIC and FEM displacement through the same observation chain.
`EVM_RECONSTRUCTED_3D` uses the complete total-strain tensor. They are distinct
named observables. PEEQ is accumulated and must not be confused with the
instantaneous norm of the plastic-strain tensor.
