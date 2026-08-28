# Tensor and Kelvin conventions

**Mode:** reference  
**Domain:** numerics

The maintained six-component Kelvin order is
`[xx, yy, zz, xy, xz, yz]`; the in-plane order is `[xx, yy, xy]` and the
relaxed order is `[zz, xz, yz]`.  Kelvin scaling preserves the tensor double
product.  Conversion helpers in `fem_inhouse.core.tensor_reconstruction` are
the single implementation point for factors of two and square root of two.

Legacy two-dimensional fields may use engineering shear
`[eps_xx, eps_yy, gamma_xy]`.  Such fields are explicitly labelled
`EVM_HISTORICAL` and must not be silently mixed with Kelvin quantities.

Structural plane stress sets the three transverse stresses to zero; it does
not set `eps_zz`, `eps_xz` or `eps_yz` to zero.  Completed three-dimensional
outputs retain those relaxed strain components and record the residual.
