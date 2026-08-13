# SRIX scalar non-local source

The first non-local SRIX transposition uses one scalar field only. Its local
source is

\[
\Gamma = \sum_{s=1}^{12} p_s,
\]

where `p_s` is the positive accumulated slip exposed by SRIX as
`EquivalentPlasticSlip[s]`. This quantity is deliberately named
`accumulated_slip`; it is not J2 PEEQ and must not be reported as equivalent
plastic strain.

The spatial problem is the same scalar Helmholtz problem used by the qualified
J2 infrastructure,

\[
H\chi = \Gamma, \qquad H = I-\ell^2\Delta,
\]

with the existing transaction-safe nested/staggered infrastructure. The
external MFront field keeps the historical bridge entry name
`NonlocalEquivalentPlasticStrain` for interface compatibility, but its meaning
for SRIX is the scalar `\chi` associated with `\Gamma`, not PEEQ.

The SRIX local law is unchanged when the micromorphic coupling modulus is zero.
When enabled, the coupling modulus is an explicit numerical transposition
parameter and is not a newly identified 316L SRIX parameter.

The initial implementation is available through the 3-D condensed
plane-stress bridge. The generic solver remains responsible for Helmholtz,
transactions, Newton, staggered coupling, and globalisation; SRIX supplies the
constitutive response and the scalar source.
