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

## Qualification status

The scalar source is qualified through the production nested path on a small
heterogeneous orientation map. With four prescribed increments, the case
converged after three constitutive cutbacks; all accepted subincrements
converged, the non-local coupling had no unrecovered failure, and the maximum
plane-stress residual was below (10^{-6}) MPa. The corresponding automated
coverage is in `tests/integration/test_fcc_crystal_fem.py`.

This is a numerical transposition of the J2 scalar architecture, not a
physical identification of a non-local SRIX law. The monolithic SRIX path is
not qualified yet: it requires constitutive cross-tangent blocks for
(sigma_{,chi}), (Gamma_{,arepsilon}), and (Gamma_{,chi}). Until those
blocks are exposed coherently, SRIX non-local validation must use the nested
reference path.
