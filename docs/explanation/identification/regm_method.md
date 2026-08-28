# REGM as a screening surrogate

**Mode:** explanation  
**Domain:** identification

REGM replays a constitutive candidate around a frozen reference state instead
of running a complete nonlinear forward for every candidate. With the
equilibrium gap

$$f(\theta)=B^T W\sigma(\theta),$$

the reference homogeneous operator gives a displacement correction

$$\delta u(\theta)=-K_0^{-1}f(\theta),$$

which is passed through the same declared observation operator as the measured
field:

$$r(\theta)=W_D O(\delta u(\theta)).$$

This makes REGM useful as a fast ranking or pre-screening surrogate. It is not
automatically an identification replacement for FEMU: the approximation
linearises equilibrium and freezes the causal history used to produce the
stress replay.

The exact-space digital-twin result is positive (Spearman ``0.866``, log
Pearson ``0.878``), but those numbers do not transfer automatically through a
real DIC observation chain.
