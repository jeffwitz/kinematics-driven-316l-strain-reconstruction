# First rung of the adaptive basis: results, and a broken instrument

P43, 100x100, states 21 to 40 against state 20, held-out square 30x30 (9 % of
material points). The metric is the share of the elastic defect that survives,
`|eps_measured - eps_sim| / |eps_measured - eps_elastic|`, so 1.0 means the
simulation is no better than pure elasticity and 0 means the defect is gone.
Coefficients are fitted outside the held-out square; every number below is read
at state 40.

Startup checks: adjoint dot product `1.5e-15`, elastic lifting residual
`6.2e-12` relative.

## The numbers

| basis | r | fitted | held out | negative dissipation | cond `A Phi` |
|---|---|---|---|---|---|
| free field | full | 0.0000 | **0.6028** | -- | -- |
| krylov | 2 | 0.7612 | 0.8725 | 47.7 % | 1.00 |
| krylov | 8 | 0.5844 | 0.8565 | 47.8 % | 1.03 |
| krylov | 32 | 0.1142 | 0.7181 | 48.0 % | 1.05 |
| aligned | 2 | 0.9310 | 1.0351 | 0.00 % | 1.7 |
| aligned | 8 | 0.8814 | 1.0406 | 0.00 % | 45 |
| aligned | 32 | 0.7893 | 0.9364 | 0.01 % | 3.4e4 |
| aligned, free sign | 8 | 0.8208 | 0.9379 | 29.0 % | 42 |
| aligned, free sign | 32 | 0.7587 | 0.8716 | 28.2 % | 3.3e4 |

## What the free field says, and it is the most important line

The unconstrained inverse fits the training region **exactly** -- 0.0000, which
is the surjectivity of `A` doing what it always does -- and still leaves
**0.6028** inside the held-out square. That is a ceiling on every method in the
table, reached with an unlimited number of parameters and perfect knowledge of
the surrounding data.

So the preregistered criterion, `E_DIC(16) <= 0.10` on held-out data, was
**unreachable by construction**. It is not that the bases failed to reach it;
nothing can. Equilibrium, the boundary data and the DIC everywhere outside a
30-pixel square do not determine the plastic field inside that square. The
threshold is not being moved to accommodate a disappointing result -- the
instrument is shown incapable of ever reaching it, by a control that involves
none of the methods under test. Designing that instrument was my error, and the
free-field bound is what exposed it.

**The registered criterion is therefore recorded as unreachable and the
campaign as inconclusive on its own terms.** The measurements below stand on
their own and are reported as such, not as partial credit against a threshold.

## What the measurements do establish

**The fixed basis fits where it is fitted and does not transfer.** Krylov goes
from 0.76 to 0.11 on the fitting region as the rank grows to 32, while the
held-out figure moves only 0.87 to 0.72. Rank buys compression, not prediction.
This is the refuted premise measured honestly for the first time.

**An earlier version of this table was wrong, and the leak was mine.** The
Krylov seeds were built from the full-field residual, held-out square included,
which handed the basis the answer: rank 32 then read 0.0081 held out instead of
0.7181. That is the exact failure the preregistration names -- a near-zero
error being a signature of leakage rather than a success -- and it appeared in
the baseline rather than in a network. Seeds are now masked exactly as the
coefficient fit is.

**Isotropic normality is not the direction the specimen takes.** The
state-generated basis carrying the current flow direction `N(sigma)`, banded by
equivalent stress, never gets meaningfully below 1.0 with dissipation enforced,
and reaches only 0.87 with the sign condition released at rank 32 -- no better
than the fixed basis it was meant to beat. Adding state dependence *of this
kind* buys nothing, which is consistent with the Ludwik verdict, where the
correction applied was orthogonal to the defect at cosine +0.006 to +0.038.

**Dissipation is affordable but not free.** Comparing the same modes with and
without `a >= 0`: 1.041 against 0.938 at rank 8, 0.936 against 0.872 at rank 32.
Roughly 0.07 of the metric, not the main gap. The reduced cone is *not* trivial
here -- the construction `phi_k = m_k N(sigma)` makes every non-negative
combination dissipative by design, which is why an NNLS finds nonzero solutions
where the Krylov QP found only `q = 0`.

**The fixed basis is thermodynamically inadmissible at half its points**, 48 %
of midpoint dissipations negative at every rank, stable across the sweep. It
buys its fit by dissipating negative energy in half the specimen.

## What follows

The spatial holdout measures extrapolation into a hole, which the mechanics
cannot supply. A temporal holdout does not have that defect: the basis is
regenerated from the state at every increment, so asking it to carry increments
it was not fitted on tests the thing the formulation actually claims. That is a
new campaign and needs its own preregistration before it is run.

The direction result points at the crystallography rather than away from it.
`N(sigma)` is the isotropic guess; the slip-system form
`sum_alpha gamma^alpha sign(tau^alpha) P^alpha` is dissipative on the same terms
and is the natural next basis, with the EBSD orientations now verified against
the DIC frame.
