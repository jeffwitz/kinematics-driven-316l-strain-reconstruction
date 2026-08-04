# EBI-TET verdict

`experimental_negative`

EBI-TET solves the targeted numerical problem: one SRIX state per pixel, two
kinematic samples, exact matrix-free tangent, no empirical stabilisation, and
verified convergence beyond `1e-8`. It also suppresses the one-point
high-frequency instability.

It is not scientifically qualified against CPS4: the registered plastic case
still exceeds every 1% field threshold under refinement through 24x24, especially
for accumulated slip and reactions. This triggers falsifier F9. Keep the branch
as a documented negative experiment; do not merge it as the production solver.

The causal decomposition is now complete: with the same TRI2 stencil, TET2 vs
CPS4 reaches `0.72%` in accumulated slip at 24x24, whereas EBI vs TET2 remains
`5.39%`. The dominant defect is the one-state SRIX sharing assumption. The
simple one-state EBI plastic formulation should therefore be closed for this
target; adding more B0 tuning or inexact GMRES forcing cannot remove that error.
