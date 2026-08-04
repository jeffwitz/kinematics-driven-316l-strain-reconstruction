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

