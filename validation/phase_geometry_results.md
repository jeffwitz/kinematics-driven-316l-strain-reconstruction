# Phase-space geometry — results

The visual question asked before any statistical tool: what geometry do the
400 000 admissible local experiments draw in constitutive space — a curve,
branches, clusters, or a cloud? The answer is measured, not eyeballed
(observable-projected fields, predictive projected-Krylov line at r=16).

## The geometry, in numbers

| question | quantity | measured | geometry |
|---|---|---|---|
| 1-D law `Delta p = f(sigma_eq)` | Spearman `(sigma_eq, dp)` | **0.35** | weak trend |
| scatter at fixed `sigma_eq` | CoV of `dp` within bins | **0.74–1.41** | cloud, not curve |
| history variable | Spearman `(p_eq, dp)` | **0.52** | the best scalar |
| branching by history | within-bin Spearman `(dp, p_eq \| sigma_eq)` | **0.44** | partial, no clean separation |
| J2 direction law `n ∥ s` | circular correlation `(theta_s, theta_n)` | **-0.04** | exact independence |
| share on the J2 diagonal (17°) | | **9.0 %** | uniform expectation is 9.5 % |

## What the geometry says

1. **The amplitude is a cloud with a faint trend.** The median `Delta p`
   rises with `sigma_eq` (1.5e-4 → 4.2e-4 across bins) but the within-bin
   scatter stays at 74–141 % of the median: at a fixed stress, the increment
   varies over an order of magnitude. No yield threshold is visible.
2. **The history level is the best scalar, and still insufficient.**
   `p_eq` correlates with `Delta p` at 0.52 and partially separates the
   cloud inside `sigma_eq` bins (0.44) — branches exist in the amplitude
   dimension, blurred, not clean.
3. **The direction is statistically independent of the stress.** The
   circular correlation is -0.04 and the J2 diagonal carries exactly the
   uniform share (9.0 % vs 9.5 %): the flow direction draws no curve, no
   diagonal, no family in `(theta_s, theta_n)`. Combined with the clustering
   verdict (orientation families do not condition the direction either),
   the direction of the *effective* inelastic increment is not a function
   of any tested local state.

## Figures

All saved under `validation/_generated/shared_tensor_generator/`:

* `phase_geometry_sigmaeq_dp.png` and `_time`, `_schmid`, `_position` —
  the 1-D law test, colored by time, orientation summary and position;
* `phase_geometry_peq_dp.png`, `phase_geometry_sigmaeq_peq.png` — the
  history variable and the visited domain;
* `phase_geometry_flow_direction.png` — `(theta_s, theta_n)` density with
  the J2 diagonal dashed;
* `phase_geometry_amplitude_branches.png` — the branching test colored by
  `p_eq` quantile.

## Conclusion

Not a curve, not clean branches, not clusters — **no discrete constitutive
regimes, but a continuous geometry**: the `p_eq -> Delta p` fan carries
about a quarter of the amplitude variance in-sample, the direction is
exactly isotropic, and the visited `(sigma_eq, p_eq)` domain is a genuine
two-dimensional region. The corrected reading, confirmed by the continuous
conditioning analysis (`validation/phase_space_conditioning_results.md`):
the state `(sigma_eq, p_eq)` conditions the amplitude weakly and
state-specifically (LOSO `R^2 = 0.13`), orientation conditions nothing, and
the direction is conditioned by no tested state. The decisive next step
remains the `Delta eps^inel = Delta eps_D + Delta eps_0` decomposition,
with this same geometry rerun on the dissipative component alone.
