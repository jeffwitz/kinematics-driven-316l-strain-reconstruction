# Causal TANN-FCC on P43 — amended run status (sigma_ref = 200 MPa)

> **Historical incomplete run, superseded on 2026-08-23.** It never completed
> a qualified 100 x 100 training trajectory and predates the corrections in
> `tann_fcc_recovery_strategy.md`.

Amendments 3 and 4 registered before the run; the material gates and the
full unit suite green at the new operating point.

## What converged

* The **25x25 smoke** plays the full 17-increment trajectory at
  `sigma_ref = 200 MPa` (all increments converged). Step-0
  `E_holdout = 1.34`: the untrained strong-plasticity law is worse than
  elastic, as expected at initialisation, but the law now *moves* --
  the primary run's flatness is gone.
* The **100x100 amended run** converges increments 1-15 and, with the
  Amendment-4 limiter, increment 16 (after 4 failed adaptive
  subdivisions); the overflow that killed the first amended attempt is
  eliminated by the limiter.

## Where it stopped

Increment 17 of the 100x100 run does not converge within the adaptive
subdivision budget (at `1e-10` the run raised "maximum cutbacks per load
step exceeded"; a diagnostic pass at `1e-8` was stopped at the same
wall). The failures occur early in the subdivisions (no long Newton
stalls), i.e. the first equilibrium trial of the subdivided increment is
already outside the basin. This is a solver/law-coupling problem at the
late, strongly plastic states of the full-field run -- not an
identifiability verdict: the material-level gates and the small-field
trajectory are qualified.

## Registered follow-up (not a threshold change)

The amended 100x100 run remains open. The next levers, in the solver's
own machinery, are: equilibrium tolerance `1e-8` (partially tested,
insufficient alone at 17), preconditioner reference refresh per
increment, and a larger adaptive cutback budget. The law, the
integrator, the holdout, the seeds and all bars stay as registered.

## Final state of the campaign (2026-08-19, campaign stopped by decision)

* With the Amendment-4 limiter at `1e-8` the 100x100 run now converges
  increments 1-17 (16 and 17 previously impossible) and stalls at 18
  with the adaptive cutback budget exhausted at the finest subdivision.
* The integrator benchmark (`validation/tann_fcc_integrator_benchmark.md`)
  refuted the stiff-regime hypothesis: RK4 + limiter matches Radau at
  the operating scales; the one-step implicit Euler is first-order and
  its unrolled-Newton tangent fails (needs IFT); Radau itself gives up
  at extreme excursions. The failure is a global-solver equilibrium
  problem, not an integrator one.
* Per-increment trajectory checkpointing exists
  (`scripts/train_tann_fcc_p43.py --resume-increment`); the resume is
  approximate (Newton warm-start not restored) and is documented as an
  exploration tool.
* `reference_update_mode="per_increment"` is a measured regression.
* The EVM comparison figure (DIC archive vs elastic/TANN, same von Mises
  definition both sides) validates the elastic mechanics at state 40
  (1 % agreement) and shows the early-state ratio is metrology-floor
  dominated (positive-definite EVM under noise), not an elastic failure:
  on a full-Dirichlet problem the elastic field is modulus-independent,
  so no calibration could close the gap -- the gap IS the plastic
  signature to explain.
* Decision recorded: the campaign stops here. The primary verdict stands
  (`median(E_holdout) = 1.052`, bars 1-2 failed, structural diagnosis in
  Amendment 3); the amended operating point is qualified at the material
  level and blocked at the 100x100 equilibrium -- the honest open item
  for a future session is the solver's late-increment convergence, not
  the law.
