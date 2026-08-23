# Causal TANN-FCC identification from DIC — preregistration (T0)

> **Historical preregistration, superseded on 2026-08-23.** The implementation
> used by this campaign contained trajectory-reset, observation, tensor-
> convention and plane-stress defects. Do not relaunch it or interpret its
> thresholds without first reading `tann_fcc_recovery_strategy.md`.

Registered before any run. Thresholds frozen. Negative results kept.

## The scientific question

Can a **local, causal constitutive law** carried by the twelve FCC systems,
with its own latent internal variables and its own dynamics, produce —
after mechanical equilibrium — the experimental DIC fields?

The temporal continuity is **not** something the network may discover. It is
imposed structurally:

```text
Y_{n+1} = Integrate(F_theta, Y_n, loading n -> n+1)
```

with the explicit prohibitions `Y_{n+1} = Network(DIC_{n+1})` and
`Y_{n+1} = free_parameter[n+1]`: no image may own an independent plastic
state. This is the architectural translation of the closed discovery path
(the effective inelastic field is a kinematic inverse, not a constitutive
state; see the phase-space results recorded 2026-08-17).

## The architecture, frozen (T0)

* **Mechanics.** The existing Dirichlet spectral solver, unmodified. The
  TANN supplies the constitutive law; the solver supplies equilibrium.
* **FCC geometry.** The twelve octahedral systems from the existing
  SRIX/Méric convention, rotated per point from the EBSD Bunge maps —
  no second convention; a coherence test ties `tau^alpha = sigma : P^alpha`
  to the existing geometry to numerical precision.
* **State.** Per point and system: `gamma^alpha` (signed slip) and
  `z^alpha in R^d`, `d = 2`. Initialised at zero everywhere; no spatially
  free initial latent field. No physical names are attached to `z`.
* **Thermodynamics, by construction.**

```text
Psi = Psi_el(eps - eps_p) + 1/2 sum_alpha ||z^alpha||^2
A^alpha = - dPsi/dq^alpha                    (generalised forces)
dq^alpha/ds = ||Delta eps|| * M_theta^alpha A^alpha
M_theta^alpha = L_theta^alpha (L_theta^alpha)^T
```

  so `D = sum_alpha (A^alpha)^T M A^alpha >= 0` identically. The
  construction deliberately allows part of the work to be stored in the
  latent variables rather than forcing every system to dissipate on its
  own.
* **Network.** One network shared by all twelve systems: a local embedding
  `phi_theta(A^alpha, z^alpha)`, a permutation-invariant pool
  `c = mean_beta phi_theta(...)`, and a mobility head
  `L^alpha = rho_theta(local, c)`. No system identifier, no one-hot, no
  coordinates, no frame index. `latent_dim = 2`, width 32, two hidden
  layers, SiLU, float64 for the qualifications.
* **Integrator.** RK4, fixed `n_constitutive_substeps = 4`, rate scale
  `||Delta eps||` so `Delta eps = 0 => Delta q = 0` exactly. The substep
  invariance (1 vs 2 vs 4 vs 8) is qualified before any P43 run.
* **Elasticity.** Fixed, the solver's current one. No cubic elasticity in
  T0; no simultaneous changes of law, anisotropy and spatiality.

## Inputs and prohibitions

Allowed inputs: static EBSD orientation, FCC geometry, the previous
simulated internal state, simulated stress and strain, the solver's strain
increment, and (only in later stages) simulated spatial context.
Forbidden: current interior DIC as a network input, `eps_p_Krylov`,
`gamma_Krylov`, pixel indices, coordinates, image numbers. The measured
boundary conditions are part of the mechanical problem and may be used at
a held-out increment; interior DIC of the same increment may not.

## Loss

A displacement loss, not a derived-strain loss:

```text
L_DIC = 1/2 || W_D (u_sim - u_DIC) ||^2
```

on the observed interior degrees of freedom, with the existing qualified
spectral whitener `W_D` (never a dense covariance). No force/reaction
terms — none exist in this problem.

## Protocol and holdout

* P43 100×100, states 21-40, played as one causal trajectory from
  `q_0 = 0`; the state is never reinitialised between states.
* **Holdout: `{24, 28, 32, 36, 39}`** — masked states distributed over the
  history, state 40 kept in training so the strong-plasticity domain is
  seen. The test is temporal interpolation with missing observations, not
  online forecasting: at a holdout state `h` the trajectory continues from
  the **predicted** state `q_h_pred`, never from a state recalibrated with
  the DIC of `h`. Training states after `h` may indirectly constrain the
  dynamics that crossed `h` — this is intentional.
* The historical split `{24, 28, 32, 36, 40}` is computed secondarily for
  comparison with the earlier work.
* The noise margin on the split is recomputed with the existing margin
  machinery and **inscribed here before the first training run**.

## Differentiation

Discrete trajectory adjoint — no naive unrolling through Newton/GMRES.
At increment `n`: `R_n(u_n; q_{n-1}, theta) = 0`, and the material
provides `q_n = Q_n(u_n, q_{n-1}, theta)` after convergence. With the
local loss `ell_n(u_n)` and the material co-state `v_n`, the mechanical
adjoint `lambda_n` and the parameter contribution follow the registered
structure (signs re-derived from the discrete Lagrangian before
implementation). Local VJPs by AD on the material batch; the mechanical
adjoint stays matrix-free. The learned Jacobian is not assumed symmetric.
The small gate precedes any long training: 8×8 / 16×16, 2-4 increments,
gradient vs central FD `<= 1e-4` (target `1e-5`), dot tests `~1e-8`.

## Material qualifications, all before P43

Zero increment; `D >= -tol` on thousands of random states (no "8 %
negative power"); permutation equivariance of the twelve systems; substep
invariance 1/2/4/8; algorithmic tangent vs FD `<= 1e-5` (target `1e-6`)
with a step study; transaction semantics (double evaluate identical,
revert restores exactly, failed Newton leaves the state unchanged); FCC
geometry coherence with the existing convention.

## Frozen bars

1. **Minimal viability.** `median(E_holdout) < 1` where
   `E_n = ||u_model - u_DIC|| / ||u_elastic - u_DIC||` — the causal law
   carries constitutive information beyond elasticity.
2. **Robustness.** Improvement over the elastic baseline on **at least 4
   of the 5 holdout states**.
3. **Strong signal** (reported, not a barrier): `median(E_holdout) ~ 0.7`
   or below.
4. **Anti-latent-storage bar.** The split of the generalised work between
   the slip channel and the latent channel is reported per state. A model
   passing bars 1-2 with `W_gamma < 0.1 * W_D` is declared a **nonlinear
   elastic surrogate, not a plastic law** — the named failure mode the
   f_0 lesson predicts, registered before any result.
5. No threshold may move after results are seen. Negatives are kept: a
   qualified T0 with `median(E_holdout) >= 1` is the recorded scientific
   result that licenses the spatial context stages (T1/T2).

## Comparisons

The elastic baseline is mandatory. SRIX and Méric on exactly the same
Dirichlet data and states if trivially replayable. Krylov appears only in
a separate table titled "kinematic inverse reference — not constitutive".

## Artifacts

Every real run produces one self-contained JSON (git SHA, dirty status,
date, machine, seed, crop, states, holdout, architecture, latent dim,
normalisations, integrator, optimizer, losses, per-state metrics,
aggregates, Newton/Krylov iterations, timings, dissipation, slip and
state-continuity metrics) plus the training history, the checkpoint and
the exact configuration. Figures are generated from that artifact only.

## Frozen seeds

`20260817` for the primary run, `+1` per rerun; recorded in the artifact.

## Amendment 1 — force normalisation (2026-08-18, before any P43 run)

The first material-gate execution (seed `20260817`, before any training
run) exposed that the preregistered dynamics as first written are not
integrable at `n_constitutive_substeps = 4`. With the generalised force
`A = -dPsi/dq` fed unscaled into the mobility network, the elastic
feedback rate is

```text
c = ||Delta eps|| M (2 mu) (P : P) ~ 1e2 per substep
```

against the RK4 stability limit `c h <= 2.785` at `h = 1/4` — a factor
~50 over the limit. The measured symptom was the first RK4 substeps
growing `1e0 -> 1e10 -> 1e57 -> NaN` from the zero state at a
`2e-3` strain increment, i.e. divergence, not a coding defect. The
substep-invariance gate (D) could not converge on that formulation.

**The amendment.** The generalised force is normalised by
`sigma_ref = 2 mu = E/(1 + nu)` before entering the network:

```text
dq_alpha/ds = ||Delta eps|| M_alpha (A_alpha / sigma_ref),
M_alpha = L_alpha L_alpha^T
```

The GENERIC structure is unchanged — the mobility simply carries units
of `1/MPa`, and `D = sum_alpha A^T M (A/sigma_ref) >= 0` still holds
identically. The elastic feedback rate becomes `O(||d eps|| M)` and the
preregistered four-substep RK4 is stable by a wide margin. The
dissipation quadrature reports the true `A^T dq`, not a rescaled one.

No frozen threshold is moved by this amendment. It changes the network
input scaling only; `latent_dim`, width, layers, activation, integrator,
substeps, holdout, seeds and all bars of the preregistration remain
exactly as registered.

## Amendment 2 — noise margin on the split (2026-08-18, before any training run)

The preregistered protocol requires the noise margin on the split to be
inscribed here before the first training run, using the existing margin
machinery. The machinery is the archived 95 % surrogate-sensitivity
interval of `dic_uncertainty_propagation_p0043` (the same crop, states
and measurement chain): on the displacement-relative-L2 family of
metrics -- the family the primary metric

```text
E_n = ||u_model - u_DIC|| / ||u_elastic - u_DIC||
```

belongs to -- the archived interval width is **0.0202**
(`dic_multistep_p0043_observed_path_comparison_preregistration.md`,
archived interval `[0.4763, 0.4965]`).

Inscripted rule: a difference of `E_n` (or of its aggregate
`median(E_holdout)`) smaller than **0.0202** between two models is inside
the sensitivity of the metric to DIC noise alone and is recorded as
indistinguishable; the viability bars 1 and 2 are read with this margin
in mind (an `E < 1` claim below `1 - 0.0202` is not a significant
improvement over elasticity). The bars themselves are unchanged. A full
recomputation of the surrogate intervals was not rerun: the archived
campaign covers the same crop, states and measurement chain, and this
inscription states that correspondence explicitly rather than implying a
fresh computation.

## Amendment 3 — plastic-scale force reference (2026-08-18, inscribed from the first steps of the registered run, before any amended run)

The first two Adam steps of the registered run (seed `20260817`,
`sigma_ref = 2 mu` per Amendment 1) moved the whitened loss by
`4.518841e-05 -> 4.518777e-05` and `E_holdout` by `1.0519 -> 1.0520`:
the objective is flat. The cause is structural, and the measured numbers
close it: with the generalised force normalised by the ELASTIC modulus,
`A/sigma_ref ~ 1e-3` at the operating point, so the softplus mobility
sits on its `0.69` floor, the per-increment slip is
`||d eps|| M A_norm ~ 2e-6` against the `~1e-3` the measured elastic
defect requires, and the exact adjoint gradient is `3.8e-9` -- the
network cannot influence the response at this operating point, at any
capacity. The registered run continues to completion and its verdict is
recorded as the primary; the characterisation above is what makes it
interpretable rather than merely negative.

**The amendment.** The reference stress is a characteristic PLASTIC
stress, not the elastic modulus:

```text
sigma_ref = 200 MPa
```

(a yield-scale resolved stress for 316L, one order below the SRIX
thresholds of the earlier campaigns, chosen so the resolved stresses of
the P43 history operate the mobility away from its floor). The stability
argument of Amendment 1 is re-checked at this value, not assumed: the
RK4 stability condition `c h <= 2.785` becomes
`||d eps|| M (2 mu / sigma_ref) (P : P) h ~ 0.25` at the largest P43
increments -- stable by an order of magnitude -- and the material gates
are re-run at `sigma_ref = 200 MPa` before the amended run. The GENERIC
identity, all bars, the holdout, the seeds and the margin of Amendment 2
are unchanged.

## Amendment 4 — slope limiter on the substep flow (2026-08-18, before the amended rerun)

The first amended rollout diverged at increment 16: a Newton excursion
presented the integrator with a trial strain whose stiff elastic descent
(`rate (2 mu / sigma_ref) M (P : P) h` beyond the RK4 stability limit)
overflowed the stress to infinity, and neither the line search nor the
adaptive subdivision rescued it (every retrial hit the same excursion).
The fix is a smooth per-point slope limiter on the RK4 right-hand side:

```text
flow -> flow / sqrt(1 + ||flow||^2 / (kappa ||d eps||)^2),   kappa = 128
```

At the operating point the flow is `||d eps|| * O(1)` against
`kappa = 128`, so the law is unchanged there to better than `1e-3`;
a positive per-point scalar on the mobility preserves the GENERIC
structure, so `D >= 0` holds identically and the zero-increment property
stays exact. The limiter only binds on excursions far outside the
equilibrium envelope. The material gates and the full unit suite are
re-run with the limiter before the amended rerun. No bar moves.
