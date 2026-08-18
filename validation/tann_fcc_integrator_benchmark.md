# Integrator stiffness benchmark: RK4 vs implicit Euler vs Radau

Requested after the amended run stalled at increment 17: test, on the
material states of the failing increment, whether the dynamics are
genuinely stiff (in which case an L-stable implicit integrator is the
right tool and RK4 is not). States: the committed state after increment
16 and the trial increment of 17, from the completed 25x25 smoke archive
(`state_36_*`), on the real EBSD geometry, `sigma_ref = 200 MPa`.
`scripts/bench_tann_integrators.py`.

## Results

| excursion scale | RK4 (limiter) vs Radau | implicit Euler vs Radau | Radau nfev |
|---|---|---|---|
| x1 | 4.3e-8 .. 6.3e-8 | 7.5e-4 .. 1.2e-3 | 319-344 |
| x2 | 6.1e-7 .. 9.0e-7 | 3.5e-3 .. 4.5e-3 | 554-596 |
| x5 | 1.6e-4 .. 3.5e-4 | 1.9e-2 .. 2.7e-2 | 1294-1448 |
| x10 | 0.21 .. 0.61 (limiter binding) | 0.60 .. 1.18 | 3085-3442 |
| x50 | finite (state 3.3) | finite (state 0.06) | **solve_ivp gives up** |

## Verdict

* The stiff-regime hypothesis is **refuted at the operating scales**: a
  stiff system is one where Radau takes large steps with few evaluations
  while RK4 explodes; here RK4 agrees with Radau to 1e-8..3e-4 at x1-x5
  with 16 RHS evaluations against hundreds for Radau.
* At extreme excursions (x50) even Radau fails: the excursion itself is
  outside any reasonable envelope, not an integrator-choice problem.
* The one-step implicit Euler is first order (error O(rate)); matching
  RK4's accuracy would require an SDIRK of order >= 2 at the price of
  36x36 local Jacobians per step. It is kept implemented and
  gate-tested as the fallback, not adopted.
* The Amendment-4 limiter is what keeps RK4 finite everywhere; the
  registered RK4 remains the T0 integrator. The increment-16/17 failure
  of the 100x100 run is an extreme Newton excursion -- the lever is
  excursion control in the global solver (trust region / bounded
  correction), not the constitutive integrator. `sigma_ref` stays 200 MPa;
  the integrator adapts to the law, not the reverse -- and the benchmark
  shows the law does not require it to.

## Implicit-Euler material gates (for the record)

The implicit path passes the zero-increment (exact) and dissipation
gates, but its algorithmic tangent **fails**: differentiating through
the unrolled local Newton amplifies the convergence residual (Richardson
errors ~5e9 against the 1e-5 bar). Making the implicit route
production-ready requires the implicit-function-theorem tangent
`dq/deps = -(dg/dq)^{-1} dg/deps` at the converged point -- a separate
piece of work. The fallback stays implemented but is not adopted; the
registered RK4 + limiter remains the T0 integrator.
