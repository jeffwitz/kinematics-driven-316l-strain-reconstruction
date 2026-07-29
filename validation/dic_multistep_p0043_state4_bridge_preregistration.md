# P43 measured history: state-4 bridge preregistration

Date: 2026-07-29

## Question

Can the local P43 mechanics replay the measured boundary history when the first
temporally irregular state is replaced by a transparent piecewise-linear
bridge, while every other measured state and the constitutive evolution remain
unchanged?

## Selection rule fixed before mechanics

The repaired 40-state boundary history is analysed on the boundary nodes only.
State 4 has the largest RMS second temporal difference:

```text
state 4: 4.213850188534707e-4 mm
```

It is also the first target involved in the previously documented global
Newton failure. This agreement is used only to nominate a numerical
conditioning experiment; it does not establish that the experimental frame is
invalid.

The conditioned history shall replace state 4 by

```text
u_4 = 0.5 * (u_3 + u_5)
```

and shall preserve states 0--3 and 5--40 bit for bit. The final state, material
maps, constitutive law, tolerances and number of nominal states remain
unchanged.

## Constitutive-state rule

Plastic strain and every MFront internal variable shall **not** be
interpolated. The mechanics must integrate the two linear boundary increments
from the last committed state. Trial states remain transactional:

- evaluate every Newton trial from the last committed material state;
- commit once after global convergence;
- revert before any cutback;
- advance all internal variables only through constitutive integration.

## Predictor comparison

Two displacement predictors are admissible:

1. `elastic`: the existing elastic response to the current boundary increment;
2. `secant-corrected-elastic`: the current elastic response plus the
   time-scaled nonlinear correction measured over the previous converged
   increment.

The second predictor may extrapolate displacement only. It must never
extrapolate or interpolate constitutive internal variables. Both predictors
must converge to the same fields within `rtol=1e-10`, `atol=1e-12` on a
reduced regression case.

## Execution order

1. Generate and hash an immutable conditioned-history artefact.
2. Validate exact preservation of all non-replaced states.
3. Validate both predictors on reduced cases.
4. Run the full P43 case without Newton line search, first with the
   `secant-corrected-elastic` predictor.
5. Stop and report a negative result if the conditioned history still requires
   repeated minimum-size cutbacks. Do not add further skipped states
   opportunistically.

## Interpretation boundary

Success would show that a transparent temporal bridge and a better
displacement predictor make the measured-history replay numerically tractable.
It would not prove that state 4 is experimentally corrupted, and it would not
validate any constitutive parameter. Failure would remain a numerical result
and would not authorise automatic frame rejection or Kalman smoothing.
