# P43 measured-history bridge and stateful-predictor result

Date: 2026-07-30

## Result

The experiment is a **negative numerical result**. Replacing the largest
temporal-curvature state by a piecewise-linear displacement bridge and using a
secant-corrected elastic displacement predictor does not make the full P43
measured-history solve converge.

No constitutive internal variable was interpolated or extrapolated.

## History conditioning

The corrected temporal indexing identifies state 3 as the largest RMS second
difference of the boundary history:

| Quantity | Value |
|---|---:|
| bridged state | 3 |
| source RMS second difference | `4.21385e-4 mm` |
| maximum boundary-component change | `2.59829e-4 mm` |
| unchanged states | 0--2 and 4--40, bitwise |
| unchanged final state | yes, bitwise |

The bridge is

```text
u_3 = 0.5 * (u_2 + u_4).
```

The diagnostic figure is
`reference_data/dic_multistep_history_p0043_state3_bridge_v1/state_bridge_diagnostic.png`.

## Mechanical replay

The local native-plane-stress MFront solve used:

- 40 nominal measured states;
- the state-3 bridge above;
- no Newton line search;
- the `secant-corrected-elastic` displacement predictor;
- unchanged constitutive law, material maps, tolerances and cutback policy.

It converged through state 3, then failed on the transition to state 4:

| Diagnostic | Value |
|---|---:|
| converged increments | 3 |
| attempted increments | 14 |
| cutbacks | 11 |
| Newton iterations | 28 |
| secant predictor uses | 13 |
| first failed target pseudo-time | `0.10` |
| last failed pseudo-time | `0.0750244140625` |
| first rejected maximum engineering strain | `69.529` |
| last rejected maximum engineering strain | `90.230` |

The first rejected strain is lower than the `82.257` observed with the
unconditioned elastic predictor, but it remains nonphysical. The predictor
therefore changes the route to failure, not the converged mechanics.

## Constitutive-state interpretation

The current implementation already applies the correct state protocol:

1. start every trial from the last committed MFront state;
2. integrate the constitutive response along the imposed displacement
   subincrement;
3. commit once after global Newton convergence;
4. revert before cutback.

Direct interpolation or extrapolation of plastic strain, PEEQ or other MFront
state variables is neither required nor admissible. It would create a state
that need not satisfy the constitutive evolution.

## Consequence

State 3 was selected from a measurement-only temporal-curvature diagnostic.
The remaining blocked target is state 4. A future experiment may explicitly
bridge state 4, or the complete interval states 3--4, but that is a distinct
solver-conditioning experiment and must be preregistered separately.

A Kalman filter is not introduced. Without an independently validated temporal
noise/state model, it would smooth the prescribed experiment and make the
mechanical path harder to audit. Piecewise-linear bridging remains the
preferred transparent diagnostic.

This result does not modify any scientific field comparison and does not
authorise micromorphic re-identification.
