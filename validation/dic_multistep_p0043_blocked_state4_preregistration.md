# P43 measured history: blocked target state-4 bridge preregistration

Date: 2026-07-30

## Purpose

Test the explicit numerical-conditioning proposal that a measured target which
blocks global equilibrium may be omitted and replaced by a linear boundary
path between its retained neighbours.

This experiment is separate from the measurement-only temporal-curvature
experiment. State 4 is selected because it is the first failed **target** in
both:

- the repaired measured history with the elastic predictor;
- the state-3 bridge with the secant-corrected elastic predictor.

## Fixed history

Use the immutable repaired history and replace only state 4:

```text
u_4 = 0.5 * (u_3 + u_5).
```

States 0--3 and 5--40, including the final endpoint, must remain bitwise
identical to the repaired source.

## Fixed mechanics

- local native-plane-stress MFront J2;
- 40 nominal states;
- existing material maps and solver tolerances;
- `boundary_history_predictor = elastic`;
- Newton line search disabled;
- existing cutback policy;
- no nonlocal coupling.

The elastic predictor is restored to isolate the effect of bridging the blocked
target. No constitutive internal variable is interpolated. Every internal
variable evolves by MFront integration from the last committed state.

## Decision

- **success:** all 40 target states converge and the report contains finite
  final fields;
- **negative result:** any target exhausts the unchanged cutback policy.

If another target fails, stop. Do not add another bridged state in the same
campaign. A sequence of solver-driven omissions would change the measured
loading path and requires a separate policy and sensitivity analysis.

No FEM--DIC metric or constitutive parameter is used to select the bridge.
