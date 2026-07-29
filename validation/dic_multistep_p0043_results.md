# P43 measured-boundary multi-step result

## Question

Can the local P43 reconstruction replay the 40 measured DIC boundary states
instead of assuming a proportional ramp to the final displacement?

## Measurement history

The 40 states were correlated directly from the same reference image.  They
are therefore an ordered pseudo-time history, not an incrementally accumulated
optical-flow history and not a force-synchronised loading history.

The OpenCV 4.14 reconstruction of the last state differs from the immutable
prepared endpoint by:

- component RMS: `7.764e-4 mm`;
- maximum component: `1.209e-3 mm`;
- relative displacement-vector norm: `1.583 %`.

The history was consequently anchored by a linear-in-state correction whose
final value is the prepared endpoint.  This preserves every deviation from a
proportional path relative to the newly reconstructed final field.

## Documented corrupted states

The first mechanics attempt revealed extreme transient EVM at states 31 and
32.  This is independently consistent with the historical map-identification
script, which declares frames 31 and 32 corrupted.

The pre-registered displacement repair interpolates states 31 and 32 between
states 30 and 33.  It changes no other state.

| Diagnostic | Original | Repaired |
|---|---:|---:|
| maximum incremental EVM | `5.459e-2` | `5.623e-3` |
| state 31 EVM maximum | `5.893e-2` | `9.681e-3` |
| state 32 EVM maximum | `4.204e-2` | `1.034e-2` |
| unaffected states bitwise identical | — | yes |
| final state bitwise identical | — | yes |

The primary diagnostic is
`reference_data/dic_multistep_history_p0043_repaired_v1/corrupted_frames_diagnostic.png`.

## Mechanical replay

The repaired measured-boundary calculation **does not converge** with the
unchanged nominal local solver:

| Quantity | Value |
|---|---:|
| nominal increments | 40 |
| converged history knots | 3 |
| first failed target | state 4, pseudo-time `0.10` |
| first failure | MFront integration status `-1`, Newton iteration 7 |
| cutbacks | 11 |
| last attempted pseudo-time | `0.0750244` |
| committed state after failure | no |

The failure occurs before the repaired frames.  Removing their optical-flow
artefact was necessary for a valid history, but it is not sufficient to make
the current undamped Newton path robust to the measured non-proportional
boundary evolution.

## Interpretation boundary

This is a **negative numerical result**, not evidence that the measured loading
path is physically impossible.  It establishes:

1. the historical frames 31--32 are corrupted and must not be imposed raw;
2. the current proportional-ramp solver cannot yet replay the measured path
   merely by substituting boundary snapshots;
3. repeated cutback alone does not resolve the first difficult transition;
4. no multi-step FEM/DIC prediction claim is authorised yet.

No solver tolerance, constitutive parameter, material map, or non-local
parameter was changed.  The next numerical diagnostic is to distinguish a
MFront integration limitation from a global Newton overshoot on the
state-3-to-state-4 transition.  A line search or a different constitutive
backend must not be adopted without a separate pre-registration and parity
test.
