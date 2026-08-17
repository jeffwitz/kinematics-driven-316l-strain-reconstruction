# Local path memory — results

Against `validation/path_memory_preregistration.md`, thresholds frozen
before the runs. The reading is **nothing** — and the monotone decline with
window depth is what makes it decisive.

## The windows (LOSO R², magnitude target, same population as the ladder)

| window | features | R² | predicted |
|---|---|---|---|
| baseline | `(|tau_n|, Gamma_{n-1})` | **+0.036** | 400k |
| W1-tau | `tau_n, tau_{n-1}, Delta tau_n` | -0.017 | 380k |
| W1 | + `Delta gamma_{n-1}` | -0.052 | 380k |
| W2 | two steps | -0.109 | 360k |
| W4 | four steps | **-0.191** | 320k |

## The frozen reading

`< 0.10` on every window — the registered outcome: **the 2-D effective
field does not contain alone an exploitable local constitutive closure.**
The discovery path is closed with this as the recorded evidence.

Two further facts, both informative:

1. **The decline is monotone with the window depth.** Every added step of
   the observable past makes the prediction worse — no window length shows
   even a hint of a structure to compress. This is not a failure at one or
   two steps; the past is actively anti-predictive across increments.
2. **Neither past carries it.** The loading path (`W1-tau`, negative) does
   not hold the missing information, and the response's own past (`W1`
   worse than `W1-tau`) does not either — the increment-to-increment
   structure the in-sample Spearman showed is not a local path effect the
   data can hand to a law.

## What this closes, and what it does not

Closed, with evidence: the search for a local state — scalar, tensorial,
resistance-like, first-order dynamical, or a window of the observable past
— that makes the *effective* inelastic response predictable from the
reconstructed phase space alone. Four independent negative protocols
(ladder, resistance, memory families, path memory) agree.

Not closed: whether a *known* constitutive structure (SRIX/Méric internal
variables, closed-loop dynamics) explains these trajectories — that is a
different question, and it remains open. Nor does anything here bear on
the forward path: putting the constitutive structure back into the model
and validating against the DIC — which this whole chain of analyses has
now qualified as the validation instrument.

## Conclusion

The reconstruction's effective field is kinematically exact (FCC span,
machine precision), gauge-stable, system-invariant — and constitutively
empty beyond the in-sample driving-force correlation. The increments
carry the structure; no local state recoverable from the 2-D field
compresses it. The project's next step is the forward path: a constitutive
generator inside the equilibrium problem, validated by these instruments
against the held-out DIC.
