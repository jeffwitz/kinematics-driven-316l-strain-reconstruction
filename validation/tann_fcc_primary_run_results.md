# Causal TANN-FCC on P43 — primary run verdict (registered seed 20260817)

> **Historical and invalid for constitutive claims.** The response is retained
> for provenance only. The forward trajectory was not reset between Adam
> steps and the loss applied the DIC transfer to the measured field a second
> time. See `tann_fcc_recovery_strategy.md`.

Run artifact: `validation/_generated/shared_tensor_generator/tann_fcc_p43_run.json`
(+ companion `tann_fcc_p43_run.npz`), figures
`validation/figures/tann_fcc_p43/`. Commit `e367ce0`.

## What was run

The preregistered T0 (Amendments 1-2): GENERIC TANN on the twelve FCC
systems from the P43 EBSD, `sigma_ref = 2 mu`, four-substep RK4, causal
trajectory over states 21-40 played once from `q_0 = 0`, holdout
`{24, 28, 32, 36, 39}`, whitened displacement loss, gradients by the
discrete trajectory adjoint, Adam `lr = 1e-3`, four steps
(~4.7 h total; rollout ~40 min, adjoint ~20-48 min per step). All
material gates green before the run; the adjoint FD gate green; the
noise margin (0.0202) inscribed.

## Verdict against the frozen bars

| bar | value | verdict |
|---|---|---|
| 1. `median(E_holdout) < 1` | **1.052** | **failed** |
| 2. improvement on >= 4 of 5 holdout states | **1 of 5** (state 24: 0.836) | **failed** |
| 3. strong signal `~0.7` (reported, not a bar) | no | — |
| 4. anti-latent-storage bar | not reached (bars 1-2 already failed) | — |

Per-state `E_n` at the final step: 24: **0.836**, 28: 1.194, 32: 1.052,
36: 1.028, 39: 1.137. The whitened loss is flat across the four steps
(`4.518841e-05 -> 4.519175e-05`, slightly upward); the exact adjoint
gradient norm is `3.8e-9 -> 5.1e-9`.

## Why — the structural diagnosis (Amendment 3)

The run is interpretable, not merely negative. At `sigma_ref = 2 mu` the
normalised generalised force is `A/sigma_ref ~ 8e-4` at the operating
point, so the softplus mobility sits on its `0.693` floor, the
per-increment slip is `||d eps|| M A_norm ~ 8e-7` against the `~1e-3`
the measured elastic defect requires, and the chain-rule sensitivity of
the slip to the weights is `~1e-9` -- matching the measured gradient.
The network cannot influence the response at this operating point at any
capacity; the effective law is a fixed-mobility linear viscoelastic
response that the elastic baseline already covers. The numbers were
inscribed in Amendment 3 BEFORE the amended run, not after seeing it.

## What the failed primary licenses

Per the preregistration, a qualified T0 with `median(E_holdout) >= 1` is
the recorded scientific result; the follow-up is the registered
Amendment-3 operating point (`sigma_ref = 200 MPa`, material gates
re-run and green, RK4 stability re-verified), whose run is
`tann_fcc_p43_run_sr200.json`. The bars are unchanged.
