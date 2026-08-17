# Local path memory — preregistration

Registered before any run. Thresholds frozen. Negative results kept.

## Object

The scalar memories built from the *activity* fail. The remaining suspect is
the **local loading path**: two points with the same `(tau_n, Gamma_n)` can
have arrived by opposite paths (`100 -> 150 -> 200` vs `300 -> 250 -> 200`),
and sign reversals are exactly what `Gamma` destroys. Before closing the
phase-space discovery path, give the model a small window of the observed
local past — nearly non-parametric, no tuning — and ask the fundamental
question:

```text
Is the information the response needs contained in the local observable past?
```

## Windows (signed features, target `|Delta gamma_n^alpha|`, k-NN k=50, LOSO)

| window | features |
|---|---|
| W1-tau | `tau_n, tau_{n-1}, Delta tau_n` (loading path only) |
| W1 | + `Delta gamma_{n-1}` (the response's own past) |
| W2 | `tau_n, tau_{n-1}, tau_{n-2}, Delta gamma_{n-1}, Delta gamma_{n-2}` |
| W4 | four steps of both |

Baseline in the same metric: `(|tau_n|, Gamma_{n-1}) -> |Delta gamma_n|`
(measured 0.036). Samples without a full causal past (the first states) are
excluded from prediction; the training fold always has a full past.

## Frozen readings

1. **Closure candidate.** Any window reaches LOSO `R^2 >= 0.30`: a local
   closure exists and needs path memory — the next step is compressing
   that memory into an internal variable.
2. **Partial.** Any window reaches `R^2 >= 0.10` (below 0.30): the path
   carries part of the missing information, not all.
3. **Nothing.** All windows stay below `0.10`: the 2-D effective field
   does not contain alone an exploitable local constitutive closure — the
   "generator in the forward model + DIC validation" path becomes the
   convincing one, and the discovery path is closed with this as the
   recorded evidence.
4. **Loading vs response.** The W1 vs W1-tau gap names which past carries
   the information: the stress history (a memory the law could legitimately
   compress) or the previous response (which a closed law must regenerate,
   and whose predictive power is the marker of the increments carrying the
   structure).

## Outputs

`validation/_generated/shared_tensor_generator/path_memory.json` and this
file's results companion.
