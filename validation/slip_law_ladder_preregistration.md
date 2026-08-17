# The shared slip-law ladder — preregistration

Registered before any run. Thresholds frozen. Negative results kept.

## Object

The slip geometry shows a two-quadrant sign structure (imposed — the
dissipation cone we put in), an interior envelope that tightens at large
`|tau|` (discovered — and unphysical for a naive threshold law), and a
per-system history fan `Gamma^alpha -> Delta gamma^alpha` (the promising
one). Two experiments now:

1. **The gauge test.** The 12 slips represent a 5-D tensor, so the
   decomposition is a chosen representative. Decompose the *same* tensors
   three admissible ways — L2, L1, and a causally time-regularised variant
   (`lambda_t sum (gamma_n - gamma_{n-1})^2`), all under
   `tau^alpha Delta gamma^alpha >= 0` — and compare the **laws** each
   decomposition supports, not the pointwise activities. If the laws
   agree, the structure is imposed by the experimental tensors; if they
   diverge, the individual slips are not identifiable from the 2-D
   observation.
2. **The shared-law ladder.** One function for all twelve systems — the
   model never sees the system index, only its features — in k-NN
   leave-one-state-out on the pooled (state, point, system) samples:

   * S1: `(tau^alpha)`
   * S2: `(tau^alpha, Gamma^alpha_{n-1})`
   * S3: `(tau^alpha, Gamma^alpha_{n-1}, Gamma^{beta != alpha}_{n-1} summed)`
   * S5: `(tau^alpha, Gamma^1_{n-1}, ..., Gamma^12_{n-1})`

   `Gamma` is always the history *before* the increment — the causal
   convention, never including the response being predicted.

## Frozen bars

1. **Gauge stability.** The S2 ladder scores from the L1 and the
   time-regularised decompositions stay within `+/-0.05` of the L2 one.
2. **Slip phase space carries the structure.** The shared S2 law reaches
   LOSO `R^2 >= 0.30` (the tensor-phase baseline was 0.13).
3. **Latent hardening.** `R^2(S3 or S5) - R^2(S2) >= 0.10`: the history of
   the *other* systems controlling `alpha` is observed in the data.
4. **Crystallographic invariance.** The pooled shared law's `R^2` is at
   least `0.8 x` the best per-system `R^2` (the twelve systems need no
   separate laws).

## Outputs

`validation/_generated/shared_tensor_generator/slip_law_ladder.json` and
this file's results companion.
