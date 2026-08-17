# FCC slip decomposition of the effective inelastic field — preregistration

Registered before any run. Thresholds frozen. Negative results kept.

## Object

Absorb the EBSD orientation into the mechanics the way a crystal does: for
every point, rotate the twelve octahedral FCC systems into the specimen
frame, resolve the stress onto them, and decompose the reconstructed
effective inelastic increment onto them — then ask whether the phase space
becomes a slip-law phase space.

```text
sigma_n   ->  (tau^1, ..., tau^12),          tau^alpha = sigma : P^alpha
Delta eps^inel_n  ->  (Delta gamma^1, ..., Delta gamma^12) + r
```

with `P^alpha = 1/2 (s^alpha o m^alpha + m^alpha o s^alpha)` in the specimen
frame, and the decomposition minimising the plastic-gauge distance
`(2/3) |Delta eps - sum_alpha Delta gamma^alpha P^alpha|_F^2` under the
per-system dissipation constraint `tau^alpha Delta gamma^alpha >= 0`.

## The two admissible decompositions, and their stability

The 12 systems span a 5-dimensional deviatoric space, so the decomposition
is not unique. Two registered choices, both under the sign constraint:

* **diffuse** (L2): minimise the constraint alone (`lambda = 0`);
* **sparse-favouring** (L1): add `lambda sum |Delta gamma^alpha|` with
  `lambda = 1e-6 x scale`.

The stability across the two choices is itself information: the correlation
between the represented parts `sum gamma P` of the two variants is reported
and must be `>= 0.9` for the decomposition to be declared variant-stable.
Both are solved with the same projected FISTA (200 iterations, cone
projection per sign), vectorised over the 400 000 points, together with the
unconstrained least squares as the representability ceiling.

## Frozen bars

1. **Representability.** The effective increment is declared
   *largely FCC-representable* if the median `e_FCC <= 0.5` **and**
   `rho_FCC >= 0.5` on the observable-projected increments, with
   `e_FCC = |r|_Gp / |Delta eps|_Gp` and `rho_FCC` the represented share.
   The raw-field numbers are reported beside.
2. **Variant stability.** Correlation of the represented parts `>= 0.9`.
3. **Slip-law structure.** The per-system k-NN leave-one-state-out
   conditioning `(tau^alpha, Gamma^alpha) -> Delta gamma^alpha`, aggregated
   over the systems with significant activity, reaches `R^2 >= 0.5` for the
   decomposition to be declared a slip-law phase space. The 12-system
   interaction test (all `tau`, all `Gamma` as features) is reported after.

## The registered caveat

Only the in-plane tensor is observed: `eps_xz`, `eps_yz` are unknown and
`eps_zz` follows the plastic-incompressibility closure. The recovered
activities are therefore **FCC slip activities compatible with the observed
2-D projection**, not claimed true slips — and this ambiguity is measured
via `e_FCC`, not ignored.

## Outputs

`validation/_generated/shared_tensor_generator/fcc_slip_decomposition.json`
plus the per-system work shares and the representability histogram.
