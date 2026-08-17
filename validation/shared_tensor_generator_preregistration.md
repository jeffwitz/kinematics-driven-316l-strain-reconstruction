# Milestone 4 — the shared tensor generator, designed against the measured nullspace

Registered before any run. Thresholds frozen. Negative results kept.

## Why this milestone exists

Milestone 3 (`b57914d`) quantified the two extremes of the plastic
representation, on identical mechanics and an identical objective:

```text
Delta p + n_J2 :  identifiable to 2.1e-4,  but 5.19 decades worse at fitting
free tensor    :  5.19 decades better,    but irreducibly 52-80 % wrong as a field
```

The scalar family is identifiable **because it is restricted**: one amplitude
per point pinned to the spatially varying `n_J2(sigma)`, so it cannot produce
the uniform eigenstrain that the free family can. The free family's failure is
not the optimiser — a uniform eigenstrain produces exactly zero displacement,
and more generally every self-equilibrated eigenstress is invisible (the whole
Airy family in two dimensions). The measured nullspace is 19 of 192 directions
with condition `3.5e16`.

The displacement data therefore determines `Delta eps^p` only modulo
`ker A`. This milestone tests whether a **shared learned structure** is the
minimal restriction that regularises the kernel without falling back to the
far-too-strong `Delta p * n_J2` constraint.

## Decisions adopted before any run

* **The global-reaction avenue is dropped.** This problem has no synchronised
  measured reaction (lot V0). No gate, no architecture decision and no further
  benchmark may rely on a force observable that does not exist.
* **The kernel is a property of the observation, not a defect to repair.** It
  exists, it is numerically important, and that is sufficient. No campaign to
  enumerate the Airy family, no attempt to resolve the kernel with more
  coefficients. The measured 19 directions and the 52-80 % floor are the design
  facts; they are reused below, not re-measured.
* **The shared structure is the identifiability device, not an efficiency
  device.** A network with enough capacity can re-parameterise the kernel
  degrees of freedom; the safety of this milestone therefore lives in its
  gates (temporal holdout, transversality, kernel-component stability), not in
  the architecture.

## The model under test

```text
S_n(x)  --F_theta-->  Phi_n(x) = [phi_{n,1}, ..., phi_{n,r}](x),  phi in R^3 (Kelvin)

v_n(x) = Phi_n(x) a_n,        a_n in R^r fitted per increment

Delta eps^p_n(x) = P_{H(sigma_pred,n(x))} [ v_n(x) ],   H(sigma) = { z : sigma^T z >= 0 }
```

* `theta` is shared across **all** increments; `a_n` is the only per-increment
  unknown. `r in {4, 8, 16}`; width 48, as already implemented and qualified.
* **No current DIC in the generator input.** Input channels are the
  predictor state only: stress, strain, accumulated plastic state and plastic
  path, all Kelvin, each normalised by its own standard deviation. No interior
  DIC, no spatial coordinates. The boundary kinematics drive the predictor —
  they are the measured statement of the problem, not an interior leak.
* The projection is applied **after assembly** (`P_H(Phi a)`, never
  mode-wise); `P_H` is the qualified half-space projection, and because
  `sigma_zz = 0` in plane stress the in-plane Kelvin product equals the full
  `sigma : eps^p`, so the half-space is the complete thermodynamic condition.
* Coefficients per state come from the qualified normal-equation route with
  ridge `1e-6` (free arm), or the constrained non-negative solve
  (dissipative arm). Fixed-point rounds 4.
* The mechanics is the qualified linear operator `A: Delta eps^p,K -> u` with
  an exact `A^T` — no Newton loop, no new mechanics is implemented or needed.

The architecture already exists as `scripts/learn_flow_direction_p43.py`
(asserted Kelvin conversions, elastic-lift residual check, temporal holdout,
adjoint dot-product test at `1e-8`). This milestone **freezes the experimental
design around it** and adds the gates below. No new synthetic problem is
built; no network is redesigned.

## Protocol and data

* P43 100x100, origin `(1580, 1030)`, reference state 20, states 21-40,
  temporal holdout `{24, 28, 32, 36, 40}`. Training states are the rest.
* 400 steps, learning rate `3e-4`, orthogonality penalty `1e-2`, dissipation
  penalty `1e-2`, ridge `1e-6` — the qualified script defaults.
* Five seeds, `{20260816, 20260817, 20260818, 20260819, 20260820}`, each run
  twice (Adam and AdamW) for gate 6. No run is ever labelled "exploratory"
  after the fact: everything below is part of the frozen design.
* The objective is the defect-scaled residual already implemented,
  `sum_s |eps_obs - eps_sim|^2 / defect_s^2`, plus the two penalties above.
  **The milestone-2 trap is registered as a clause:** any new objective
  implementation must scale by a fixed data norm before optimisation, and a
  Gauss-Newton or direct-normal-equation arm must exist so that no stopping
  rule decides a scientific verdict (the `07c1fc9` false negative is the
  precedent).
* **The milestone-3 trap is registered as a clause:** every arm must carry a
  `moved` flag; an arm that never moves measures a dead ReLU, not a family
  difference (the `b57914d` dead-ReLU scalar arm is the precedent).
* **Window caveat, carried from the standing rules.** The 100x100 crop has
  measured DIC imposed on its contour; a just-outside plastic source can be
  re-attributed inside. Results here qualify the machinery and rank the arms;
  they are never proof about the material.

## Metric

Per held-out state `s`:

```text
E_s = |eps_DIC,s - eps_sim,s| / |eps_DIC,s - eps_elastic,s|,
```

reported per state, and the decision quantity is the **median over the five
held-out states**. Both strains come from the same Kelvin derivation applied
to displacement fields, so the comparison is symmetric by construction. The
observation operator is the direct `kelvin_strain` of the displacement
history, exactly as in the qualified script; the transfer-corrected comparison
is not reopened in this milestone.

The noise margin on `E` is defined operationally before the first training
run: the archived DIC repetition residual
(`validation/reference_data/dic_uncertainty_propagation_p0043_v1/centred_repeat_flow_pixels.npy`)
propagated through the same Kelvin derivation, giving
`margin(E) = median_s |eps_noise,s| / defect_s`.

**Frozen 2026-08-17, before any run: `margin(E) = 0.204`** — computed by
`scripts/freeze_generator_margin.py`; artifact
`validation/_generated/shared_tensor_generator/margin_frozen.json`.
Guard: the replicated elastic-lifting residual is `1.5e-10` (the broken
historical conversion would read `0.32`). Per-state margins for states
24/28/32/36/40 are `0.92/0.36/0.204/0.196/0.176` — states 24 and 28 are
noise-dominated; the decision quantity remains the median. No comparison
below one margin means anything.

## The four arms

All arms share the identical linear mechanics and the identical loss. Only the
representation differs.

| arm | representation | role |
|---|---|---|
| A1 | `Delta p(x) n_J2(sigma_pred(x))`, `Delta p >= 0` | identifiable baseline |
| A2 | learned `Phi`, free signed coefficients | separates "tensor freedom" from "sharing" |
| A3 | learned `Phi`, dissipative by construction (`P_H` after assembly) | **the candidate** |
| A4 | free tensor per increment, 192 local coefficients (milestone 3 family) | fit ceiling, unidentifiable |

Registered expectations on the holdout median, per rank:

> **Amendment, recorded before any run.** The gate-7 criterion was originally
> frozen at two margins. The frozen margin (`0.204`) made two margins
> (`0.41`) unreachable given the already-measured J2-vs-learned gap
> (`~0.26`), so the criterion was re-frozen at **one margin** before any
> result was examined. No threshold changed after results exist.

* A4 is the ceiling: `E_A3 >= E_A4` always; equality to within half a decade
  (`E_A3 <= 3 * E_A4`) means the shared structure retains most of the tensor
  gain.
* `E_A3 <= E_A1 - margin(E)`: the generator must beat J2 beyond noise, at
  least at one rank; a rank where it does not is reported, not hidden.
* A2 sits between and is the ablation: if A3 beats A1 but A2 does not, the
  gain is dissipation, not freedom.

## Gates, in order

**Gate 1 — the transpose of the full chain.** Dot-product test through the
network and the mechanics, relative `<= 1e-8`, as the script already asserts.

**Gate 2 — the gradient.** Central differences against the adjoint of the
full objective in `(theta, a_n)` over a four-decade step sweep, best relative
error `<= 1e-5` **and** V-shaped. The milestone-3 clauses are inherited
verbatim: the base point is chosen away from the projection kink, the sweep is
rejected if any point changes activity, and if the objective proves exactly
quadratic in the swept variables the V-shape is replaced by the constant
second-difference argument — proved, not asserted.

**Gate 3 — transversality.** At the trained `theta`, per held-out increment,
the singular values of `du/da_n`: **all `r` above the combined floor**
(`validation/reference_data/p0043_small_parameter_matrix_v1/floors.json`,
measurement and solver floors in quadrature). A singular value at or below the
floor means the manifold `M_theta` is tangent to the fibre
`Delta eps^p + ker A` at that state: the shared structure does not resolve the
kernel there, and the gate fails regardless of any score. The joint
`(theta, a_n)` Gauss-Newton spectrum is reported in full, without threshold —
it informs the next design, not this gate.

**Gate 4 — twin with controlled invisible share.** The milestone-3 lesson is
a clause: a twin truth with a large spatial mean is mostly invisible (78 % of
the registered truth, floor 80 %), and the gate then measures the truth's
construction, not the method. The truth here is a smooth, strictly
dissipative tensor field with its mean removed, and its kernel share against
the measured 19 null directions is reported and must be `<= 20 %`. Pass:

```text
plastic-gauge error of the observable part   <= 5 %
E_twin on the held-out states                <= kernel share + 5 %
```

using the plastic gauge `Gp = (2/3)[[2,1,0],[1,2,0],[0,0,1]]`, never a plain
Euclidean norm.

**Gate 5 — admissibility and dissipation.** Zero points with
`sigma_pred^T Delta eps^p < 0` after `P_H`. Negative-power share of the
assembled increment `<= 12 %` on A3 (previous measurements sat at 8-11 %).
The projection activity (`touched` fraction) is reported at every run: if the
projection rewrites most of the field, the learned direction is a fiction and
that is a finding, not a silent fact.

**Gate 6 — kernel-component stability. This is the decisive falsifier.**
For each seed and both optimisers, decompose the final field against the
measured 19 null directions (the frozen milestone-3 SVD artifact). Registered:

```text
std across runs of ||P_ker Delta eps^p|| / ||Delta eps^p||   <= 5 %
std across runs of the holdout median E                        <= 2 * margin(E)
```

A good `E` with an unstable kernel component means the network has hidden the
identifiability problem, not solved it; the gate fails. This is the
quantitative replacement for the hope that `M_theta ∩ (Delta eps^p + ker A)`
is small — it is measured here, not assumed.

**Gate 7 — the real-DIC comparison.** The decisive test, on the qualified
pipeline and the frozen holdout: A1-A4 compared per rank under the registered
expectations above. No number from the historical `scripts/*_p43.py` is
reusable for it (the elastic-lifting defect is recorded); this script is one
of the two living ones and is the only admissible source.

**Gate 8 — cost.** Reported, no threshold.

## Registered falsifiers

* The gradient sweep plateaus instead of showing a V.
* Gate 4 reaches a low objective with a plastic-gauge error still large: `A`
  is surjective, and fitting a displacement is not evidence about a field.
* Gate 3 finds a singular value at or below the floor at any held-out state.
* Gate 6 instability in either registered quantity.
* A3 does not beat A1 by one margin at any rank, or reaches `E_A3 <= E_A4`
  (a retained error at or below the free-tensor fit is a leak, not a success).
* `P_H` is active on most of the field at the optimum.
* An arm reports no `moved` flag (dead-ReLU trap).

## Valid negative outcomes, pre-declared as results

All three are results, not failures of the campaign:

1. **A3 never beats A1 beyond one margin.** The tensor enrichment is not
   demanded by this observation; the J2 direction is acquitted for P43 under
   this operator.
2. **A3 beats A1, but gate 3 or gate 6 fails.** The displacement data cannot
   determine a tensor plastic rule even with shared structure. The answer is
   an equivalence class modulo `ker A`, reported as such, with the geometry
   named — never a unique field.
3. **The chi/amplitude trade appears** (huge increments nearly orthogonal to
   the stress, admissible but implausible). The family exploits admissibility
   rather than representing plasticity; reported with the trade visible.

## Authorized conclusion wording

On P43, under the measured boundary kinematics and this observation operator,
over a temporal holdout of a single loading history, a shared dissipative
tensor plastic rule **[is / is not] demanded by the data** to predict held-out
increments better than J2, and **[does / does not]** resolve the kernel of the
displacement-to-plastic map.

Nothing else is authorised by this milestone: no material law, no claim about
316L, no uniqueness of the plastic field, no transferability to another ROI,
resolution or test, no window-to-full-field extrapolation.

## Out of scope, deliberately

No reduced integration domain (its reopening condition is registered elsewhere
and nothing here touches it). No `q > 1` partition-of-unity orthogonalisation.
No exhaustive kernel or Airy-family characterisation. No full-field ROI run.
No crystal plasticity. No new synthetic problem. No new network architecture.

## Artifacts

```text
validation/_generated/shared_tensor_generator/
  gates_1_2_3.json, twin_gate4.json, stability_gate6.json,
  comparison_gate7.json, cost_gate8.json
validation/shared_tensor_generator_results.md
```

The run driver is `scripts/learn_flow_direction_p43.py` in the frozen
configuration, plus the gate driver registered as part of this milestone.

## After this

Whatever the verdict, the next design step is written here in advance: if
gates 3 and 6 pass, the generator goes to the full-field qualification path;
if they fail, the missing information source is named (constitutive prior,
temporal coherence, or none available at this resolution) and the equivalence
class is reported as the deliverable.
