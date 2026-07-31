# Micromorphic identification under symmetric observation — preregistration

Date: 2026-07-31
Authorised by the gate decision of 2026-07-30, recorded in `Claude.md`, which
opened micromorphic identification after lots V2 and V3 completed, under three
conditions restated and honoured below.

**Status: written and costed, not executed.** No run is launched by this
document.

## Why the existing tooling cannot be reused as is

`configs/joint_nonlocal_identification_p0043.yaml` already defines a 21 by 21
grid over `ell` and `alpha` with a fidelity ladder. Its objective is **not**
admissible under the gate, and this must be stated plainly before anything is
run:

- `identification/observation.py` declares `spatial_filter: Literal["none"]`,
  so no observation operator is applied to the FEM side at all;
- the only transformations available are `grid_reduction` and core masking.

That is the raw FEM against image-observed DIC comparison which lot V3 showed
changes amplitude, morphology **and the ranking of coupling candidates**. Reusing
it would reproduce exactly the error that caused identification to be suspended.

### The consequence for the fidelity ladder

The symmetric operator warps a displacement field onto the reference image and
re-observes it through DISFlow. That is only defined when **one element is one
pixel**. The configured `spatial_reduction: 2` low-fidelity tier is therefore
incompatible with the registered objective and is **not used**.

Temporal reduction remains available, and is the only reduction admitted here.

## Registered design

### Fidelity

Full spatial resolution throughout, `660 x 610` elements on P43 with padding
150, so the image operator is valid at every grid point. Increments fixed at
**20**, matching the archived coupled campaigns so their three points are
directly comparable and need not be recomputed.

Justification for not going below 20: the discretisation sensitivity measured in
`dic_multistep_p0043_path_dependence_results.md` is `0.20 %` on core PEEQ
between 20 and 40 increments. That bounds the error of using 20 rather than 40.
It does **not** bound the error of using 10, which has not been measured, so 10
is not used.

### Grid

| Parameter | Values |
|---|---|
| `ell` (µm) | 20, 30, 40, 50, 58.88 |
| `alpha` | 1, 2, 3, 4, 6 |

25 points, of which `(alpha, ell) = (1, 58.88)`, `(2, 58.88)` and `(4, 58.88)`
are archived and already observed symmetrically. **22 new runs.**

### Objective

Each run is warped and re-observed through DISFlow with `legacy_script_2021`
primary and `declared_medium_v4` as sensitivity, then scored on the core with
padding excluded, using the metrics and the significance margins already
registered in
`dic_multistep_p0043_observed_path_comparison_preregistration.md`:
relative L2 `0.0202`, Pearson `0.0185`, top-10 % IoU `0.0189`, absolute-q90 IoU
`0.0217`.

**No single scalar score is used to select a point.** The archived symmetric
result already shows the ranking depends on the objective: `alpha = 4` is best
on relative L2 and Pearson, `alpha = 1` on q90 IoU, and the local model on
top-10 % IoU. Collapsing that into one number would hide the disagreement and
manufacture an identification.

The registered deliverable is instead:

1. the four metric surfaces over the grid;
2. the **Pareto-non-dominated set** across the four;
3. the per-objective optimum, reported separately.

### The question this campaign is actually for

Not "which `alpha` is best". The scientifically decisive question is whether
`H_chi` and `ell` are **separately identifiable**, or whether the objective is
degenerate along `A_chi = H_chi * ell**2`.

Registered test: fit the orientation of the objective valley in
`(log H_chi, log ell)` and compare it with the constant-`A_chi` direction.

| Outcome | Conclusion |
|---|---|
| valley clearly not aligned with constant `A_chi`, on the majority of metrics | the two parameters are separately identifiable in this observation |
| valley aligned with constant `A_chi` within the metric margins | **only the product is identifiable**; no separate `ell` may be claimed |
| Pareto set spans more than half the grid | identification fails at this resolution; report and stop |

The third outcome is considered likely and is not a failure of execution.

## Conditions of the gate, honoured

1. **Symmetric objective only.** The raw objective is computed and archived as a
   known-biased control, never used for selection.
2. **Margins preregistered**, taken from the measured DIC-noise sensitivity
   intervals, not chosen here.
3. **The 16 % path systematic is cited.** All runs use the proportional path, so
   the systematic measured in the path-dependence campaign applies uniformly. It
   is not applied as a correction, and because it is uniform across the grid it
   should not move the ranking; that assumption is stated, not verified.

## Cost estimate

Derived from the archived coupled P43 timings at 20 increments and full
resolution: `26:40`, `29:56` and `36:14`, so about `31 min` per run.

| Item | Count | Estimate |
|---|---:|---:|
| new coupled runs | 22 | `11.4 h` |
| symmetric replay per point | 25 | negligible, seconds on `661 x 611` |
| archived points reused | 3 | none |
| **total** | | **about 11 to 12 h** |

This is an overnight campaign on this machine. Cost grows with the square of
grid refinement, so a finer grid needs a decision, not an assumption.

## What this campaign cannot deliver

- **no transferable material internal length.** That needs unchanged-parameter
  transfer to another ROI, another observation resolution and ideally another
  test, none of which is in scope;
- **no prediction claim.** The boundary conditions remain measured;
- no resolution of the objective-dependence itself. If the Pareto set is wide,
  the honest output is that the present observable does not identify the pair,
  and that a different observable is required.

## Deliverable

`validation/micromorphic_symmetric_identification_results.md` and
`reference_data/micromorphic_symmetric_identification_p0043_v1/`.
