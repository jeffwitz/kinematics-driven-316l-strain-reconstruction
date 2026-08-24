# SRIX/FEMU direct sensitivity — cold-start handoff

Last updated: 2026-08-24  
Branch: `agent/plastic-observability`  
Purpose: **single compact handoff for resuming the SRIX identification work from a new conversation/session without reconstructing the REGM history.**

## Executive state

The low-cost REGM surrogate route is **closed / NO-GO** for four-parameter SRIX identification. The current and only scientifically authorized route is a **direct tangent differentiation of the real M8 FEMU solver**, validated first on a common fixed discrete load path.

No P43 identification, no parameter optimization on experimental data, no new REGM variant, and no analytical MFront parameter derivative is authorized before the direct FEMU sensitivity gate passes on M8.

The target parameter coordinates are

```text
eta = log(tau0, R, Q, b)
```

with the registered SRIX twin preset and local `Hchi = 0` for the first sensitivity gate.

## Why REGM is closed

The reference observed FEMU Jacobian on the exact M8 twin has normalized singular values

```text
FEMU_observed = (1, 0.5415, 0.4067, 0.0679)
condition number ~= 14.7
```

The REGM family loses the two weak but real FEMU directions:

```text
REGM K0 exact                         (1, 0.4220, 0.0324, 4.65e-5)
REGM K_alg                            (1, 0.3759, 0.0347, 8.62e-5)
sequential one correction             (1, 0.5625, 0.0576, 2.30e-4)
sequential cumulative displacement    (1, 0.4646, 0.0938, 2.17e-4)
```

The cumulative endpoint correction fixed a real observable inconsistency, but it did not recover the FEMU geometry. `E-SRIX-REGM-009` therefore closes the surrogate branch.

Important theoretical interpretation: the one-correction REGM script does **not** differentiate the true FEMU residual. It uses the REGM/global weak-discrete operator (`TensorPlasticObservabilityOperator`, `weak_equilibrium_residual`, `_assemble_sparse_stiffness`). Exact constitutive tangents and causal state updates cannot make that different global operator information-equivalent to the FEMU solver.

Do not try to repair REGM further.

## Current authorized gate

The next gate is conceptually:

```text
E-SRIX-FEMU-DIRECT-001
```

Question:

> Does a direct tangent differentiation of the exact M8 FEMU solver, using persistent constitutive shadow histories and the exact matrix-free FEMU tangent action, reproduce a finite-difference FEMU Jacobian computed on the same discrete load path?

Primary comparison is raw columns, **before any SVD**:

```text
relative column L2 error < 2 %
column cosine > 0.999
```

for each of `log(tau0)`, `log(R)`, `log(Q)`, `log(b)`.

Only after those four column gates pass may singular values, Fisher geometry, principal angles, or identification be interpreted.

## Direct sensitivity implementation already present

Primary driver:

```text
scripts/qualify_srix_femu_direct_sensitivity.py
```

Relevant commit:

```text
55726cb — validation: qualify direct FEMU path oracle
```

The direct method uses:

- one converged base FEMU trajectory;
- the exact accepted algorithmic tangent from the two-state M8 forward;
- the true matrix-free global tangent action of `solve_two_state_dirichlet_plane_stress` / `TraditionalTwoStateTriangleBatch`;
- persistent `+/-` constitutive shadow histories for the four log-parameters;
- fixed-current-strain shadow evaluations to build the parameter/history stress forcing;
- one tangent global solve per parameter direction in the current prototype;
- then `epsilon'_j = B U_j`, followed by shadow commits at `epsilon +/- h epsilon'_j` so the causal constitutive history sensitivity is propagated.

The current direct prototype is intentionally semi-analytic: it removes perturbed nonlinear global FEMU solves but still uses local central differences through MFront shadows. This shadow implementation is the oracle for a later fully analytical MFront local kernel.

Do **not** replace the exact FEMU global operator by REGM infrastructure.

## Why the archived FEMU FD is not the primary oracle

The archived adaptive finite difference is not a derivative of one fixed discrete algorithmic path:

```text
base accepted increments : 338
Q+ accepted increments   : 326
Q- accepted increments   : 328
```

Therefore the archived adaptive FD contains changes of the adaptive controller path in addition to parameter sensitivity.

The direct-vs-adaptive V2 column errors

```text
(0.942, 0.967, 0.997, 0.998)
```

must **not** be used as a scientific rejection of the direct sensitivity method.

Reference:

```text
validation/srix_femu_fd_adaptive_path_audit.md
validation/reference_data/srix_femu_direct_sensitivity_v2/
```

## Fixed-path oracle attempts and current numerical blocker

The fixed-path driver is

```text
scripts/qualify_srix_femu_fixed_path_gate.py
```

It freezes the base `LoadPathStep` sequence and gives perturbed solves a generous nonlinear budget (`Newton <= 80`, up to 20 line-search reductions).

Common-path FD attempts have not yet produced a complete oracle:

```text
h = 3e-3   fixed base path failure at increment 18
h = 1e-3   fixed base path failure at increment 5
h = 1e-4   fixed base path failure at increment 12
```

A uniform refinement by two at `h=1e-3` reached 676 prescribed increments and failed at

```text
407 / 676
```

This is recorded as a **numerical fixed-path/oracle robustness blocker**, not as a failure of direct differentiation.

The non-monotone failure location with decreasing `h` argues against the simplistic interpretation “FD step too large”. Diagnose branch/continuation and solver cost instead of shrinking `h` blindly.

## Synchronized common-path gate

Implemented in:

```text
scripts/qualify_srix_femu_common_path_gate.py
```

Commits:

```text
2e244f6 — feat: add synchronized common path sensitivity gate
29f6002 — validation: record synchronized path branch blocker
```

The driver currently:

1. runs the nine adaptive trajectories (`base`, `+/-` for four log-parameters);
2. unions their accepted fractions;
3. replays all variants on one common partition;
4. bisects only failing common intervals;
5. writes a machine-readable report and supports per-trajectory timeout.

Current M8 result:

```text
base ... b_plus : adaptive paths completed
b_minus         : remained in the adaptive solver > 40 minutes
common path     : not yet constructed
FD comparison   : not authorized
SVD             : not authorized
P43             : blocked
```

Artifact:

```text
validation/reference_data/srix_femu_common_path_gate_v1/report.json
validation/srix_femu_common_path_gate_results.md
```

This is a branch/cost diagnostic of the adaptive controller, **not** a negative direct-sensitivity result.

## Next implementation step — planned, not yet implemented

Do **not** wait 40 minutes for `b_minus` to finish an individual adaptive trajectory. Individual adaptive paths are only seed-node generators; they are not part of the final scientific oracle.

The next change should make common-path construction fail-fast and separate three configurations:

```text
SEED_CONFIG
PATH_SEARCH_CONFIG
ORACLE_CONFIG
```

### 1. Persistent per-variant adaptive cache

Cache each completed seed trajectory immediately, with strict provenance:

```text
theta / eta / h
git SHA
MFront library fingerprint
solver config fingerprint
pixels / orientation fingerprint
boundary-history fingerprint
accepted end fractions
timing / diagnostics
```

The already completed eight seed trajectories must not be recomputed when only `b_minus` is missing.

### 2. Missing perturbed adaptive path must not abort the gate

If a perturbed adaptive variant times out and the base path exists:

```text
record timeout
continue
build seed union from all completed adaptive paths
run the common replay with all nine variants anyway
```

`_synchronise()` only needs the available adaptive paths to seed fractions; it must still test **all nine** physical variants on the common path.

Thus the next run should spend essentially zero time waiting for a full standalone `b_minus` adaptive path.

### 3. Fast seed configuration

For seed generation only (never used as FD data), use approximately:

```text
relative_equilibrium_tolerance = 1e-5
verify_final_state = False
maximum_newton_iterations = 12
maximum_line_search_reductions = 6
minimum adaptive increment = 1/1024
maximum cutbacks per step = 3
increment growth factor = 2.0
target difficult Newton count > 10
seed timeout ~= 60 s
```

The current adaptive controller marks **any** reduced line search (`minimum_line_search_factor < 1`) as difficult. For seed generation this is unnecessarily conservative. Preserve the production default, but add a configurable threshold and use roughly

```text
line_search_difficult_threshold = 0.25
```

for `SEED_CONFIG`, so 1/2 and 1/4 steps do not automatically force the next load increment smaller.

This relaxed seed policy is allowed because seed trajectories contribute only candidate path nodes; they never contribute the final FD values.

### 4. Fail-fast path-search configuration

During common-path discovery, keep the physical equilibrium tolerance strict but stop defending a bad interval for dozens of nonlinear iterations:

```text
relative_equilibrium_tolerance = 1e-6
verify_final_state = False
maximum_newton_iterations = 12
maximum_line_search_reductions = 6
adaptive stepping = False
per-variant search budget ~= 120 s
```

Interpretation:

> if the imposed common interval cannot converge reasonably in 12 Newton steps and a line search down to about 1/64, bisect the interval instead of spending hundreds of MFront evaluations on it.

Start common-path search with the known difficult direction:

```text
b_minus, b_plus, Q_minus, Q_plus, R_minus, R_plus, tau0_minus, tau0_plus, base
```

A suggested partition may be built from the eight available adaptive seed paths; `b_minus` then discovers any missing local nodes through synchronized bisection.

### 5. Limit local pathological refinement

Suggested search stop guard:

```text
max local bisections for one interval ~= 10
or minimum common interval ~= 1/65536
```

If reached, record `blocked_local_branch` with the exact load interval and direction instead of refining indefinitely.

### 6. Strict oracle replay only once

After a candidate common path is found, replay all nine variants once using the strict oracle configuration:

```text
relative_equilibrium_tolerance = 1e-6
verify_final_state = True
maximum_newton_iterations = 80
maximum_line_search_reductions = 20
adaptive_stepping_enabled = False
load_path_override = common_path
```

Only this strict replay may create the finite-difference oracle used for qualification.

If the strict replay fails at one interval, return that interval to the fast path-search/bisection stage and then repeat the strict replay.

### 7. Optional later optimization, not first priority

If common-path bisection still becomes expensive late in the trajectory, add exact restart/checkpoint support after accepted increments so a failure near step 407 does not require recomputing steps 1...406.

A second optional optimization is to initialize perturbed Newton solves with the tangent predictor

```text
u_plus  ~= u_base + h U_j
u_minus ~= u_base - h U_j
```

from the direct sensitivity. This is only an initial Newton guess; it must not alter the converged FD result.

Do not implement these two optimizations before testing whether fail-fast search + seed caching already resolves the cost problem.

## Solver/controller facts relevant to the cost problem

Current adaptive logic in `AdaptiveLoadStepController.accept()` marks a converged step as difficult if, among other conditions,

```text
newton_iterations > target_newton_iterations_max
minimum_line_search_factor < 1.0
```

and then reduces the next increment. This is conservative for production but can generate many micro-steps in a seed-only run.

The fixed-path oracle currently raises the nonlinear budget to `80` Newton iterations and `20` line-search reductions. That generosity belongs in the **final oracle replay**, not in path discovery.

Do not gain time by loosening SRIX constitutive equations, plane-stress closure, cubic elasticity, FCC slip systems, EBSD orientations, or the experimental boundary history.

## SRIX / MFront facts that must not be rediscovered incorrectly

Production source:

```text
mfront/Fcc316LForestRubinSrix.mfront
```

Important points:

- `@DSL Implicit`, Newton-Raphson, `@Theta 1`;
- cubic elasticity through `StandardElasticity`;
- 12 FCC slip systems;
- primary runtime parameters include `tau0`, `R`, `Q`, `b`;
- `Hchi` is a material property for the non-local micromorphic extension; use local `Hchi=0` for the first sensitivity gate;
- state/history includes elastic strain plus per-slip `g`, `p`, `a`;
- the existing implementation intentionally constructs `Deq` from the local implicit increments (`deel`, `dg`) so the consistent tangent contains the required rank-one term;
- semi-smooth conventions at `abs(dg)==0` and at the overstress threshold must be preserved exactly.

MFront/MGIS already provides the consistent constitutive tangent

```text
dsigma / depsilon
```

through the implicit Jacobian and tangent operator.

However, the standard documented MGIS API does **not** automatically expose

```text
dsigma / d(tau0, R, Q, b)
```

merely because these are `@Parameter`, nor does it expose the complete local implicit Newton Jacobian as a public state API.

Do not treat `manager.K` as the local Newton Jacobian; it stores tangent-operator blocks.

The fully analytical local SRIX sensitivity kernel is a **phase-2 optimization only after the shadow/direct global gate passes**. The safest final local formulation is directional and can reduce the local implicit solve to a 12x12 Schur system with four RHS. Do not create a brittle hack against private generated C++ symbols.

## Key code to read for direct sensitivity work

Read in this order:

1. `validation/srix_femu_direct_cold_start.md` — this file;
2. `validation/srix_regm_worklog.md` — historical gate record and evidence;
3. `scripts/qualify_srix_femu_common_path_gate.py`;
4. `scripts/qualify_srix_femu_fixed_path_gate.py`;
5. `scripts/qualify_srix_femu_direct_sensitivity.py`;
6. `src/fem_inhouse/spectral2d/newton_two_state.py`;
7. `src/fem_inhouse/spectral2d/step_control.py`;
8. `src/fem_inhouse/core/plane_stress_material.py` and the SRIX MFront batch/condensation bridge;
9. `mfront/Fcc316LForestRubinSrix.mfront`;
10. `validation/reference_data/srix_regm_information_geometry_v1/report.json` for the archived FEMU geometry.

Do not start from old TANN or old REGM scripts when the task is direct SRIX/FEMU sensitivity.

## Environment

Use the repository `.venv` directly.

For MFront/MGIS tests:

```text
source /home/jeff/.local/share/tfel/env/env.sh
MFRONT_BEHAVIOUR_LIBRARY=$PWD/build/mfront/src/libBehaviour.so
SRIX_GENERIC_MFRONT_BEHAVIOUR_LIBRARY=$PWD/build/srix-generic/src/libBehaviour.so
```

The two libraries are distinct.

## Current test / branch status at this handoff

At the latest recorded synchronized-path blocker:

```text
branch: agent/plastic-observability
ruff: green
pytest: 137 passed, 1 skipped (pyFFTW)
working tree: reported clean
latest blocker artifact commit: 29f6002
```

The synchronized driver commit immediately before it is `2e244f6`.

## Hard stop rules

Until `E-SRIX-FEMU-DIRECT-001` passes on a strict common-path FD oracle:

- **NO P43 identification**;
- **NO optimization of SRIX parameters on experimental data**;
- **NO new REGM surrogate variant**;
- **NO interpretation of the old adaptive FD as the primary derivative oracle**;
- **NO SVD/Fisher conclusion from a non-common-path comparison**;
- **NO analytical MFront parameter-derivative project before the shadow/direct method itself is validated**;
- **NO relaxation of the final oracle physics/accuracy merely to make the gate pass**.

The next useful implementation is therefore operational, not conceptual: cache completed seed paths, allow missing perturbed adaptive paths, build the common seed from available paths, and make path discovery fail-fast so `b_minus` causes local bisection instead of a 40-minute standalone adaptive run.
