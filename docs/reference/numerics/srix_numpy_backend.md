# Native NumPy SRIX backend

This page is the entry point for the native crystal-plasticity backend.  It
explains both the configuration knobs and the reason why several apparently
similar options exist.  If you are discovering the project, read the
sections in order; if you only need a setting, use the tables.

## The short version

The production reference remains MFront/MGIS.  The native backend is an
independent implementation of the qualified Forest--Rubin SRIX law, used when
we need a point-batched NumPy implementation or a future CuPy/GPU path.

There are **two independent choices**:

1. the constitutive implementation (`mfront` or `numpy-srix`);
2. the local plane-stress closure (`nested` or `coupled`).

They must not be conflated:

```text
global FFT/Newton solver
          |
  PlaneStressMaterialBatch
          |
  +-------+----------------+
  |                        |
MFront 3-D law       native NumPy SRIX law
  |                        |
nested closure       nested or coupled closure
```

`nested` is the conservative reference strategy and is the only strategy
available to a generic MFront 3-D behaviour.  `coupled` is a native SRIX
strategy: it solves the twelve slip equations and the three transverse
traction equations in one local Newton problem.  It is mathematically
equivalent to `nested`, but usually faster for large point batches.

The safe defaults are therefore:

```yaml
solver:
  constitutive_backend: mfront-structural-plane-stress
  constitutive_options: {}
```

For an explicit native experiment, select every non-default choice in the
provenance.  The general case configuration currently selects the backend;
the additional native controls are exposed by the native factory and the P43
qualification runner (see the command below).

The exact spelling accepted by the factory is also
`numpy-srix-plane-stress`; it is an alias of `numpy-srix`.

## What the backend computes

The material law is three-dimensional and uses twelve FCC slip systems, cubic
elasticity, the FCC interaction matrix, isotropic saturation `(Q, b)`,
kinematic hardening `(C, d)`, initial resistance `tau0`, and the SRIX
overstress modulus `R`.  Parameter objects come from the central SRIX registry;
the NumPy backend does not define a second parameter table.

The global solver supplies the in-plane Kelvin strain

```text
[xx, yy, xy]
```

and the local closure solves for

```text
[zz, xz, yz]
```

such that all three transverse tractions vanish:

```text
sigma_zz = sigma_xz = sigma_yz = 0.
```

The six-component Kelvin order is `[xx, yy, zz, xy, xz, yz]`.  This is not
the engineering-Voigt order used by the historical J2 backend.  The tangent
returned to the spectral solver is the condensed in-plane algorithmic tangent;
it is not a production finite-difference approximation.

Orientations are point dependent and use the repository's
`Q_global_to_material` convention.  The EBSD assignment convention is **F**
(the spatial map from EBSD pixels to material points).  Internal spectral
arrays remain **C** ordered `(x, y, subcell)`.  F is not a global array-order
switch: applying it to stresses or shadows is a bug.  See
{doc}`../../reference/ebsd_orientation_contract` and
{doc}`../../reference/fcc_interaction_matrix_mapping` for the detailed
contracts.

## Options at a glance

### Backend and local closure

| Option | Values | Default | Meaning |
|---|---|---|---|
| `constitutive_backend` | `mfront-*`, `numpy-srix`, `python` | `mfront` | Selects the material implementation. |
| `plane_stress_solver` | `nested`, `coupled` | `nested` | Selects the local three-traction closure. `coupled` requires native NumPy SRIX. |
| `local_transverse_predictor` | `committed`, `tangent` | `committed` | Initial guess for `[zz,xz,yz]`. `tangent` extrapolates from the local tangent; `committed` is more conservative. |
| `local_tolerance_mpa` | positive float | `1e-8` | Absolute local transverse-stress tolerance, in MPa. |
| `plane_stress_max_iterations` | positive integer | `15` | Iteration limit for the outer nested closure, or the coupled local closure. |

The legacy key `maximum_local_iterations` sets both material and plane-stress
limits.  Prefer the explicit pair below so that a difficult SRIX integration
does not accidentally inherit the much smaller closure limit:

```yaml
constitutive_options:
  material_newton_max_iterations: 100
  plane_stress_max_iterations: 15
```

### Native SRIX integration

These keys are accepted by the native constructor/factory in
`constitutive_options` unless noted otherwise.  The plane-stress controls
(`local_tolerance_mpa`, `local_transverse_predictor` and
`plane_stress_solver`) belong to `local_plane_stress_options` when calling the
factory directly.

| Option | Values/default | Meaning |
|---|---|---|
| `parameter_set` | registry name | Selects one immutable SRIX parameter set. This is a provenance choice, not a fitted value. |
| `parameters` | mapping or `null` | Explicit runtime overrides, normally used for a controlled sensitivity or replay. |
| `crystal_orientation` | identity | `homogeneous` (one Bunge triple) or `ebsd` (one orientation per point). Real EBSD runs should provide the co-registered map. |
| `material_newton_max_iterations` | `100` | Maximum iterations of the native 12-slip constitutive Newton. |
| `local_tolerance` | `1e-11` | Dimensionless residual tolerance of the 3-D native constitutive solve (constructor-level control). |
| `local_linear_solver` | `numpy` | `numpy` uses batched LAPACK; `numba-lu12` uses a specialized point-local LU for one-RHS 12x12 solves. Both are algebraically equivalent. |
| `batch_size` | `null` | Optional chunk size. Bounds Newton workspace memory; it is not a physical discretization parameter. |
| `parallel_backend` | `serial` | `dask-threads` is an experimental prototype around chunks, not the default production path. |
| `dask_workers` | `1` | Number of Dask threaded workers when `parallel_backend=dask-threads`. |

`material_newton_max_iterations` is deliberately separate from
`plane_stress_max_iterations`.  The first controls the SRIX equations; the
second controls the three traction equations.  A failure in either layer must
be reported as a local convergence failure, not hidden by increasing an
unrelated tolerance.

### Coupled performance path

These options matter when `plane_stress_solver=coupled`.

| `coupled_block_solver` | Behaviour | When to use |
|---|---|---|
| `numpy` | NumPy construction of the coupled blocks and batched linear algebra. | Reference implementation and debugging. |
| `numba-fused` | Point-local Numba construction of `A/B`, LU12, Schur 3x3 and the coupled correction. | Small and medium active batches. |
| `numba-fused-state` | Also evaluates the SRIX state in the point-local kernel before the correction. | Large active batches where eliminating batch temporaries wins. |
| `auto` | Chooses between the two Numba paths from `pending.size`. | Recommended native path when the machine has been benchmarked. |

The corresponding `fused_state_threshold` option is an integer number of
pending material points (default `12000`).

`fused_state_threshold` is the switch used by `auto`; the current conservative
default is **12,000 pending points**.  It is deliberately configurable because
the crossover depends on CPU, BLAS, memory bandwidth and system load.  During a
local Newton solve `pending.size` only decreases, so no hysteresis is needed:
the path may switch from `numba-fused-state` to `numba-fused` as points
converge.  Do not present the threshold as a universal performance constant.

The performance options do not change the equations, tolerances, parameters,
or accepted solution.  The NumPy block path remains the oracle for qualifying a
new accelerated kernel.

### Response and transaction contract

Every backend implements the same transactional interface:

```text
evaluate(...)             -> trial response
evaluate_in_plane(...)    -> in-plane response/tangent
complete_trial(...)       -> enrich the latest accepted trial
commit()                  -> trial becomes the new committed state
revert()                  -> discard trial, restore committed state
```

`response_level` controls how much is returned:

| Level | Returned payload | Typical use |
|---|---|---|
| `residual` | stress/residual needed by the global iteration | cheap equilibrium or line-search evaluations |
| `tangent` | stress plus condensed consistent tangent | global Newton/GMRES |
| `complete` | tangent plus elastic/plastic strains, slips and observables | output archiving and field plots |

At the lower-level 3-D native bridge the equivalent switch is
`tangent_mode="none" | "transverse" | "full"`: `none` returns only the state
and stress, `transverse` computes the columns needed by the plane-stress
closure, and `full` computes the complete six-by-six tangent.  The public
plane-stress wrapper normally selects this automatically from `response_level`.

An `evaluate` call never commits plastic state.  A rejected global line search,
cutback or failed closure must be followed by `revert`; only an accepted load
increment calls `commit`.  This is essential when a closure performs several
trial evaluations.

## Choosing a configuration

### 1. Reference and qualification

Use MFront with nested condensation when qualifying a new law or checking the
native implementation:

```yaml
solver:
  constitutive_backend: mfront-3d-condensed-plane-stress
  mfront_behaviour_id: fcc_forest_rubin_srix
  constitutive_options:
    crystal_orientation: {mode: ebsd}
```

Then compare the same load path, orientations, parameters and committed-state
sequence with:

```text
backend                 = numpy-srix
plane_stress_solver     = nested
coupled_block_solver    = numpy
```

In the general `CaseStudyConfig`, the local closure fields are assembled from
the solver's plane-stress settings.  The native P43 runner exposes the full
set of experimental controls as command-line flags; this avoids implying that
every native performance knob is already a stable case-file field.

The expected comparison is on stress, elastic strain, twelve slip fields,
accumulated slip, transverse strains, condensed tangent and final displacement,
not on wall time alone.

### 2. Native production-sized forward

After nested equivalence is established, use the native coupled path explicitly
and retain the nested path as a regression oracle:

```bash
PYTHONPATH=.:src .venv/bin/python \
  scripts/run_p0043_m20_numpy_srix_forward.py \
  --pixels 20 \
  --plane-stress-solver coupled \
  --local-linear-solver numba-lu12 \
  --coupled-block-solver auto \
  --fused-state-threshold 12000 \
  --output validation/reference_data/p0043_m20_numpy_srix_native
```

For a small M20 batch, `numba-fused` is often preferable.  For a large active
batch (M100-scale), `auto` can select `numba-fused-state`.  Benchmark on the
target machine and record the selected option in the JSON provenance.

## How the optimization was obtained

The optimization was deliberately staged so that performance changes could not
silently become changes to the mechanics.

### Step 1 — fix the data contract

The EBSD-to-element mapping was corrected to F order, while the spectral
solver's internal C-order storage was left unchanged.  Shadow sensitivities
were then checked against a full finite-difference oracle.  This separated a
registration/storage bug from a constitutive bug before any acceleration work.

### Step 2 — keep independent reference paths

The validation ladder is:

```text
MFront 3-D + nested closure       external constitutive reference
native NumPy SRIX + nested        independent native implementation
native NumPy SRIX + coupled       faster resolution of the same local problem
```

The coupled path was never allowed to replace nested silently.  On identical
local strains its condensed tangent agrees with the nested tangent to about
`1e-9` or better in the qualification tests, and final displacement fields are
at numerical round-off level.

### Step 3 — remove algebraic work, not physics

The native constitutive Newton was reduced analytically from the six elastic
increment unknowns plus twelve slips to the twelve slips alone.  The elastic
increment is reconstructed exactly.  Fixed-size LU12 kernels, active-point
compression, a transverse predictor and lazy/direct tangent construction then
removed unnecessary allocations and repeated solves.  These changes preserve
the residual equations and are checked against the original NumPy/LAPACK path.

### Step 4 — solve the two local closures together

The nested algorithm first converges SRIX for a guessed transverse strain and
then changes that strain, forcing SRIX to be solved again.  The coupled method
solves the twelve slip residuals and three transverse tractions in one local
Newton system.  It reuses the same SRIX equations and state transaction; only
the driver changes.  This is why the two strategies can be compared directly.

### Step 5 — fuse only where the CPU benefits

The point-local `A/B + LU12 + Schur` kernel removed global `N x 12 x 12`
temporaries and produced the largest CPU gains.  The direct plane-stress
tangent similarly avoids rebuilding a full 3-D tangent before condensation.
Conversely, fusing every state operation into a scalar point loop is slower for
small batches because NumPy's vectorized exponentials and contractions are
already highly optimized.  The final design is therefore hybrid rather than
"everything fused": NumPy for regular batch algebra, Numba for small dense
point-local systems.

### Step 6 — adapt to the active batch size

Benchmarks show that `numba-fused-state` is slower below roughly 8--10k active
points but can be about 20% faster around 10--20k points on the tested machine.
The threshold is consequently conservative (`12_000`) and machine dependent.
The local microbenchmark is the evidence for this switch; a single M100 wall
time is not, because different Newton trajectories can have different numbers
of global iterations.

Representative evidence, stated cautiously:

* the earlier nested M100 reference was about `477 s`;
* successive coupled/fused runs reached roughly `225 s` on some runs;
* the same runs retained indistinguishable fields, but not always the same
  number of global Newton iterations, so those wall times are not one clean
  A/B speedup measurement;
* the local fused-block and tangent benchmarks do show algebraic speedups with
  identical trajectories.

The correct interpretation is: the native coupled path is qualified as an
equivalent high-performance path, while timing claims must always include the
machine, thread settings, warm-up state, global Newton count and GMRES count.

## Benchmark ledger: what was measured and what it means

This section is intentionally explicit.  A timing is useful only when the
algorithm, the requested response, the thread settings and the global Newton
trajectory are known.  The JSON reports named below are the primary evidence;
the numbers in this page are summaries, not a replacement for those reports.

### P43 M20: native SRIX path

The M20 case contains 800 material points (20 x 20 pixels, two subcells) and
uses the corrected F mapping, the same DIC boundary conditions and the same
load path.  The following sequence shows the effect of each structural change.

| Stage | Plane-stress / kernel | Wall time | Global Newton / GMRES | What changed | Interpretation |
|---|---|---:|---:|---|---|
| Nested reference | `nested`, active-point NumPy path | 20.418 s | 121 / 2303 | independent native reference | Baseline for the native M20 implementation. |
| Coupled driver | `coupled`, NumPy blocks | 21.579 s | 146 / 2984 | solve SRIX and the three traction equations together | Same local problem; more global iterations on this run, so not a pure speed comparison. |
| Cached/compact coupled | coupled, cached blocks | 20.010 / 19.645 s | 146 / 2984 | reuse invariant data and reduce temporary arrays | Small, trajectory-preserving implementation gains. |
| Fused coupled block | `numba-fused` A/B + LU12 + Schur | 9.032 s | 119 / 2265 | point-local construction and solve | Large gain from removing `N x 12 x 12` temporaries; the trajectory also changed. |
| Direct tangent | fused block + direct plane-stress tangent | 12.924 s | 146 / 2984 | avoid rebuilding a full 3-D tangent before condensation | The tangent agrees with the oracle; this run is not an A/B timing because its trajectory differs from the 9.032 s run. |
| Fused-state experiment | `numba-fused-state` | 10.786 s | 119 / 2265 | also evaluate state in the point-local kernel | Useful as a large-batch experiment, but slower than vectorized state at M20. |

The M20 table contains deliberately different trajectories.  Consequently,
the wall times are a development history, not one additive speedup claim.  The
strongest conclusions are the field equivalence and the local block/tangent
benchmarks performed at identical states.

The corresponding raw displacement RMS is approximately
`3.952271743e-6 mm` for all qualified native paths.  The coupled-vs-nested
field difference is at round-off level in the qualification reports.  The
older `multi_rhs_dispatch` run (27.640 s) is retained as a profiling artifact,
not as a regression: all phase counters were about 1.35--1.45 times slower on
that machine run, so it is not comparable to the 19--21 s runs.

### P43 M100: scaling evidence

M100 is the useful scaling test because the number of active material points is
large enough for point-local constitutive work to dominate.  The entries below
are separate, fully completed runs.  They must not be combined as if they were
one controlled experiment when the Newton/GMRES counts differ.

| Run | Wall time | Global Newton / GMRES | Main timings reported | Valid conclusion |
|---|---:|---:|---|---|
| Nested reference | 477.091 s | 124 / 3390 | -- | Reference cost of the nested algorithm. |
| First coupled | 379.395 s | 140 / 3926 | -- | Coupled local closure scales better despite more global work. |
| Coupled blocks/Schur optimized | 341.456 s | 140 / 3926 | block 107.56 s; state 37.11 s; tangent 92.71 s | Reduced temporaries and cached blocks preserve the trajectory. |
| Fused block | 224.977 s | 140 / 3926 | block 51.94 s; direct tangent 43.48 s | Strong evidence for the point-local A/B--LU12--Schur kernel on this trajectory. |
| Fused tangent, unfavorable machine run | 243.721 s | 140 / 3926 | block 59.77 s; direct tangent 33.37 s | The tangent is faster, but the wall time is not an A/B result because the unchanged block was 15% slower. |
| Fused-state experiment | 211.687 s | 124 / 3390 | fused state 74.74 s; direct tangent 48.15 s | Indicative only: fewer global iterations; use the local crossover benchmark, not this wall time, to quantify the kernel gain. |

The conservative statement supported by these runs is that the coupled path
and point-local fused kernels remove a substantial amount of constitutive
work.  A single wall-time ratio is not a scientific speedup unless the run is
pre-warmed, interleaved with its control, and has the same global trajectory.
For this reason, the validated status is “equivalent high-performance native
path”, not “a universal 2.12x speedup”.

### MFront comparison: direct P43 references and a separate J2 benchmark

There are two kinds of MFront evidence in the repository, and they answer
different questions.

**Direct P43 SRIX/MGIS references exist.**  The script
`scripts/run_p0043_m20_c_f_forward_identified.py` runs the qualified
`mfront-3d-condensed-plane-stress` behaviour for both element orders on M20.
The M100 script uses the same MFront route for the larger corrected-F and
historical-C forwards.  Their headline results are:

| Direct MFront P43 run | Mesh/path | Wall time | RAW RMS | GMRES | max verification residual |
|---|---|---:|---:|---:|---:|
| M20, C mapping | 20 x 20, 32 steps | 11.828 s | `4.187305e-6 mm` | 2393 | `5.40e-7` |
| M20, F mapping | 20 x 20, 32 steps | 13.228 s | `3.576405e-6 mm` | 2379 | `5.80e-7` |
| M100, C mapping | 100 x 100, 64 steps | 406.528 s | `2.251707e-5 mm` | 9005 | `5.48e-12` |
| M100, F mapping | 100 x 100, 64 steps | 358.237 s elapsed (`326.244 s` solver) | `2.813641e-5 mm` | 7999 | `9.30e-8` |

The corresponding artifacts are
`validation/reference_data/p0043_m20_c_f_forward_identified_v1/report.json`,
`validation/reference_data/p0043_c_m100_forward_identified_v1/report.json`
and
`validation/reference_data/p0043_f_m100_forward_identified_v1/report.json`.
These are genuine MFront SRIX forwards and are therefore suitable references
for checking native fields, stresses and EVM on the same geometry when the
parameters, crop and load path are matched.

The native NumPy M100 scaling table above is a different campaign: it uses a
32-step crop and a different identified parameter/crop provenance.  Its
477--212 s chronology must not be presented as a direct MFront-versus-NumPy
race.  A strict performance comparison requires rerunning MFront and NumPy
with identical inputs and selected options; the existing direct MFront P43
reports are nevertheless the correct constitutive reference artifacts.

**Separate backend context benchmark.**  The independent
`plane_stress_backend_performance_100x100_v1` benchmark is a homogeneous
J2, 100 x 100, 20-increment test (not P43 SRIX and not EBSD).  It gives a
reproducible view of the generic MFront plane-stress routes:

| Backend (J2 100 x 100) | Process wall median | Solver wall median | Constitutive median | Newton | max plane-stress residual |
|---|---:|---:|---:|---:|---:|
| MFront native plane stress, 2 threads | 27.03 s | 25.89 s | 9.52 s | 93 | `9.11e-14` |
| MFront 3-D + external condensation, 2 threads | 83.43 s | 82.30 s | 65.44 s | 93 | `3.75e-08` |
| Historical Python J2 | 134.36 s | 133.31 s | 99.33 s | 183 | -- |

The J2 table shows why a native MFront plane-stress behaviour can be much
faster than repeatedly calling a generic 3-D behaviour.  It is not a
qualification of the SRIX NumPy implementation and must not be used to infer
an MFront-versus-SRIX speedup.

An older constitutive-only benchmark on 200,000 heterogeneous points measured
Python/NumPy at a median of 12.347 s, MFront serial at 13.333 s, and MFront at
eight threads at 3.527 s.  This benchmark is also a separate tabulated-law
context test; it does not include the P43 FFT solve.  The complete reports are
`validation/reference_data/mfront_performance_v1` and
`validation/reference_data/plane_stress_backend_performance_100x100_v1`.

### Isolated solver and parallelism benchmarks

These experiments explain why the implementation uses a hybrid NumPy/Numba
design rather than forcing one library everywhere.

| Experiment | Setup | Result | Lesson |
|---|---|---|---|
| LU12 solve-only | 100,000 random nonsymmetric 12 x 12 systems, one RHS, BLAS pinned to one thread | NumPy/LAPACK 0.214 s; Numba LU12 0.152 s at 1 thread, 0.039 s at 4 threads; relative solution error `1.8e-16` | A fixed-size point-local solve is an excellent Numba target. This does not predict full-forward speed. |
| ThreadPool kernel test | 800 points, four chunks, BLAS one thread | 1 worker 0.077 s; 2 workers 0.073 s; 4 workers 0.105 s | Python threads do not help these small batches; do not extrapolate to large grids. |
| Dask threaded prototype | Dask graph rebuilt inside each constitutive evaluation | 31.968 s serial; 32.805 s (1 worker); 32.308 s (2); 40.568 s (4); 66.069 s (8) | The graph was placed too low in the nested closure. It is not evidence that a coarse chunk-level GPU/CPU design cannot scale. |

The Dask prototype is intentionally not a production option.  A future
experiment would submit one complete SRIX+plane-stress chunk per task, rather
than constructing thousands of tiny synchronized graphs.

### Crossover benchmark for `auto`

The `numba-fused-state` path is not uniformly faster.  On identical warmed-up
local data, the ratio `(fused-state time)/(vectorized-state + fused-block
time)` was:

| Pending points | Ratio, crossover v1 | Ratio, v2/v3 evidence |
|---:|---:|---:|
| 800 | 1.53 | -- |
| 2,000 | 1.77 | -- |
| 5,000 | 1.54 | -- |
| 6,000--10,000 | -- | 1.43--1.38 (v2) |
| 12,000 | -- | 0.65 (v3) |
| 16,000--20,000 | 0.94--0.92 | 0.94--0.92 (v3) |

The exact crossover moved with CPU load and library state.  The robust
engineering choice is therefore a conservative configurable threshold of
`12_000`, not a universal physical constant.  The `auto` path can start with
`numba-fused-state` while many points are pending and switch to
`numba-fused` as the monotone active set shrinks.

### Optimization ledger: mechanism, gain, and proof

The following is the pedagogical map from the old implementation to the
current one.  “Gain” means the effect measured for that isolated change when a
controlled comparison exists; it does not mean that the percentages can be
added.

| Optimization | Why the old path was expensive | What was changed | Evidence / gain | Scientific guard |
|---|---|---|---|---|
| Correct F mapping, preserve C spectral storage | A spatial permutation contaminated the physical fields and shadow path | F only for EBSD-to-material assignment; C remains internal spectral storage | F prior RMS improved from about `4.7247e-6` to `4.0672e-6 mm` before fitting (about -13.9%) | Sentinel mapping and strict re-forward |
| 12-slip reduction | The local Newton solved an 18 x 18 system although six elastic equations can be eliminated exactly | Solve only the 12 slip unknowns and reconstruct elastic strain analytically | Same qualified stress/state/tangent as the 18 x 18 oracle | 18 x 18 path retained as test oracle |
| Active-point compression | Converged points still paid for residual/Jacobian construction | Remove converged points from subsequent local work | Reduced local work; fields unchanged | Compare all committed state arrays |
| Predictor and separate iteration limits | Poor transverse initial guesses and one limit accidentally controlled two Newton layers | Tangent/committed predictor; separate material and closure limits | Fewer closure evaluations and no hidden tolerance change | `revert`/`commit` tests and residual gates |
| Fixed-size LU12 | LAPACK was called for many independent 12 x 12 one-RHS systems | Specialized Numba LU for one-RHS point solves | 0.214 -> 0.152 s in the solve-only benchmark at 1 thread | Relative solve error below `2e-16` |
| Coupled closure | Nested closure fully reconverged SRIX after each transverse correction | Solve 12 slip residuals plus 3 traction residuals together | M100 477.1 -> 379.4 s on separate runs; same local solution, but counts differ | Nested remains the reference |
| Cached blocks / one A factorization | Orientation and elastic blocks were rebuilt; `A` was solved separately for each RHS group | Cache invariants; solve `[R_gamma, B]` together | M100 379.4 -> 341.5 s on the same 140/3926 trajectory | Compare stress, tangent and fields |
| Fused A/B--Schur | Global `N x 12 x 12` and `N x 12 x 3` temporaries caused allocation and memory traffic | Build, factor and apply the local Schur in Numba | Block phase 107.6 -> 51.9 s on the corresponding M100 trajectory; M20 14.51 -> 12.92 s in the direct-tangent A/B run | NumPy/LAPACK path remains oracle |
| Direct plane-stress tangent | Full 3-D tangent was reconstructed and condensed after the coupled solve | Differentiate the coupled 15-variable system directly to obtain `C_PS` | Tangent error about `2e-16`; M100 tangent 43.5 -> 33.4 s in the machine-adjusted comparison | Full tangent retained for verification |
| Fused state for large batches | Vectorized state arrays become costly at very large active batches | Evaluate state and fused correction in one point-local kernel | About 20--22% local gain around 10k--20k points in crossover tests; M20 is slower | `auto` threshold 12,000, not default production |
| Dask/thread experiments | Attempted to distribute small repeated constitutive calls | Tested coarse and fine threaded prototypes | No benefit for the current fine-grained placement; not used in production | No change to equations or transactions |

The stages are not additive: for example, the direct tangent and fused block
share work, and a full forward also includes global FFT/GMRES, line-search and
output construction.  A proper A/B report must therefore include wall time,
warm-up status, BLAS/Numba threads, selected options, global Newton count,
GMRES count, raw RMS and equilibrium residual.

### Reading the benchmark numbers safely

Three rules prevent the most common misinterpretations:

1. **Do not sum phase counters.**  `state`, `line-search`, `tangent` and block
   timers can overlap because a line-search calls the state evaluator.
2. **Do not call a different trajectory a regression.**  A change of one global
   Newton step can dominate a small kernel gain.  Compare local kernels at the
   same state, or interleave old/new complete forwards after warm-up.
3. **Do not transfer J2/MFront numbers to P43 SRIX.**  MFront remains the
   constitutive reference, but the available direct MFront table is a separate
   J2 benchmark.  Native P43 claims are about equivalence and measured scaling,
   not an unperformed MFront SRIX race.

## What to validate before trusting a result

For a new backend or optimization, check in this order:

1. **Local equations:** residuals, stress, hardening, twelve slips and
   dissipation against the reference path.
2. **Plane stress:** the maximum of `|sigma_zz|`, `|sigma_xz|` and
   `|sigma_yz|`, not only `sigma_zz` or an average norm.
3. **Tangent:** condensed tangent against the independent oracle at identical
   strains and states.
4. **Transactions:** `evaluate -> revert`, `evaluate -> commit`, repeated
   trials and failed trials must leave the expected state.
5. **Global solve:** displacement, stress, strain, slip fields, equilibrium
   residual, Newton history and GMRES count.
6. **Performance:** warm-up first; report medians and dispersion; separate
   constitutive time from FFT/GMRES and line-search time.

Do not infer parameter identifiability from a faster forward.  Do not infer a
better physical model from a lower EVM correlation when the authoritative
objective is RAW displacement.  Do not change the SRIX parameters, mapping,
tolerances or defaults merely to improve a benchmark.

## Deliberately out of scope

The first native backend does not implement CuPy, Dask-distributed, MPI, Dask
Array, Numba point-per-voxel state ownership, or a replacement for MFront.
`dask-threads` exists only as a small experimental chunk prototype.  A future
GPU port should reuse the validated equations and transaction contract rather
than become a second, independently tuned material law.

For the mathematical condensation reference, see
{doc}`three_dimensional_condensation`; for SRIX's nonsmooth Jacobian, see
{doc}`srix_semismooth_jacobian`; for a user-facing MFront route, see
{doc}`../../how-to/use_srix_crystal_law`.
