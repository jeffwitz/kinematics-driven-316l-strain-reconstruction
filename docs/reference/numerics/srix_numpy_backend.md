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
